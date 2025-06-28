import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import linen as nn
from einops import rearrange

import tiktoken
from utils import modelConfig
from typing import Optional, Tuple, List
from jaxtyping import Array, PyTree
from functools import partial


class Embeddings(nn.Module):
    model_dimension: int
    vocab_size: int
    model_dtype: jnp.dtype

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dimension,
            dtype=self.model_dtype,
        )
        self.layer_norm = nn.LayerNorm()

    def __call__(self, x: Array, out: bool = False) -> Array:
        if not out:
            x = self.embedding(x)
            if self.is_mutable_collection("params"):
                _ = self.layer_norm(x)
        else:
            x = self.embedding.attend(self.layer_norm(x))

        return x


class NoisyKGate(nn.Module):
    model_dimension: int
    n_experts: int
    k: int
    model_dtype: jnp.dtype

    def setup(self):
        self.centroids = nn.Dense(features=self.n_experts, dtype=self.model_dtype)

    def top(self, x: Array) -> Tuple[Array, Array]:
        assert x.shape[0] == self.n_experts, "x must be of shape (n_experts, )"
        g_i, i = jax.lax.top_k(x, self.k)
        g = jnp.zeros((x.shape[0],), dtype=x.dtype)
        g = g.at[i].set(g_i)
        g = g / jnp.sum(g, axis=-1)
        g = g[i]

        return g, i

    def __call__(self, x: Array) -> Tuple[Array, Array, Array]:
        s = nn.sigmoid(self.centroids(x))
        g_scores, indices = jnp.apply_along_axis(func1d=self.top, axis=-1, arr=s)
        # s = s / jnp.sum(s, axis=-1, keepdims=True)

        return g_scores, indices, s


class MoE(nn.Module):
    model_dimension: int
    n_shared: int
    n_experts: int
    k: int
    dropout: float
    model_dtype: jnp.dtype
    grad_checkpoint: bool

    def setup(self):
        self.shared = nn.Dense(
            features=self.model_dimension * self.n_shared,
        )
        self.experts = [
            FeedForward(
                model_dimension=self.model_dimension,
                ff_dim=4 * self.model_dimension,
                dropout=self.dropout,
                model_dtype=self.model_dtype,
                grad_checkpoint=self.grad_checkpoint,
                name=f"expert_{i}",
            )
            for i in range(self.n_experts)
        ]

        self.gate = NoisyKGate(
            model_dimension=self.model_dimension,
            n_experts=self.n_experts,
            k=self.k,
            model_dtype=self.model_dtype,
        )

    def get_gScores(self, scores: Array, indices: Array, x: Array, train: bool = True):
        def head_fn(i):
            return lambda mdl, x: mdl.experts[i](x, train=train)

        expert_lambda = [head_fn(i) for i in range(self.n_experts)]

        if self.is_mutable_collection("params"):
            for expert_ffn in expert_lambda:
                _ = expert_ffn(self, x)

            return jnp.zeros_like(x)

        expert_fn = lambda j, experts, x: nn.switch(j, expert_lambda, self, x)
        expert_parallel = jax.vmap(fun=expert_fn, in_axes=(0, None, None), out_axes=(0))

        expert_scores = expert_parallel(indices, self.experts, x)  # (K) -> (K, C)
        gScore = scores[:, None] * expert_scores  # (K, 1), (K, C) -> (K, C)
        gScore = jnp.sum(gScore, axis=0)  # (K, C) -> C

        return gScore

    def __call__(self, x, train=True):
        B, T, C = x.shape

        res_shared = self.shared(x)
        res_shared = rearrange(
            res_shared, "B T (n d) -> B T n d", n=self.n_shared, d=self.model_dimension
        )
        res_shared = jnp.sum(res_shared, axis=2)  # (B, T, n, d) -> (B, T, d)

        g, i, s = self.gate(x)

        g_route = jnp.reshape(g, (B * T, -1))
        i_route = jnp.reshape(i, (B * T, -1))
        x_route = jnp.reshape(x, (B * T, C))

        gscore_parallel = jax.vmap(
            fun=lambda g, i, x: self.get_gScores(g, i, x, train=train),
            in_axes=(0, 0, 0),
            out_axes=(0),
        )

        res_route = gscore_parallel(g_route, i_route, x_route)
        res_route = jnp.reshape(res_route, (B, T, C))

        res = x + res_shared + res_route

        f = jnp.zeros((B, self.n_experts), dtype=jnp.float32)
        p = jnp.zeros((B, self.n_experts), dtype=jnp.float32)

        s = s / jnp.sum(s, axis=-1, keepdims=True)
        s = jnp.take_along_axis(s, i, axis=-1)
        s = jnp.reshape(s, (B, -1))

        i = i.reshape(B, -1)

        for h in range(self.n_experts):
            load_i = jnp.where(i == h, 0, 1)

            f_i = jnp.sum(load_i, axis=-1)
            f = f.at[:, h].set(f_i + f[:, h])

            p_i = jnp.sum(s * load_i, axis=-1)
            p = p.at[:, h].set(p_i + p[:, h])

        return res, (f, p)


class FFBody(nn.Module):
    model_dimension: int
    ff_dim: int
    dropout: float
    model_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(
            features=self.ff_dim,
            dtype=self.model_dtype,
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            features=self.model_dimension,
            dtype=self.model_dtype,
        )(x)

        return x


class FeedForward(nn.Module):
    model_dimension: int
    ff_dim: int
    dropout: float
    model_dtype: jnp.dtype
    grad_checkpoint: bool

    @nn.compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        ff = FFBody
        if self.grad_checkpoint:
            ff = nn.remat(FFBody)

        ff = ff(
            model_dimension=self.model_dimension,
            ff_dim=4 * self.model_dimension,
            dropout=self.dropout,
            model_dtype=self.model_dtype,
        )
        x_ff = nn.Dropout(rate=self.dropout)(ff(x), deterministic=not train)

        return x_ff


class RoPE:
    def __init__(self, T: int, model_dim: int):
        assert model_dim % 2 == 0, "model_dim must be even"

        freq = jnp.arange(T, dtype=jnp.float32)[:, None] + 1

        pos = jnp.arange(model_dim // 2, dtype=jnp.float32)[:, None]
        pos = pos.repeat(2, axis=-1).reshape(1, -1)
        log_theta_base = jnp.log(10000.0)
        theta = jnp.exp(-2 * pos / model_dim * log_theta_base)

        self.cos = jnp.cos(freq * theta)
        self.sin = jnp.sin(freq * theta)

    def __call__(
        self,
        x: Array,
        t_start: int,
        offset: Optional[int] = None,
        transpose: bool = False,
    ) -> Array:
        B, T, C = x.shape
        x_proj = x.astype("float32")

        if offset is None:
            offset = T

        cos_rope = x_proj * self.cos[t_start : t_start + offset, :]

        x_inter = x_proj.reshape((B, T, C // 2, 2))
        x_inter = jnp.flip(x_inter, axis=-1) * jnp.array([-1, 1])
        x_inter = x_inter.reshape((B, T, C))
        if transpose:
            x_inter *= -1
        sin_rope = x_inter * self.sin[t_start : t_start + offset, :]

        x_rope = cos_rope + sin_rope
        x = x_rope.astype(x.dtype)
        return x


class MLA(nn.Module):
    model_dim: int
    n_heads: int
    T: int
    latent_dim: int
    dhR: int
    model_dtype: jnp.dtype
    grad_checkpoint: bool
    dropout: float = 0.0

    def setup(self):
        self.W_down = nn.Dense(features=2 * self.latent_dim, dtype=self.model_dtype)

        self.W_uKV = nn.Dense(features=2 * self.model_dim, dtype=self.model_dtype)
        self.W_uQ = nn.Dense(features=self.model_dim, dtype=self.model_dtype)

        self.dk = self.model_dim // self.n_heads

        if self.grad_checkpoint:
            self.output = nn.remat(nn.Dense)(
                features=self.model_dim, dtype=self.model_dtype
            )
        else:
            self.output = nn.Dense(features=self.model_dim, dtype=self.model_dtype)
        self.out_dropout = nn.Dropout(rate=self.dropout)

        self.rope = False
        if self.dhR != 0:
            self.rope = True
            self.Wkr = nn.Dense(features=self.dhR, dtype=self.model_dtype)
            self.Wqr = nn.Dense(
                features=(self.dhR * self.n_heads), dtype=self.model_dtype
            )
            self.rope_k = RoPE(model_dim=self.dhR, T=self.T)
            self.rope_q = RoPE(model_dim=self.dhR * self.n_heads, T=self.T)

    def __call__(
        self,
        x: Array,
        *,
        cKV_cache: Optional[Array] = None,
        kRT_cache: Optional[Array] = None,
        train=True,
    ) -> Tuple[Array, Tuple[Array, Array]]:
        B, T, C = x.shape

        cKVt, cqt = jnp.split(self.W_down(x), 2, axis=-1)

        if self.rope:
            t_start = 0
            if cKV_cache is not None:
                t_start = cKV_cache.shape[1]

            kRt = self.rope_k(self.Wkr(x), t_start)

            qRt = self.rope_q(self.Wqr(x), t_start)
            qRt = rearrange(qRt, "B T (nh d) -> B nh T d", nh=self.n_heads, d=self.dhR)

        if not train:
            if cKV_cache is not None:
                cKVt = jnp.concatenate([cKV_cache, cKVt], axis=1)
            cKV_cache = cKVt

            if self.rope:
                if kRT_cache is not None:
                    kRt = jnp.concatenate([kRT_cache, kRt], axis=1)
                kRT_cache = kRt

        k, v = jnp.split(self.W_uKV(cKVt), 2, axis=-1)
        q = self.W_uQ(cqt)

        k = rearrange(k, "B T (nh d) -> B nh T d", nh=self.n_heads, d=self.dk)
        q = rearrange(q, "B T (nh d) -> B nh T d", nh=self.n_heads, d=self.dk)
        v = rearrange(v, "B T (nh d) -> B nh T d", nh=self.n_heads, d=self.dk)

        if self.rope:
            q = jnp.concatenate([q, qRt], axis=-1)
            kRt = jnp.repeat(kRt[:, None, :, :], self.n_heads, axis=1)
            k = jnp.concatenate([k, kRt], axis=-1)

        def scaledDotProd(q, k, v, mask):
            q = q.astype("float32")
            k = k.astype("float32")
            w = jnp.einsum("B n T d, B n t d -> B n T t", q, k) * (1 / (self.dk**0.5))
            w = jnp.where(mask == 0, -9e15, w)
            w = jax.nn.softmax(w, axis=-1).astype(self.model_dtype)
            output = jnp.einsum("B n T t, B n t d -> B n T d", w, v)
            return output

        if self.grad_checkpoint:
            scaledDotProd = jax.remat(scaledDotProd)

        if T == 1:
            mask = jnp.ones((B, self.n_heads, 1, k.shape[2]))
        else:
            mask = jnp.tril(
                jnp.ones((B, self.n_heads, q.shape[2], k.shape[2])),
            )

        output = scaledDotProd(q, k, v, mask)
        output = rearrange(output, "B nh T dk -> B T (nh dk)")

        output = self.output(output)
        output = self.out_dropout(output, deterministic=not train)
        return output, (cKV_cache, kRT_cache)


class Block(nn.Module):
    model_dimension: int
    n_heads: int
    dropout: float
    T: int
    latent_dim: int
    model_dtype: jnp.dtype
    dhR: int = 0
    n_shared: int = 0
    n_experts: int = 0
    k: int = 0
    moe: bool = False
    grad_checkpoint: bool = False

    @nn.compact
    def __call__(
        self,
        x: Array,
        cache: Optional[Tuple[Array, Optional[Array]]] = (None, None),
        train: bool = True,
    ):
        x_norm = nn.LayerNorm()(x)

        x_up, cache = MLA(
            model_dim=self.model_dimension,
            n_heads=self.n_heads,
            T=self.T,
            latent_dim=self.latent_dim,
            dhR=self.dhR,
            model_dtype=self.model_dtype,
            dropout=self.dropout,
            grad_checkpoint=self.grad_checkpoint,
        )(x_norm, cKV_cache=cache[0], kRT_cache=cache[1], train=train)
        x = x + x_up

        x_norm = nn.LayerNorm()(x)

        load = None
        if self.moe == True:
            x_ff, load = MoE(
                model_dimension=self.model_dimension,
                n_experts=self.n_experts,
                k=self.k,
                dropout=self.dropout,
                model_dtype=self.model_dtype,
                n_shared=self.n_shared,
                grad_checkpoint=self.grad_checkpoint,
            )(x_norm, train=train)

        else:
            x_ff = FeedForward(
                model_dimension=self.model_dimension,
                ff_dim=4 * self.model_dimension,
                dropout=self.dropout,
                model_dtype=self.model_dtype,
                grad_checkpoint=self.grad_checkpoint,
            )(x_norm, train=train)

        x = x + x_ff

        return x, (cache, load)


class EncoderBlock(nn.Module):
    model_dimension: int
    n_heads: int
    dropout: float
    T: int
    latent_dim: int
    model_dtype: jnp.dtype
    dhR: int
    dhR_blocks: int = 4
    n_shared: int = 0
    n_experts: int = 0
    k: int = 0
    moe: bool = False
    grad_checkpoint: bool = False

    @nn.compact
    def __call__(
        self,
        x: Array,
        cache: Optional[List[Tuple[Array, Optional[Array]]]] = None,
        train: Array = True,
    ) -> Tuple[
        Array,
        Tuple[
            Optional[List[Tuple[Optional[Array], Optional[Array]]]], Optional[PyTree]
        ],
    ]:
        cKV_cache = []
        kRT_cache = []
        load = None
        for i in range(self.dhR_blocks):
            if cache is None:
                layer_cache = (None, None)
            else:
                cKV = cache[0][i]
                kRT = cache[1][i] if i < self.dhR_blocks - 1 else None
                layer_cache = (cKV, kRT)

            x, (current_cache, current_load) = Block(
                model_dimension=self.model_dimension,
                n_heads=self.n_heads,
                dropout=self.dropout,
                T=self.T,
                latent_dim=self.latent_dim,
                dhR=self.dhR if (i < self.dhR_blocks - 1) else 0,
                moe=self.moe,
                n_experts=self.n_experts,
                n_shared=self.n_shared,
                k=self.k,
                model_dtype=self.model_dtype,
            )(x, cache=layer_cache, train=train)
            if load is None:
                load = current_load
            else:
                add_tree = lambda x, y: jax.tree.map(lambda a, b: a + b, x, y)
                load = (
                    add_tree(load[0], current_load[0]),
                    add_tree(load[1], current_load[1]),
                )

            cKV_cache.append(current_cache[0])
            if i < self.dhR_blocks - 1:
                kRT_cache.append(current_cache[1])

        if train:
            out_cache = None
        else:
            out_cache = (
                jnp.stack(cKV_cache, axis=0),
                jnp.stack(kRT_cache, axis=0) if self.dhR_blocks > 1 else None,
            )

        return x, (out_cache, load)


class Decoder(nn.Module):
    model_dimension: int
    n_heads: int
    dhR: int
    dhR_blocks: int
    T: int
    vocab_size: int
    dropout: float
    blocks: int
    n_experts: int
    n_shared: int
    k: int
    moe: bool
    latent_dim: int
    model_dtype: jnp.dtype
    grad_checkpoint: bool

    @nn.compact
    def __call__(
        self,
        x: Array,
        cache: Optional[List[Tuple[Optional[Array], Optional[Array]]]] = None,
        train: bool = True,
    ) -> Tuple[
        Array, Tuple[Optional[List[Tuple[Optional[Array], Optional[Array]]]], Array]
    ]:
        if cache is not None:
            x = x[:, -1:]

        embed = Embeddings(
            model_dimension=self.model_dimension,
            vocab_size=self.vocab_size,
            model_dtype=self.model_dtype,
        )
        x = embed(x)
        cKV_cache = []
        kRT_cache = []
        load = None
        for i in range(self.blocks):
            if cache is None:
                layer_cache = None
            else:
                cKV = cache[0][i]
                kRT = cache[1][i] if cache[1][i] is not None else None
                layer_cache = (cKV, kRT)

            x, (current_cache, current_load) = EncoderBlock(
                model_dimension=self.model_dimension,
                n_heads=self.n_heads,
                dropout=self.dropout,
                T=self.T,
                latent_dim=self.latent_dim,
                dhR=self.dhR,
                dhR_blocks=self.dhR_blocks,
                moe=self.moe,
                n_experts=self.n_experts,
                n_shared=self.n_shared,
                k=self.k,
                model_dtype=self.model_dtype,
            )(x, cache=layer_cache, train=train)

            if load is None:
                load = current_load
            else:
                add_tree = lambda x, y: jax.tree.map(lambda a, b: a + b, x, y)
                load = (
                    add_tree(load[0], current_load[0]),
                    add_tree(load[1], current_load[1]),
                )

            cKV_cache.append(current_cache[0])
            kRT_cache.append(current_cache[1])

        out_cache = (
            jnp.stack(cKV_cache, axis=0),
            jnp.stack(kRT_cache, axis=0) if self.dhR_blocks > 1 else None,
        )

        x = embed(x, out=True)

        if load is not None:
            load = load[0] * load[1]

        return x, (out_cache, load)

    def generate(
        self,
        params: PyTree,
        key: jax.random.key,
        x: str = "",
        *,
        B: int = 1,
        k: int = 10000,
        temperature: int = 1,
        max_tokens: int = 100,
        use_cache=True,
    ) -> List[str]:
        enc = tiktoken.get_encoding("gpt2")

        out = jnp.array([enc._special_tokens["<|endoftext|>"]], dtype=jnp.int32)
        if x != "":
            x_encode = jnp.array(enc.encode(x), dtype=jnp.int32)
            out = jnp.concatenate([out, x_encode], axis=-1)

        prompt_length = out.shape[0]
        out = jnp.repeat(out[None, :], B, axis=0)
        cache = None

        def sample(key, params, inp, cache, B, k, temperature):
            if not use_cache:
                cache = None
            logits, (cache, _) = self.apply(
                {"params": params}, inp, cache=cache, train=False
            )

            print(f"logits {logits.shape} | {cache[0].shape} | {cache[1].shape}")

            logits, idx = jax.lax.top_k(logits[:, -1, :], k=k)
            logits /= temperature

            out_next_idx = jax.random.categorical(key, logits, axis=-1, shape=(B,))
            out_next = idx[jnp.arange(B, dtype=jnp.int32), out_next_idx][:, None]

            return out_next, (cache, logits)

        for _ in range(min(max_tokens, self.T - prompt_length)):
            key, sample_key = jax.random.split(key)
            out_next, (cache, logits) = sample(
                sample_key, params, out, cache, B, k, temperature
            )
            out = jnp.concatenate([out, out_next], axis=-1)

        tokens = jax.device_get(out[:, 1:])
        outputs = list(map(lambda x: enc.decode(x), tokens))

        return outputs

    @classmethod
    def get_model(
        cls: "Decoder", model_config: modelConfig, init_key: jax.random.key
    ) -> Tuple["Decoder", PyTree]:
        x = jnp.ones((1, model_config.T), dtype=jnp.int32)

        model = cls(
            model_dimension=model_config.model_dimension,
            n_heads=model_config.n_heads,
            dhR=model_config.dhR,
            dhR_blocks=model_config.dhR_blocks,
            T=model_config.T,
            vocab_size=model_config.vocab_size,
            dropout=model_config.dropout,
            blocks=model_config.blocks,
            n_experts=model_config.n_experts,
            k=model_config.k,
            moe=model_config.moe,
            latent_dim=model_config.latent_dim,
            n_shared=model_config.n_shared,
            model_dtype=jnp.bfloat16
            if (model_config.model_dtype == "bfloat16")
            else jnp.float32,
            grad_checkpoint=model_config.grad_checkpoint,
        )

        params = model.init(
            init_key,
            x,
            train=False,
        )["params"]

        _ = model.generate(
            params,
            init_key,
            x="hello",
            B=1,
            k=model_config.vocab_size,
            temperature=1,
            max_tokens=10,
        )

        return model, params


class shardedModel:
    @staticmethod
    def generate(
        model: Tuple[Embeddings, EncoderBlock],
        params: PyTree,
        key: jax.random.key,
        mesh: jax.sharding.Mesh,
    ) -> List[str]:
        raise NotImplementedError("have to do")

    @staticmethod
    def shard_params(
        params: Tuple[PyTree, PyTree],
        mesh: jax.sharding.Mesh,
    ) -> Tuple[PyTree, PyTree]:
        embed_sharding = jax.tree.map(
            lambda _: jax.sharding.NamedSharding(mesh, P()), params[0]
        )
        layer_sharding = jax.tree.map(
            lambda _: jax.sharding.NamedSharding(mesh, P("model")), params[1]
        )
        params_sharding = (embed_sharding, layer_sharding)
        params = jax.device_put(params, params_sharding)

        return params

    @staticmethod
    def get_model(cfg):
        dtype = jnp.bfloat16 if (cfg.model_dtype == "bfloat16") else jnp.float32
        embedding_layer = Embeddings(
            model_dimension=cfg.model_dimension,
            vocab_size=cfg.vocab_size,
            model_dtype=dtype,
        )
        layer = EncoderBlock(
            model_dimension=cfg.model_dimension,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            T=cfg.T,
            latent_dim=cfg.latent_dim,
            dhR=cfg.dhR,
            dhR_blocks=cfg.dhR_blocks,
            moe=cfg.moe,
            n_experts=cfg.n_experts,
            n_shared=cfg.n_shared,
            k=cfg.k,
            model_dtype=cfg.model_dtype,
        )

        return embedding_layer, layer

    @staticmethod
    def get_params(
        cfg: modelConfig,
        model: Tuple[Embeddings, EncoderBlock],
        mesh: jax.sharding.Mesh,
        key: jax.random.key,
    ) -> Tuple[PyTree, PyTree]:
        embedding_layer, layer = model

        x = jnp.ones((1, cfg.T), dtype=jnp.int32)
        key, init_key = jax.random.split(key)
        embedding_params = embedding_layer.init(init_key, x)["params"]
        embed_sharding = jax.tree.map(
            lambda _: jax.sharding.NamedSharding(mesh, P()), embedding_params
        )
        embedding_params = jax.device_put(embedding_params, embed_sharding)

        model_devices = mesh.devices.shape[1]
        assert cfg.blocks // model_devices

        layers_per_device = cfg.blocks // model_devices

        @partial(
            jax.shard_map, mesh=mesh, in_specs=(P("model")), out_specs=(P("model"))
        )
        def init_pipeline(key):
            key = key[0]
            x = jnp.ones((1, cfg.T, cfg.model_dimension))
            layer_params = []
            for _ in range(layers_per_device):
                key, init_key = jax.random.split(key)
                current_params = layer.init(init_key, x, train=False)["params"]
                layer_params.append(current_params)

            layer_params = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *layer_params)

            return layer_params

        key, *layer_keys = jax.random.split(key, model_devices + 1)
        layer_keys = jnp.array(layer_keys)
        layer_params = init_pipeline(layer_keys)
        params = (embedding_params, layer_params)

        return params

    @staticmethod
    def get_model_and_params(
        cfg: modelConfig, mesh: jax.sharding.Mesh, key: jax.random.key
    ) -> Tuple[Tuple[Embeddings, EncoderBlock], PyTree]:
        model = shardedModel.get_model(cfg)
        params = shardedModel.get_params(cfg, model, mesh, key)

        return model, params


if __name__ == "__main__":
    import json
    import numpy as np

    def print_params(params):
        def tree_shapes(tree):
            return jax.tree_util.tree_map(lambda x: tuple(x.shape), tree)

        shapes = tree_shapes(params)
        print(json.dumps(shapes, indent=4))

    model_cfg = modelConfig(
        model_dimension=16,
        n_heads=4,
        dhR=8,
        dhR_blocks=2,
        T=4,
        vocab_size=32,
        dropout=0.1,
        blocks=4,
        n_experts=4,
        n_shared=2,
        k=2,
        moe=True,
        latent_dim=8,
        model_dtype="bfloat16",
        grad_checkpoint=False,
    )

    key = jax.random.PRNGKey(0)

    model, params = Decoder.get_model(model_cfg, key)
    print_params(params)

    devices = np.array(jax.devices()).reshape((2, 4))
    mesh = jax.sharding.Mesh(devices=devices, axis_names=("data", "model"))
    print(mesh)
    model, params = shardedModel.get_model_and_params(model_cfg, mesh, key)
    print_params(params)
    breakpoint()
