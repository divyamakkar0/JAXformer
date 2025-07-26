import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import linen as nn
from einops import rearrange

import tiktoken
from utils import modelConfig
import numpy as np
from typing import Optional, Tuple, List
from jaxtyping import Array, PyTree
from functools import partial


class RMSNorm(nn.Module):
    model_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        rms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        rms = jax.lax.pmean(rms, "tensor")
        x = x / jnp.sqrt(rms + 1e-6)

        gamma = self.param(
            "gamma", nn.initializers.ones, (1,1,x.shape[-1]), self.model_dtype
        )
        beta = self.param(
            "beta", nn.initializers.zeros, (1,1,x.shape[-1]), self.model_dtype
        )

        x = x * gamma + beta

        return x


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
        self.norm = RMSNorm()

    def __call__(self, x: Array, out: bool = False) -> Array:
        if not out:
            x = self.embedding(x)
            axis = x.ndim - 1
            x = jax.lax.all_to_all(x, "tensor", split_axis=axis, concat_axis=axis-1, tiled=True)
            if self.is_mutable_collection("params"):
                x = jax.lax.all_gather(x, "tensor", axis=-1, tiled=True)
                _ = self.norm(x)
        else:
            x = jax.lax.all_to_all(x, "tensor", split_axis=x.ndim - 2, concat_axis=x.ndim - 1, tiled=True)
            x = self.norm(x)
            x = self.embedding.attend(x)

        return x


class NoisyKGate(nn.Module):
    model_dimension: int
    n_experts: int
    k: int
    model_dtype: jnp.dtype

    def setup(self):
        self.centroids = Dense(features=self.n_experts, dtype=self.model_dtype)

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
        # s = s / jnp.sum(s, axis=-1, keepdims=True) #TODO: why is this commented out?

        return g_scores, indices, s


class MoE(nn.Module):
    model_dimension: int
    n_shared: int
    n_experts: int
    k: int
    dropout: float
    model_dtype: jnp.dtype

    def setup(self):
        self.shared = Dense(
            features=self.model_dimension * self.n_shared,
        )
        self.experts = [
            FeedForward(
                model_dimension=self.model_dimension,
                ff_dim=4 * self.model_dimension,
                dropout=self.dropout,
                model_dtype=self.model_dtype,
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

        # TODO: don't think we need experts
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


class Dense(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array):
        if not self.is_mutable_collection("params"):
            params = self.scope.get_variable("params", "Dense_0")
            params["kernel"] = jax.lax.all_gather(
                params["kernel"], "fsdp", axis=-1, tiled=True
            )
            tensor_count = jax.lax.axis_size('tensor')
            x = x @ params["kernel"] + (1/tensor_count) * params["bias"]
        else:
            x = nn.Dense(features=self.features, dtype=self.dtype)(x)

        x = jax.lax.psum_scatter(x, "tensor", scatter_dimension=x.ndim - 1, tiled=True)

        return x


class FeedForward(nn.Module):
    model_dimension: int
    ff_dim: int
    dropout: float
    model_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        x = Dense(
            features=self.ff_dim,
            dtype=self.model_dtype,
        )(x)
        x = nn.gelu(x)
        x = Dense(
            features=self.model_dimension,
            dtype=self.model_dtype,
        )(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return x


class RoPE(nn.Module):
    T: int
    model_dim: int

    def setup(self):
        assert self.model_dim % 2 == 0, "model_dim must be even"

        freq = jnp.arange(self.T, dtype=jnp.float32)[:, None] + 1

        pos = jnp.arange(self.model_dim // 2, dtype=jnp.float32)[:, None]
        pos = pos.repeat(2, axis=-1).reshape(1, -1)
        log_theta_base = jnp.log(10000.0)
        theta = jnp.exp(-2 * pos / self.model_dim * log_theta_base)

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
    dropout: float = 0.0

    def setup(self):
        self.rope = self.dhR != 0

    @nn.compact
    def __call__(
        self,
        x: Array,
        *,
        cKV_cache: Optional[Array] = None,
        kRT_cache: Optional[Array] = None,
        train=True,
    ) -> Tuple[Array, Tuple[Array, Array]]:
        B, T, C = x.shape

        x = Dense(features=2 * self.latent_dim, dtype=self.model_dtype)(x)
        cKVt, cqt = jnp.split(x, 2, axis=-1)

        if self.rope:
            t_start = 0
            if cKV_cache is not None:
                t_start = cKV_cache.shape[1]

            x_k_r = Dense(features=self.dhR, dtype=self.model_dtype)(x)
            x_q_r = Dense(features=(self.dhR * self.n_heads), dtype=self.model_dtype)(x)

            rope_k = RoPE(model_dim=x_k_r.shape[-1], T=self.T)
            rope_q = RoPE(model_dim=x_q_r.shape[-1], T=self.T)

            kRt = rope_k(x_k_r, t_start)

            qRt = rope_q(x_q_r, t_start)
            qRt = rearrange(qRt, "B T (nh d) -> B nh T d", nh=self.n_heads)

        if not train:
            if cKV_cache is not None:
                cKVt = jnp.concatenate([cKV_cache, cKVt], axis=1)
            cKV_cache = cKVt

            if self.rope:
                if kRT_cache is not None:
                    kRt = jnp.concatenate([kRT_cache, kRt], axis=1)
                kRT_cache = kRt

        k, v = jnp.split(
            Dense(features=2 * self.model_dim, dtype=self.model_dtype)(cKVt), 2, axis=-1
        )
        q = Dense(features=self.model_dim, dtype=self.model_dtype)(cqt)

        qkv = jnp.concat([q, k,v], axis=0)
        qkv = rearrange(
            qkv,
            "B T (nh dk) -> B nh T dk",
            B=B * 3,
            nh=self.n_heads,
            dk=C // self.n_heads,
        )
        qkv = jax.lax.all_to_all(qkv, "tensor", split_axis=1, concat_axis=3, tiled=True)

        q, k, v = jnp.split(qkv, 3, axis=0)

        if self.rope:
            qRt = jax.lax.all_to_all(
                qRt,
                'tensor',
                split_axis=1,
                concat_axis=3,
                tiled=True
            )

            q = jnp.concatenate([q, qRt], axis=-1)
            kRt = jnp.repeat(kRt[:, None, :, :], self.n_heads, axis=1)
            kRt = jax.lax.all_to_all(
                kRt,
                'tensor',
                split_axis=1,
                concat_axis=3,
                tiled=True
            )
            k = jnp.concatenate([k, kRt], axis=-1)

        def scaledDotProd(q, k, v, mask):
            input_dtype = q.dtype

            q = q.astype("float32")
            k = k.astype("float32")
            v = v.astype("float32")

            w = jnp.einsum("B n T d, B n t d -> B n T t", q, k) * (1 / (self.model_dim // self.n_heads) ** 0.5)
            w = jnp.where(mask == 0, -9e15, w)
            w = jax.nn.softmax(w, axis=-1).astype(self.model_dtype)
            output = jnp.einsum("B n T t, B n t d -> B n T d", w, v)

            output = output.astype(input_dtype)
            return output

        local_n_heads = q.shape[1]
        if T == 1:
            mask = jnp.ones((B, local_n_heads, 1, k.shape[2]))
        else:
            mask = jnp.tril(
                jnp.ones((B, local_n_heads, q.shape[2], k.shape[2])),
            )

        output = scaledDotProd(q, k, v, mask)

        output = jax.lax.all_to_all(
            output, "tensor", split_axis=3, concat_axis=1, tiled=True
        )
        output = rearrange(output, "B nh T dk -> B T (nh dk)")

        output = Dense(features=self.model_dim, dtype=self.model_dtype)(output)
        output = nn.Dropout(rate=self.dropout)(output, deterministic=not train)


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

    @nn.compact
    def __call__(
        self,
        x: Array,
        cache: Optional[Tuple[Array, Optional[Array]]] = (None, None),
        train: bool = True,
    ):
        x_norm = RMSNorm()(x)

        x_up, cache = MLA(
            model_dim=self.model_dimension,
            n_heads=self.n_heads,
            T=self.T,
            latent_dim=self.latent_dim,
            dhR=self.dhR,
            model_dtype=self.model_dtype,
            dropout=self.dropout,
        )(x_norm, cKV_cache=cache[0], kRT_cache=cache[1], train=train)
        x = x + x_up

        x_norm = RMSNorm()(x)

        load = None
        if self.moe == True:
            x_ff, load = MoE(
                model_dimension=self.model_dimension,
                n_experts=self.n_experts,
                k=self.k,
                dropout=self.dropout,
                model_dtype=self.model_dtype,
                n_shared=self.n_shared,
            )(x_norm, train=train)

        else:
            x_ff = FeedForward(
                model_dimension=self.model_dimension,
                ff_dim=4 * self.model_dimension,
                dropout=self.dropout,
                model_dtype=self.model_dtype,
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

        if x == "":
            out = jnp.array([enc._special_tokens["<|endoftext|>"]], dtype=jnp.int32)
        else:
            out = jnp.array(enc.encode(x), dtype=jnp.int32)

        prompt_length = out.shape[0]
        out = jnp.repeat(out[None, :], B, axis=0)
        cache = None

        def sample(key, params, inp, cache, B, k, temperature):
            if not use_cache:
                cache = None
            logits, (cache, _) = self.apply(
                {"params": params}, inp, cache=cache, train=False
            )

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

        tokens = jax.device_get(out)
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
        cfg: modelConfig,
        params: PyTree,
        key: jax.random.key,
        x: str = "",
        *,
        B: int = 1,
        k: int = 10000,
        temperature: int = 1,
        max_tokens: int = 10,
        use_cache=True,
    ) -> list[str]:


        n_layers = cfg.blocks
        n_devices = n_layers * (jax.device_count() // n_layers)
        mesh = jax.sharding.Mesh(
            np.array(jax.devices())[:n_devices][None, :, None], axis_names=("fsdp", "model", "tensor")
        )

        model = shardedModel.get_model(cfg)

        out_spec = shardedModel.get_p_spec(
            model,
            mesh=mesh,
            config=cfg
        )

        params = jax.tree.map(
            lambda x, y: jax.device_put(x, jax.sharding.NamedSharding(mesh, y)),
            params,
            out_spec
        )

        enc = tiktoken.get_encoding("gpt2")
        if x == "":
            out = jnp.array([enc._special_tokens["<|endoftext|>"]], dtype=jnp.int32)
        else:
            out = jnp.array(enc.encode(x), dtype=jnp.int32)

        out = jnp.repeat(out[None, :], B, axis=0)[:, None, :]
        prompt_length = out.shape[0]
        T = cfg.T
        max_tokens = min(max_tokens, T - prompt_length)

        @jax.jit
        def sample(sample_key, params, out, cache):

            if not use_cache:
                cache = None

            if use_cache and cache is not None:
                out = out[:, :, -1:]

            sample_key, pipe_key = jax.random.split(sample_key, 2)
            logits, (cache, _) = shardedModel.pipe_step(
                model, params, out, pipe_key, train=False, cache=cache
            )

            logits = logits[:, :, -1, :]
            M, B_sample, V = logits.shape
            logits = logits.reshape(M * B_sample, V)
            logits, idx = jax.lax.top_k(logits, k=k)
            logits /= temperature

            sample_prob = lambda key, logits, idx: idx[
                jax.random.categorical(key, logits)
            ]
            sample_key = jnp.array(jax.random.split(sample_key, logits.shape[0]))
            out_next = jax.vmap(sample_prob)(sample_key, logits, idx)
            out_next = out_next.reshape(M, B_sample, 1)

            return out_next, (cache, logits)

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(
                P("fsdp", "model", "tensor"),
                out_spec,
                P(),
            ),
            out_specs=P("fsdp", "model", "tensor")
        )
        def generate_shard(key, params, out):

            cache = None
            key = key[0, 0, 0]
            for _ in range(max_tokens):
                key, sample_key = jax.random.split(key)
                out_next, (cache, _logits) = sample(sample_key, params, out, cache)
                out = jnp.concatenate([out, out_next], axis=-1)

            return out[None, None, ...]

        sample_keys = jax.random.split(key, n_devices)
        sample_keys = jnp.array(sample_keys)[None, :, None, :]
        sample_keys = jax.device_put(
            sample_keys, jax.sharding.NamedSharding(mesh, P("fsdp", "model", "tensor"))
        )

        out = generate_shard(sample_keys, params, out)

        tokens = jax.device_get(out)
        tokens = tokens.reshape(-1, tokens.shape[-1])
        outputs = list(map(lambda x: enc.decode(x), tokens))

        return outputs

    @staticmethod
    def pipe_step(model, params, x, key, train, cache=None, checkpoint=False):
        embedding_model, layer_model = model
        embedding_params, layer_params = params

        embeddings = embedding_model.apply({"params": embedding_params}, x, out=False)

        layer_fn = lambda x, params, cKV_cache, kRT_cache, key: layer_model.apply(
            {"params": params},
            x,
            cache=None if cKV_cache is None else (cKV_cache, kRT_cache),
            train=train,
            rngs=None if not train else {"dropout": key},
        )

        layer_fn = jax.checkpoint(
            layer_fn, policy=jax.checkpoint_policies.nothing_saveable
        )

        layer_out, (current_cache, load) = shardedModel.layer_fn(
            layer_fn, embeddings, layer_params, key, cache=cache
        )

        logits = embedding_model.apply(
            {"params": embedding_params}, layer_out, out=True
        )
        return logits, (current_cache, load)

    @staticmethod
    def layer_fn(fwd_fn, x, params, key, cache=None):
        idx = jax.lax.axis_index("model")
        n_devices = jax.lax.psum(1, "model")
        microbatch_per_device = x.shape[0]
        microbatch = n_devices * microbatch_per_device
        layers_per_device = params['Block_0']['MLA_0']['Dense_0']['Dense_0']['kernel'].shape[0]
        layers = layers_per_device * n_devices
        perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]

        outputs = jnp.zeros_like(x) * jnp.nan
        state = jnp.zeros(
            (layers_per_device, x.shape[1], x.shape[2], x.shape[3]), dtype=x.dtype
        )

        cKV_cache = []
        kRT_cache = []
        out_load = None

        for i in range(layers + microbatch - 1):
            batch_idx = i % microbatch_per_device
            layer_idx = (i + 1 - layers) % microbatch_per_device

            state = state.at[0].set(jnp.where(idx == 0, x[batch_idx], state[0]))
            current_cache = (None, None)
            if cache is not None:
                current_cache = (cache[0][i], cache[1][i])
            key, *dropout_key = jax.random.split(key, layers_per_device + 1)
            dropout_key = jnp.array(dropout_key)

            state, (layer_cache, load) = jax.vmap(fwd_fn)(
                x=state,
                params=params,
                cKV_cache=current_cache[0],
                kRT_cache=current_cache[1],
                key=dropout_key,
            )

            if layer_cache is not None:
                cKV_cache.append(jnp.nan_to_num(layer_cache[0]))
                kRT_cache.append(jnp.nan_to_num(layer_cache[1]))
            if out_load is None:
                out_load = load

            outputs = outputs.at[layer_idx].set(
                jnp.where(
                    idx == (n_devices - 1),
                    state[-1],
                    outputs[layer_idx],
                )
            )

            state_perm = jax.lax.ppermute(
                state[-1],
                axis_name="model",
                perm=perm,
            )[None, ...]

            state = jnp.concatenate([state_perm, state[:-1]], axis=0)

            if batch_idx == microbatch_per_device - 1:
                x = jax.lax.ppermute(
                    x,
                    axis_name="model",
                    perm=perm,
                )

            if layer_idx == microbatch_per_device - 1:
                outputs = jax.lax.ppermute(
                    outputs,
                    axis_name="model",
                    perm=perm,
                )

        if len(cKV_cache) == 0:
            out_cache = None
        else:
            cKV_cache = jnp.stack(cKV_cache, axis=0)

            if kRT_cache[0] is not None:
                kRT_cache = jnp.stack(kRT_cache, axis=0)
            else:
                kRT_cache = None

            out_cache = (cKV_cache, kRT_cache)

        outputs = jax.lax.ppermute(
            outputs,
            axis_name="model",
            perm=perm,
        )

        return outputs, (out_cache, out_load)

    @staticmethod
    def get_p_spec(
        model: Tuple[Embeddings, EncoderBlock],
        mesh: jax.sharding.Mesh,
        config: modelConfig,
    ) -> jax.sharding.NamedSharding:
        embed, layer = model

        T = config.T
        n_blocks = mesh.devices.shape[1]
        n_layers = config.blocks

        x_embed = jnp.ones((1, T), dtype=jnp.int32)
        x_layer = jnp.ones((1, T, embed.model_dimension), dtype=jnp.float32)
        key = jax.random.PRNGKey(0)

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P(None, 'tensor'), P(None, None, 'tensor')),
            out_specs=(P("model")),
        )
        def get_var_spec_shard(x_embed, x_layer):

            embed_shape = embed.init(key, x_embed)["params"]
            layer_shape = []
            for _ in range(n_layers // n_blocks):
                layer_shape.append(layer.init(key, x_layer, train=False)["params"])
            layer_shape = jax.tree.map(
                lambda *x: jnp.stack(x, axis=0),
                *layer_shape
            )

            return embed_shape, layer_shape

        eval_shape = jax.eval_shape(
            get_var_spec_shard,
            x_embed,
            x_layer,
        )

        def embedding_partition(_: Array) -> P:
            return P(None)

        def layer_partition(x: Array) -> P:
            if x.ndim == 4:
                return P("model", None, None, "tensor")
            if x.ndim == 3:
                return P("model", "tensor", "fsdp")
            return P("model")

        embed_p_spec = jax.tree.map(
            embedding_partition,
            eval_shape[0],
        )
        layer_p_spec = jax.tree.map(
            layer_partition,
            eval_shape[1],
        )

        return embed_p_spec, layer_p_spec

    @staticmethod
    def get_model(cfg) -> Tuple[Embeddings, EncoderBlock]:
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

        out_spec = shardedModel.get_p_spec(model, mesh, cfg)
        def replace_fsdp(p: jax.sharding.PartitionSpec):
            if p[-1] == "fsdp":
                p = P(
                    *p[:-1], None
                )
            return p

        out_spec_no_fsdp = jax.tree.map(
            lambda x: replace_fsdp(x), out_spec
        )

        embedding_layer, layer = model

        x_embed = jnp.ones((1, cfg.T), dtype=jnp.int32)
        x_layer = jnp.ones((1, cfg.T, cfg.model_dimension), dtype=jnp.float32)

        model_devices = mesh.devices.shape[1]
        tensor_devices = mesh.devices.shape[2]
        assert cfg.blocks // model_devices
        layers_per_device = cfg.blocks // model_devices

        key, embed_key = jax.random.split(key, 2)
        key, *layer_keys = jax.random.split(key, tensor_devices * model_devices + 1)
        layer_keys = jnp.array(layer_keys).reshape(model_devices, tensor_devices, 2)

        @jax.jit
        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P(None, "tensor"), P(None, None, "tensor"), P("model", "tensor")),
            out_specs=out_spec_no_fsdp
        )
        def init_weights(x_embed, x_layer, layer_key):
            layer_key = layer_key[0, 0]
            embedding_params = embedding_layer.init(embed_key, x_embed, out=False)["params"]
            layer_params = []

            for _ in range(layers_per_device):
                layer_key, init_key = jax.random.split(layer_key)
                current_params = layer.init(init_key, x_layer, train=False)["params"]
                layer_params.append(current_params)

            layer_params = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *layer_params)
            return embedding_params, layer_params

        params = init_weights(x_embed, x_layer, layer_keys)
        params = jax.tree.map(
            lambda x, y: jax.device_put(x, jax.sharding.NamedSharding(mesh, y)),
            params,
            out_spec
        )

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
            return jax.tree.map(lambda x: tuple(x.shape), tree)

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
        blocks=2,
        n_experts=4,
        n_shared=2,
        k=2,
        moe=False,
        latent_dim=8,
        model_dtype="bfloat16",
    )

    key = jax.random.PRNGKey(0)

    model, params = Decoder.get_model(model_cfg, key)
    print_params(params)

    devices = np.array(jax.devices()).reshape((1, 2, 2))
    mesh = jax.sharding.Mesh(devices=devices, axis_names=("fsdp", "model", "tensor"))
    print(mesh)
    model, params = shardedModel.get_model_and_params(model_cfg, mesh, key)
    print_params(params)
    breakpoint()
