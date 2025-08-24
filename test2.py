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

import time

cache_type = Tuple[Optional[Array], Optional[Array]]
dtype_map = {
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
        "float16": jnp.float16,
        "int32": jnp.int32,
        "int64": jnp.int64,
    }

def convert_dtype(dtype_str):
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


# @dataclass
# class modelConfig:
#     model_dimension: int
#     vocab_size: int
#     n_head: int
#     blocks: int
#     layers_per_block: int
#     T: int
#     latent_dim: int
#     dhR: int
#     dropout_rate: float = 0.1
#     model_dtype: jnp.dtype = jnp.bfloat16


class Dense(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(features=self.features, dtype=self.dtype)(x)
        x = jax.lax.psum_scatter(x, "tp", scatter_dimension=x.ndim - 1, tiled=True)
        return x

class FeedForward(nn.Module):
    model_dimension: int
    dropout_rate: float = 0.1
    model_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: Array, train=True) -> Array:
        x = Dense(features=self.model_dimension * 4, dtype=self.model_dtype)(x)
        x = nn.selu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = Dense(features=self.model_dimension, dtype=self.model_dtype)(x)
        return x


class RMSNorm(nn.Module):
    model_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x_type = x.dtype
        x = x.astype(jnp.float32)
        rms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        rms = jax.lax.pmean(rms, axis_name="tp")
        x = x / jnp.sqrt(rms + 1e-6)
        x = x.astype(x_type)

        gamma = self.param(
            "gamma", nn.initializers.ones, (1, 1, x.shape[-1]), self.model_dtype
        )
        beta = self.param(
            "beta", nn.initializers.zeros, (1, 1, x.shape[-1]), self.model_dtype
        )

        x = x * gamma + beta

        return x


class Embedding(nn.Module):
    model_dimension: int
    T: int
    vocab_size: int
    model_dtype: jnp.dtype

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dimension,
            dtype=self.model_dtype,
        )

        self.pos_embedding = nn.Embed(
            num_embeddings=self.T,
            features=self.model_dimension,
            dtype=self.model_dtype,
        )

        self.norm = RMSNorm(model_dtype=self.model_dtype)

    def __call__(self, x: Array, out: bool = False) -> Array:
        if not out:
            *_, T = x.shape
            x = self.embedding(x)
            pos_emb = self.pos_embedding(jnp.arange(T))
            x = x + pos_emb
            x = jax.lax.all_to_all(
                x, 'tp', split_axis=x.ndim - 1, concat_axis=x.ndim - 2, tiled=True
            )
            if self.is_mutable_collection("params"):
                x = jax.lax.all_gather(x, "tp", axis=-1, tiled=True)
                _ = self.norm(x)
        else:
            x = jax.lax.all_to_all(
                x, 'tp', split_axis=x.ndim - 2, concat_axis=x.ndim - 1, tiled=True
            )
            x = self.norm(x)
            x = self.embedding.attend(x)

        return x


class RoPE(nn.Module):
    T: int
    model_dim: int
    tensor_size: int

    def setup(self):

        assert self.model_dim % 2 == 0, "model_dim must be even"

        freq = jnp.arange(self.T, dtype=jnp.float32)[:, None] + 1

        pos = jnp.arange(self.model_dim // 2, dtype=jnp.float32)[:, None]
        pos = pos.repeat(2, axis=-1).reshape(1, -1)
        log_theta_base = jnp.log(10000.0)
        theta = jnp.exp(-2 * pos / self.model_dim * log_theta_base)

        idx = jax.lax.axis_index("tp")
        slice_factor = self.model_dim // self.tensor_size

        cos = jnp.cos(freq * theta)
        sin = jnp.sin(freq * theta)

        self.cos = jax.lax.dynamic_slice_in_dim(cos, slice_factor * idx, slice_factor, axis=-1)
        self.sin = jax.lax.dynamic_slice_in_dim(sin, slice_factor * idx, slice_factor, axis=-1)

    def __call__(
        self,
        x: Array,
        t_start: int,
    ) -> Array:

        B, T, C = x.shape
        x_dtype = x.dtype
        x = x.astype(jnp.float32)

        cos_rope = x * self.cos[t_start : t_start + T, :]

        x_inter = x.reshape((B, T, C // 2, 2))
        x_inter_one = x_inter[..., 0]
        x_inter_two = -1 * x_inter[..., 1]
        x_inter = jnp.stack([x_inter_two, x_inter_one], axis=-1).reshape((B, T, C))

        x_inter = x_inter.reshape((B, T, C))
        sin_rope = x_inter * self.sin[t_start : t_start + T, :]

        x = cos_rope + sin_rope
        x = x.astype(x_dtype)

        return x


class MLA(nn.Module):
    model_dimension: int
    n_heads: int
    T: int
    latent_dim: int
    dhR: int
    model_dtype: jnp.dtype
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: Array,
        *,
        cKV_cache: Optional[Array] = None,
        kRT_cache: Optional[Array] = None,
        train=True,
    ) -> Tuple[Array, Tuple[Array, Array]]:
        use_rope = self.dhR > 0

        B, T, C = x.shape

        x = Dense(features=2 * self.latent_dim, dtype=self.model_dtype)(x)
        cKVt, cqt = jnp.split(x, 2, axis=-1)

        if use_rope:
            t_start = cKV_cache.shape[1] if cKV_cache is not None else 0
            x_k_r = Dense(features=self.dhR, dtype=self.model_dtype)(x)
            x_q_r = Dense(features=self.dhR * self.n_heads, dtype=self.model_dtype)(x)

            rope_k = RoPE(model_dim=self.dhR, T=self.T, tensor_size=self.dhR // x_k_r.shape[-1])
            rope_q = RoPE(model_dim=self.dhR * self.n_heads, T=self.T, tensor_size=(self.dhR * self.n_heads) // x_q_r.shape[-1])

            kRt = rope_k(x_k_r, t_start)

            qRt = rope_q(x_q_r, t_start)
            qRt = rearrange(qRt, "B T (nh d) -> B nh T d", nh=self.n_heads)

        if not train:
            if cKV_cache is not None:
                cKVt = jnp.concatenate([cKV_cache, cKVt], axis=1)
            cKV_cache = cKVt

            if use_rope:
                if kRT_cache is not None:
                    kRt = jnp.concatenate([kRT_cache, kRt], axis=1)
                kRT_cache = kRt

        k, v = jnp.split(Dense(
            features=2 * self.model_dimension, dtype=self.model_dtype
        )(cKVt), 2, axis=-1)
        q = Dense(features=self.model_dimension, dtype=self.model_dtype)(cqt)

        q,k, v = jax.tree.map(
            lambda x: rearrange(x,  "B T (nh d) -> B nh T d", nh=self.n_heads),
            (q, k, v)
        )

        q,k,v = jax.tree.map(
            lambda x: jax.lax.all_to_all(
                x, 'tp', split_axis=1, concat_axis=3, tiled=True
            ),
            (q, k, v)
        )

        if use_rope:

            qRt = jax.lax.all_to_all(
                qRt, 'tp', split_axis=1, concat_axis=3, tiled=True
            )
            q = jnp.concatenate([q, qRt], axis=-1)

            kRt = jnp.repeat(kRt[:, None, :, :], self.n_heads, axis=1)
            kRt = jax.lax.all_to_all(
                kRt, 'tp', split_axis=1, concat_axis=3, tiled=True
            )
            k = jnp.concatenate([k, kRt], axis=-1)

        def scaledDotProd(q, k, v, mask):
            input_dtype = q.dtype

            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)
            v = v.astype(jnp.float32)

            dk = q.shape[-1]

            w = jnp.einsum("B n T d, B n t d -> B n T t", q, k) * (dk**-0.5)
            w = jnp.where(mask == 0, -jnp.inf, w)
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
            output, 'tp', split_axis=3, concat_axis=1, tiled=True
        )
        output = rearrange(output, "B nh T dk -> B T (nh dk)")

        output = Dense(features=self.model_dimension, dtype=self.model_dtype)(output)
        output = nn.Dropout(rate=self.dropout)(output, deterministic=not train)

        return output, (cKV_cache, kRT_cache)


class AttentionBasic(nn.Module):
    model_dimension: int
    n_heads: int
    model_dtype: jnp.dtype
    T: int
    dropout: float = 0.0

    def setup(self):
        self.mask = jnp.tril(jnp.ones((1, 1, self.T, self.T)))


    @nn.compact
    def __call__(self, x: Array, train=True) -> Array:

        qkv = Dense(features=3 * self.model_dimension, dtype=self.model_dtype)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = rearrange(q, "B T (nh d) -> B nh T d", nh=self.n_heads)
        k = rearrange(k, "B T (nh d) -> B nh d T", nh=self.n_heads)
        v = rearrange(v, "B T (nh d) -> B nh T d", nh=self.n_heads)

        att = (q @ k) * (q.shape[-1]**-0.5)
        att = jnp.where(self.mask == 0, -jnp.inf, att)
        att = jax.nn.softmax(att, axis=-1)

        output = att @ v
        output = rearrange(output, "B nh T d -> B T (nh d)")
        output = Dense(features=self.model_dimension, dtype=self.model_dtype)(output)

        return output, (None, None)


class Layer(nn.Module):
    model_dimension: int
    n_heads: int
    T: int
    latent_dim: int
    dhR: int
    dropout_rate: float = 0.1
    model_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self, x, cache: Optional[cache_type] = None, train=True
    ) -> Tuple[Array, cache_type]:
        x_res = x

        x = RMSNorm(model_dtype=self.model_dtype)(x)
        # x, cache = MLA(
        #     model_dimension=self.model_dimension,
        #     n_heads=self.n_heads,
        #     T=self.T,
        #     latent_dim=self.latent_dim,
        #     dhR=self.dhR,
        #     model_dtype=self.model_dtype,
        #     dropout=self.dropout_rate,
        # )(x, cKV_cache=cache[0], kRT_cache=cache[1], train=train)
        x, cache = AttentionBasic(
            model_dimension=self.model_dimension,
            n_heads=self.n_heads,
            model_dtype=self.model_dtype,
            T=self.T,
            dropout=self.dropout_rate,
        )(x, train=train)
        x = x + x_res
        x_res = x

        x = RMSNorm(model_dtype=self.model_dtype)(x)
        x = FeedForward(
            model_dimension=self.model_dimension,
            dropout_rate=self.dropout_rate,
            model_dtype=self.model_dtype,
        )(x, train=train)
        x = x + x_res

        return x, cache


class Block(nn.Module):
    layers: int
    model_dimension: int
    n_heads: int
    T: int
    latent_dim: int
    dhR: int
    dropout_rate: float = 0.1
    model_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self, x, cache: Optional[cache_type] = None, train=True
    ) -> Tuple[Array, cache_type]:
        cKV_cache = []
        kRT_cache = []

        for i in range(self.layers):
            current_cache = [None, None]
            if cache is not None:
                current_cache[0] = cache[0][i]
                if i < self.layers - 1:
                    current_cache[1] = cache[1][i]

            x, cache_out = Layer(
                model_dimension=self.model_dimension,
                n_heads=self.n_heads,
                T=self.T,
                latent_dim=self.latent_dim,
                dhR=self.dhR if i < self.layers - 1 else 0,
                dropout_rate=self.dropout_rate,
                model_dtype=self.model_dtype,
            )(x, current_cache, train=train)

            ckV, kRT = cache_out
            if ckV is not None:
                cKV_cache.append(ckV)
            if kRT is not None:
                kRT_cache.append(kRT)

        cKV_cache = jnp.stack(cKV_cache, axis=0) if len(cKV_cache) > 0 else None
        kRT_cache = jnp.stack(kRT_cache, axis=0) if len(kRT_cache) > 0 else None

        out_cache = (cKV_cache, kRT_cache)

        return x, out_cache


class Transformer(nn.Module):
    model_dimension: int
    vocab_size: int
    n_head: int
    blocks: int
    layers_per_block: int
    T: int
    latent_dim: int
    dhR: int
    dropout_rate: float = 0.1
    model_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self, x, cache: Optional[cache_type] = None, train=True
    ) -> Tuple[Array, cache_type]:
        if cache is not None:
            x = x[..., :1]

        *B, T = x.shape
        x = x.reshape(-1, T)

        embedding = Embedding(
            vocab_size=self.vocab_size,
            model_dimension=self.model_dimension,
            model_dtype=self.model_dtype,
        )

        x = embedding(x)

        cKV_cache = []
        ckRT_cache = []

        for i in range(self.blocks):
            if cache is None:
                layer_cache = None
            else:
                cKV = cache[0][i]
                kRT = cache[1][i] if cache[1] is not None else None
                layer_cache = (cKV, kRT)

            x, cache_out = Block(
                layers=self.layers_per_block,
                model_dimension=self.model_dimension,
                n_heads=self.n_head,
                T=self.T,
                latent_dim=self.latent_dim,
                dhR=self.dhR,
                dropout_rate=self.dropout_rate,
                model_dtype=self.model_dtype,
            )(x, layer_cache, train=train)

            if cache_out[0] is not None:
                cKV_cache.append(cache_out[0])
            if cache_out[1] is not None:
                ckRT_cache.append(cache_out[1])

        if len(cKV_cache) > 0:
            cKV_cache = jnp.stack(cKV_cache, axis=0)
        else:
            cKV_cache = None
        if len(ckRT_cache) > 0:
            ckRT_cache = jnp.stack(ckRT_cache, axis=0)
        else:
            ckRT_cache = None
        out_cache = (cKV_cache, ckRT_cache)

        x_out = embedding(x, out=True)
        x_out = x_out.reshape(*B, T, self.vocab_size)

        return x_out, out_cache

    @classmethod
    def get_model(cls, cfg: modelConfig) -> "Transformer":
        return cls(
            model_dimension=cfg.model_dimension,
            vocab_size=cfg.vocab_size,
            n_head=cfg.n_head,
            blocks=cfg.blocks,
            layers_per_block=cfg.layers_per_block,
            T=cfg.T,
            latent_dim=cfg.latent_dim,
            dhR=cfg.dhR,
            dropout_rate=cfg.dropout_rate,
            model_dtype=convert_dtype(cfg.model_dtype),
        )

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
        enc = tiktoken.get_encoding("cl100k_base")
        out = jnp.array(
            [enc._special_tokens["<|endoftext|>"]] if x == "" else enc.encode(x),
            dtype=jnp.int32,
        )

        prompt_length = out.shape[0]
        generation_length = min(max_tokens, self.T - prompt_length)
        out = jnp.repeat(out[None, :], B, axis=0)
        cache = None

        @jax.jit
        def sample(key, params, inp, cache):
            logits, cache = self.apply(
                {"params": params}, inp, cache=cache, train=False
            )
            logits = logits[:, -1, :]
            logits, idx = jax.lax.top_k(logits, k=k)
            logits /= temperature

            out_next_idx = jax.random.categorical(key, logits, axis=-1, shape=(B,))
            out_next = idx[jnp.arange(B, dtype=jnp.int32), out_next_idx][:, None]

            return out_next, (cache, logits)

        for _ in range(generation_length):
            start_time = time.time()
            if not use_cache:
                cache = None
            key, sample_key = jax.random.split(key)
            out_next, (cache, _logits) = sample(
                sample_key, params, out, cache
            )
            out = jnp.concatenate([out, out_next], axis=-1)
            end_time = time.time()
            token_time = end_time - start_time
            print(f"Token {_ + 1} generated \t {1/token_time:.4f} tk/s")

        tokens = jax.device_get(out)
        outputs = list(map(lambda x: enc.decode(x), tokens))

        return outputs


class shardedModel:
    def __init__(self, cfg: modelConfig):
        self.dtype = convert_dtype(cfg.model_dtype)
        self.embedding = Embedding(
            vocab_size=cfg.vocab_size,
            T = cfg.T,
            model_dimension=cfg.model_dimension,
            model_dtype=self.dtype,
        )

        self.block = Block(
            layers=cfg.layers_per_block,
            model_dimension=cfg.model_dimension,
            n_heads=cfg.n_head,
            T=cfg.T,
            latent_dim=cfg.latent_dim,
            dhR=cfg.dhR,
            dropout_rate=cfg.dropout_rate,
            model_dtype=self.dtype
        )

        self.cfg = cfg

    def init_weights(self, key, mesh):
        out_spec = shardedModel.get_p_spec([self.embedding, self.block], mesh, self.cfg)

        x_embed = jnp.ones((1, self.cfg.T), dtype=jnp.int32)
        x_layer = jnp.ones((1, self.cfg.T, self.cfg.model_dimension), dtype=self.dtype)

        layer_devices = mesh.devices.shape[1]

        assert self.cfg.blocks // layer_devices
        layers_per_device = self.cfg.blocks // layer_devices

        key, embed_key = jax.random.split(key, 2)
        key, *layer_keys = jax.random.split(key, layer_devices + 1)
        layer_keys = jnp.array(layer_keys).reshape(layer_devices, 2)

        @jax.jit
        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P(None, "tp"), P(None, None, "tp"), P("pp")),
            out_specs=out_spec,
        )
        def init_params(x_embed, x_layer, layer_key):
            layer_key = layer_key.reshape(
                2,
            )
            embedding_params = self.embedding.init(embed_key, x_embed, out=False)[
                "params"
            ]
            layer_params = []

            for _ in range(layers_per_device):
                layer_key, init_key = jax.random.split(layer_key)
                current_params = self.block.init(init_key, x_layer, train=False)[
                    "params"
                ]
                layer_params.append(current_params)
            layer_params = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *layer_params)

            return embedding_params, layer_params

        params = init_params(x_embed, x_layer, layer_keys)
        params = jax.tree.map(
            lambda x, y: jax.device_put(x, jax.sharding.NamedSharding(mesh, y)),
            params,
            out_spec,
        )

        return params

    def pipe_step(self, params, x, key, train, cache=None):
        embedding_params, layer_params = params

        if cache is not None:
            x = x[..., :1]

        embeddings = self.embedding.apply({"params": embedding_params}, x, out=False)

        layer_fn = lambda x, params, cache, key: self.block.apply(
            {"params": params},
            x,
            cache=cache,
            train=train,
            rngs={"dropout": key} if train else None,
        )

        @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def fwd_fn(state_idx, x, params, cache, key):
            def grad_fn(stop_grad):
                return (
                    lambda *args: jax.lax.stop_gradient(layer_fn(*args))
                    if stop_grad
                    else layer_fn(*args)
                )

            fns = [
                grad_fn(stop_grad=True),
                grad_fn(stop_grad=False),
            ]

            return jax.lax.switch(
                state_idx,
                fns,
                x,
                params,
                cache,
                key,
            )

        layer_out, out_cache = self.pipeline(
            fwd_fn, layer_params, embeddings, cache, key
        )

        logits = self.embedding.apply({"params": embedding_params}, layer_out, out=True)
        return logits, (out_cache, None)

    def pipeline(
        self,
        fn,
        stage_params: PyTree,
        inputs: Array,
        cache: Optional[Tuple[Array, Optional[Array]]],
        key: jax.random.PRNGKey,
    ):
        stage = jax.lax.axis_index("pp")
        n_devices = jax.lax.axis_size("pp")
        layers_per_device = stage_params["Layer_0"]["AttentionBasic_0"]["Dense_0"]["Dense_0"][
            "kernel"
        ].shape[0]
        microbatch_per_device = inputs.shape[0]
        microbatches = n_devices * microbatch_per_device
        layers = layers_per_device * n_devices
        outputs = jnp.zeros_like(inputs) * jnp.nan
        state = (
            jnp.zeros(
                (
                    layers_per_device,
                    *inputs.shape[1:],
                )
            )
            * jnp.nan
        )

        state_idx = jnp.zeros((layers_per_device,), dtype=jnp.int32)
        perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]

        ckV_cache = []
        kRT_cache = []
        for i in range(microbatches + layers - 1):
            batch_idx = i % microbatch_per_device
            layer_idx = (i - layers + 1) % microbatch_per_device

            state = state.at[0].set(jnp.where(stage == 0, inputs[batch_idx], state[0]))
            state_idx = state_idx.at[0].set(jnp.where(stage == 0, 1, state_idx[0]))

            key, *layer_keys = jax.random.split(key, layers_per_device + 1)
            layer_keys = jnp.array(layer_keys)

            current_cache = None
            if cache is not None:
                current_cache = [cache[0][i], None]
                if cache[1] is not None:
                    current_cache[1] = cache[1][i]

            state, out_cache = jax.vmap(fn)(
                state_idx, state, stage_params, current_cache, layer_keys
            )

            if out_cache[0] is not None:
                ckV_cache.append(out_cache[0])
            if out_cache[1] is not None:
                kRT_cache.append(out_cache[1])

            outputs = outputs.at[layer_idx].set(
                jnp.where(stage == n_devices - 1, state[-1], outputs[layer_idx])
            )

            state = jnp.concat(
                [jax.lax.ppermute(state[-1], "pp", perm)[None, ...], state[:-1]], axis=0
            )
            state_idx = jnp.concat(
                [
                    jax.lax.ppermute(state_idx[-1], "pp", perm)[None, ...],
                    state_idx[:-1],
                ],
                axis=0,
            )
            if batch_idx == microbatch_per_device - 1:
                inputs = jax.lax.ppermute(inputs, axis_name="pp", perm=perm)

            if layer_idx == microbatch_per_device - 1:
                outputs = jax.lax.ppermute(outputs, axis_name="pp", perm=perm)

        outputs = jax.lax.ppermute(outputs, "pp", perm)

        if len(ckV_cache) > 0:
            ckV_cache = jnp.stack(ckV_cache, axis=0)
        else:
            ckV_cache = None

        if len(kRT_cache) > 0:
            kRT_cache = jnp.stack(kRT_cache, axis=0)
        else:
            kRT_cache = None
        out_cache = (ckV_cache, kRT_cache)

        return outputs, out_cache

    def generate(
        self,
        params: PyTree,
        cfg: modelConfig,
        key: jax.random.key,
        x: str = "",
        *,
        B: int = 1,
        k: int = 10000,
        temperature: int = 1,
        max_tokens: int = 10,
        n_devices: int = 1,
        use_cache=True,
    ) -> list[str]:

        assert B % n_devices == 0, "Batch size must be divisible by number of devices"
        assert n_devices <= jax.local_device_count(), "Number of devices exceeds available devices"

        mesh = jax.make_mesh(
            (1, n_devices, 1),
            axis_names=("dp", "pp", "tp"),
            devices=np.array(jax.local_devices())[:n_devices],
        )

        model = shardedModel(cfg)
        out_spec = shardedModel.get_p_spec([model.embedding, model.block], mesh, cfg)
        params = jax.tree.map(
            lambda x, y: jax.device_put(jax.experimental.multihost_utils.process_allgather(x, tiled=True), jax.sharding.NamedSharding(mesh, y)),
            params,
            out_spec,
        )
        enc = tiktoken.get_encoding("cl100k_base")
        out = jnp.array(
            [enc._special_tokens["<|endoftext|>"]] if x == "" else enc.encode(x),
            dtype=jnp.int32,
        )
        out = jnp.repeat(out[None, :], B, axis=0).reshape(n_devices, B // n_devices, -1)

        prompt_length = out.shape[-1]
        generation_length = min(max_tokens, cfg.T - prompt_length)

        @jax.jit
        def sample(params, out, cache, sample_key):
            sample_key, pipe_key = jax.random.split(sample_key, 2)
            logits, (cache, _) = shardedModel.pipe_step(
                model, params, out, pipe_key, train=False, cache=cache
            )

            logits = logits[:, :, -1, :]
            M, B_sample, _ = logits.shape
            logits = logits.reshape(M * B_sample, -1)
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
                out_spec,
                P(),
                P("dp", "pp", "tp"),
            ),
            out_specs=P("pp", "dp", "tp"),
        )
        def generate_shard(params, out, key):
            cache = None
            key = key.reshape(2,)
            for _ in range(generation_length):
                start_time = time.time()
                if not use_cache:
                    cache = None
                key, sample_key = jax.random.split(key)
                out_next, (cache, _logits) = sample(params, out, cache, sample_key)

                out = jnp.concatenate([out, out_next], axis=-1)
                end_time = time.time()
                token_time = end_time - start_time
                print(f"Token {_ + 1} generated \t {1/token_time:.4f} tk/s")
            return out[None, None, ...]

        key = jax.random.fold_in(key, jax.process_index())
        sample_key = jnp.array(jax.random.split(key, B)).reshape(
            n_devices, B // n_devices, 2
        )
        out = generate_shard(params, out, sample_key)

        tokens = jax.device_get(out)
        tokens = tokens.reshape(-1, tokens.shape[-1])
        tokens = jax.experimental.multihost_utils.process_allgather(tokens, tiled=True)

        outputs = [enc.decode(x) for x in tokens]

        return outputs

    @staticmethod
    def get_p_spec(
        model: Tuple[Embedding, Block], mesh: jax.sharding.Mesh, config: modelConfig
    ) -> Tuple[jax.sharding.NamedSharding, jax.sharding.NamedSharding]:
        T = config.T
        n_devices = mesh.devices.shape[1]
        n_layers = config.blocks
        assert n_layers % n_devices == 0, "Number of layers must be divisible by number of devices"

        embed, layer = model

        x_embed = jnp.ones((1, T), dtype=jnp.int32)
        x_layer = jnp.ones((1, T, embed.model_dimension), dtype=jnp.float32)
        key = jax.random.PRNGKey(0)

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P(None, "tp"), P(None, None, "tp")),
            out_specs=(P("pp")),
        )
        def get_var_spec_shard(x_embed, x_layer):
            embed_shape = embed.init(key, x_embed)["params"]
            layer_shape = []
            for _ in range(n_layers // n_devices):
                layer_shape.append(layer.init(key, x_layer, train=False)["params"])
            layer_shape = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *layer_shape)

            return embed_shape, layer_shape

        eval_shape = jax.eval_shape(
            get_var_spec_shard,
            x_embed,
            x_layer,
        )

        join_fn = lambda path: " ".join(i.key for i in path).lower()

        def layer_partition(key: Tuple[str, ...], x: Array) -> P:
            path = join_fn(key)
            if "moe" in path and "feedforward" in path:
                if x.ndim == 4:
                    return P("pp", None, "tp", None)
                if x.ndim == 3:
                    return P("pp", "tp", None)

            if "gamma" in path or "beta" in path:
                return P("pp", None, None, "tp")

            if x.ndim == 3:
                return P("pp", "tp", None)

            return P("pp", None)

        embed_p_spec = jax.tree.map(
            lambda _: P(),
            eval_shape[0],
        )

        layer_p_spec = jax.tree.map_with_path(
            layer_partition,
            eval_shape[1],
        )

        return embed_p_spec, layer_p_spec


if __name__ == '__main__':

    jax.distributed.initialize()
    modelCfg = modelConfig(
        model_dimension=128,
        vocab_size=100277,
        n_head=8,
        blocks=4,
        layers_per_block=2,
        T=128,
        latent_dim=64,
        dhR=32,
        dropout_rate=0.2,
        model_dtype=jnp.bfloat16,
    )

    model = Transformer.get_model(modelCfg)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, modelCfg.T), dtype=jnp.int32))['params']

    out = model.generate(
        params=params,
        key=jax.random.PRNGKey(0),
        x="Hello, world!",
        B=1,
        k=10000,
        temperature=1,
        max_tokens=50,
        use_cache=False,
    )

    print(out)
