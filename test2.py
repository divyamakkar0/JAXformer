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

class Dense(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return nn.Dense(features=self.features, dtype=self.dtype)(x)

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

        x_type =x.dtype
        x = x.astype(jnp.float32)
        rms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
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
    vocab_size: int
    model_dtype: jnp.dtype

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dimension,
            dtype=self.model_dtype,
        )

        self.norm = RMSNorm(model_dtype=self.model_dtype)

    def __call__(self, x: Array, out: bool = False) -> Array:
        if not out:
            x = self.embedding(x)
            if self.is_mutable_collection("params"):
                _ = self.norm(x)
        else:
            x = self.norm(x)
            x = self.embedding.attend(x)

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

        x = Dense(features = 2 * self.latent_dim, dtype=self.model_dtype)(x)
        cKVt, cqt = jnp.split(x, 2, axis=-1)

        if use_rope:
            t_start = cKV_cache.shape[1] if cKV_cache is not None else 0
            x_k_r = Dense(features=self.dhR, dtype=self.model_dtype)(x)
            x_q_r = Dense(features=self.dhR * self.n_heads, dtype=self.model_dtype)(x)

            rope_k = RoPE(model_dim=self.dhR, T=self.T)
            rope_q = RoPE(model_dim=self.dhR * self.n_heads, T=self.T)

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

        k, v = jnp.split(
            Dense(features=2 * self.model_dimension, dtype=self.model_dtype)(cKVt), 2, axis=-1
        )
        q = Dense(features=self.model_dimension, dtype=self.model_dtype)(cqt)

        qkv = jnp.concat([q, k, v], axis=0)
        qkv = rearrange(
            qkv,
            "B T (nh dk) -> B nh T dk",
            B=B * 3,
            nh=self.n_heads,
            dk=C // self.n_heads,
        )

        q, k, v = jnp.split(qkv, 3, axis=0)

        if use_rope:


            q = jnp.concatenate([q, qRt], axis=-1)
            kRt = jnp.repeat(kRt[:, None, :, :], self.n_heads, axis=1)

            k = jnp.concatenate([k, kRt], axis=-1)

        def scaledDotProd(q, k, v, mask):
            input_dtype = q.dtype

            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)
            v = v.astype(jnp.float32)

            dk = q.shape[-1]

            w = jnp.einsum("B n T d, B n t d -> B n T t", q, k) * (dk ** -0.5)
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


        output = rearrange(output, "B nh T dk -> B T (nh dk)")

        output = Dense(features=self.model_dimension, dtype=self.model_dtype)(output)
        output = nn.Dropout(rate=self.dropout)(output, deterministic=not train)

        return output, (cKV_cache, kRT_cache)

class Layer(nn.Module):
    model_dimension: int
    n_heads: int
    T: int
    latent_dim: int
    dhR: int
    dropout_rate: float = 0.1
    model_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, cache, train=True):
        x_res = x

        x = RMSNorm(model_dtype=self.model_dtype)(x)
        x, cache = MLA(
            model_dimension=self.model_dimension,
            n_heads=self.n_heads,
            T=self.T,
            latent_dim=self.latent_dim,
            dhR=self.dhR,
            model_dtype=self.model_dtype,
            dropout=self.dropout_rate,
        )(x, cKV_cache=cache[0], kRT_cache=cache[1],train=train)

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
    def __call__(self, x, cache=None, train=True):
        out_cache = []
        for i in range(self.layers):
            current_cache = (*cache[i],) if cache is not None else (None, None)

            x, cache_out = Layer(
                model_dimension=self.model_dimension,
                n_heads=self.n_heads,
                T=self.T,
                latent_dim=self.latent_dim,
                dhR=self.dhR if i < self.layers - 1 else 0,
                dropout_rate=self.dropout_rate,
                model_dtype=self.model_dtype
            )(x, current_cache, train=train)
            out_cache.append(cache_out)

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
    def __call__(self, x, cache=None, train=True):

        M, B, T = x.shape
        x = x.reshape((M * B, T))
        embedding = Embedding(
            vocab_size=self.vocab_size,
            model_dimension=self.model_dimension,
            model_dtype=self.model_dtype,
        )

        x = embedding(x)
        out_cache = []
        for i in range(self.blocks):
            current_cache = cache[i] if cache is not None else None

            x, cache_out = Block(
                layers=self.layers_per_block,
                model_dimension=self.model_dimension,
                n_heads=self.n_head,
                T=self.T,
                latent_dim=self.latent_dim,
                dhR=self.dhR,
                dropout_rate=self.dropout_rate,
                model_dtype=self.model_dtype
            )(x, current_cache, train=train)
            out_cache.append(cache_out)

        x_out = embedding(x, out=True)
        x_out = x_out.reshape((M, B, T, self.vocab_size))
        return x_out, out_cache


    def generate(self):
        pass
