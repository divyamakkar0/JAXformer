import jax
import jax.numpy as jnp
from jax import random
import math
from typing import Callable
from einops import rearrange
import flax
from flax import linen as nn


class Embeddings(nn.Module):
    model_dimension: int
    vocab_size: int

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size, features=self.model_dimension
        )

    @nn.compact
    def __call__(self, x):
        x = self.embedding(x) * math.sqrt(self.model_dimension)
        return x


class RoPE:
    def __init__(self, T, model_dim):
        self.T = T
        self.model_dim = model_dim
        assert model_dim % 2 == 0, "model_dim must be even"

        freq = jnp.arange(self.T)[:, None]
        pos = jnp.arange(self.model_dim // 2)[:, None].repeat(2, axis=-1).reshape(1, -1)
        theta = 10000 ** (-2 * pos / self.model_dim)
        self.cos = jnp.cos(freq * theta)
        self.sin = jnp.sin(freq * theta)

    def __call__(self, x, t_start, t_end):
        B, nh, T, C = x.shape
        assert t_end - t_start == T, "T of x must be the same as T indices"

        cos_rope = x * self.cos[None, None, t_start:t_end, :]
        x_inter = x.reshape((B, nh, T, C // 2, 2))
        x_inter = jnp.flip(x_inter, axis=-1) * jnp.array([-1, 1])
        x_inter = x_inter.reshape((B, nh, T, C))
        x_pos = cos_rope + x_inter * self.sin[None, None, t_start:t_end, :]

        return x_pos


class MLA(nn.Module):
    model_dim: int
    n_heads: int
    T: int
    latent_dim: int
    dhR: int

    def setup(self):
        self.W_down = nn.Dense(features=2 * self.latent_dim)
        self.W_uKV = nn.Dense(features=2 * self.model_dim)
        self.W_uQ = nn.Dense(features=self.model_dim)

        self.dk = self.model_dim // self.n_heads
        self.output = nn.Dense(features=self.model_dim)

        self.rope = None

        if self.dhR != 0:
            self.Wkr = nn.Dense(features=self.dhR)
            self.Wqr = nn.Dense(features=(self.dhR * self.n_heads))
            self.rope = RoPE(model_dim=self.dhR, T=self.T)

    def __call__(self, x, cKV_cache=None, kRT_cache=None, attention_mask=None, train=True):
        B, T, C = x.shape
        if train == False:
            x = x[:, -1:, :]

        cKVt, cqt = jnp.split(self.W_down(x), 2, axis=-1)

        if self.rope:
            t_start = T - 1 if not train else 0
            kRt = self.rope(self.Wkr(x)[:, None, ...], t_start, T)
            kRt = kRt.repeat(self.n_heads, axis=1)

            qrt = rearrange(
                self.Wqr(x), "B T (nh d) -> B nh T d", nh=self.n_heads, d=self.dhR
            )
            qrt = self.rope(qrt, t_start, T)

        if not train:
            if cKV_cache is None:
                cKV_cache = cKVt
            else:
                cKV_cache = jnp.concatenate([cKV_cache, cKVt], axis=1)
            cKVt = cKV_cache

            if self.rope:
                if kRT_cache is None:
                    kRT_cache = jnp.zeros((B, 1, self.dhR))
                else:
                    kRT_cache = jnp.concatenate([kRT_cache, kRt[:, 0, :, :]], axis=1)
                kRt = kRT_cache[:, None, ...].repeat(self.n_heads, axis=1)

            if cKV_cache.shape[1] >= self.T:
                cKV_cache = cKV_cache[:, -self.T:, :]
                if self.rope:
                    kRT_cache = kRT_cache[:, -self.T:, :]

        v_k = rearrange(
            self.W_uKV(cKVt), "B T (nh d) -> B nh T d", nh=self.n_heads, d=2 * self.dk
        )
        v, k = jnp.split(v_k, 2, axis=-1)

        if self.rope:
            k = jnp.concatenate([k, kRt], axis=-1)

        q = self.W_uQ(cqt)
        q = rearrange(q, "B T (nh dk) -> B nh T dk", nh=self.n_heads, dk=self.dk)

        if self.rope:
            q = jnp.concatenate([q, qrt], axis=-1)

        weights = jnp.einsum("B n T d, B n t d -> B n T t", q, k) * (
            1 / ((self.dk) ** 0.5)
        )

        if train == True:
            size = weights.shape[-1]
            mask = jnp.tril(jnp.ones((B, self.n_heads, size, size)))
            weights = jnp.where(mask == 0, -9e15, weights)
            weights = jnp.where(attention_mask == 0, -9e15, weights) # add attention mask in case we need to pad

        weights = nn.softmax(weights, axis=-1)

        output = jnp.einsum("B n T t, B n t d -> B n T d", weights, v)
        output = rearrange(output, "B nh T dk -> B T (nh dk)")
        output = self.output(output)

        return output, (cKV_cache, kRT_cache)


class LayerNorm(nn.Module):
    model_dimension: int
    gamma_init: Callable = nn.initializers.lecun_normal()
    beta_init: Callable = nn.initializers.lecun_normal()

    def setup(self):
        self.gamma = self.param("gamma", self.gamma_init, (1, 1, self.model_dimension))
        self.beta = self.param("beta", self.beta_init, (1, 1, self.model_dimension))
        self.eps = 1e-05

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        norm = (x - mean) / jnp.sqrt(var + self.eps)
        y = norm * self.gamma + self.beta

        return y


class NoisyKGate(nn.Module):
    model_dimension: int
    n_experts: int
    k: int

    def setup(self):
        self.rng = jax.random.PRNGKey(42)
        self.Wg = nn.Dense(features=self.n_experts)
        self.Wnoise = nn.Dense(features=self.n_experts)

    def top(self, x):
        k = self.k
        y, i = jax.lax.top_k(x, k)
        y = nn.softmax(y)
        return y, i

    def __call__(self, x):
        b = x.shape[0]
        t = x.shape[1]
        Hx = self.Wg(x) + (
            (jax.random.normal(self.rng, shape=(b, t, self.n_experts)))
            * nn.softplus(self.Wnoise(x))
        )
        g_scores, indices = jnp.apply_along_axis(func1d=self.top, axis=-1, arr=Hx)
        return g_scores, indices


class MoE(nn.Module):
    model_dimension: int
    n_experts: int
    k: int
    dropout: float

    def setup(self):
        self.experts = [
            FeedForward(
                model_dimension=self.model_dimension,
                ff_dim=4 * self.model_dimension,
                dropout=self.dropout,
            )
            for i in range(self.n_experts)
        ]
        self.gate = NoisyKGate(
            model_dimension=self.model_dimension, n_experts=self.n_experts, k=self.k
        )

    def get_gScores(self, scores, indices, x, train=True):
        expert_lambda = [
            lambda mdl, x: mdl.experts[i](x) for i in range(mdl.n_experts)
        ]

        if self.is_mutable_collection("params"):
            for expert_ffn in expert_lambda:
                _ = expert_ffn(self, x)

        expert_fn = lambda j, experts, x: nn.switch(j, expert_lambda, self, x)
        expert_parallel = jax.vmap(fun=expert_fn, in_axes=(0, None, None), out_axes=(0))

        expert_scores = expert_parallel(indices, self.experts, x)  # (K) -> (K, C)
        gScore = scores[:, None] * expert_scores  # (K, 1), (K, C) -> (K, C)
        gScore = jnp.sum(gScore, axis=0)  # (K, C) -> C

        return gScore

    def __call__(self, x, train=True):
        s, i = self.gate(x)
        gscore_parallel = jax.vmap(
            fun=jax.vmap(fun=lambda s, i, x : self.get_gScores(s,i,x,train=train), in_axes=(0, 0, 0), out_axes=(0)),
            in_axes=(0, 0, 0),
            out_axes=(0),
        )
        res = gscore_parallel(s, i, x)
        return res


class FeedForward(nn.Module):
    model_dimension: int
    ff_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(features=self.ff_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        x = nn.Dense(features=self.model_dimension)(x)

        return x

class Block(nn.Module):
    model_dimension: int
    n_heads: int
    dropout: float
    T: int
    latent_dim: int
    dhR: int = 0
    n_experts: int = 0
    k: int = 0
    moe: bool = False

    @nn.compact
    def __call__(self, x, cache=(None, None), train=True):
        x_up, cache = MLA(
            model_dim=self.model_dimension,
            n_heads=self.n_heads,
            T=self.T,
            latent_dim=self.latent_dim,
            dhR=self.dhR,
        )(x, *cache, train=train)

        x = LayerNorm(model_dimension=self.model_dimension)(x + x_up)

        if self.moe == True:
            ff = MoE(
                model_dimension=self.model_dimension,
                n_experts=self.n_experts,
                k=self.k,
                dropout=self.dropout,
            )
        else:
            ff = FeedForward(
                model_dimension=self.model_dimension,
                ff_dim=4 * self.model_dimension,
                dropout=self.dropout,
            )

        x = LayerNorm(model_dimension=self.model_dimension)(x + ff(x=x, train=train))

        return x, cache


class Decoder(nn.Module):
    model_dimension: int
    n_heads: int
    dhR: int
    rope_ratio: int
    T: int
    vocab_size: int
    dropout: float
    blocks: int
    n_experts: int
    k: int
    moe: bool
    latent_dim: int

    @nn.compact
    def __call__(self, x, cache=None, train=True):

        embed = Embeddings(
            model_dimension=self.model_dimension, vocab_size=self.vocab_size
        )
        x = embed(x)

        out_cache = []
        for i in range(self.blocks):
            if cache is None:
                layer_cache = (None, None)
            else:
                layer_cache = cache[i]
            x, current_cache = Block(
                model_dimension=self.model_dimension,
                n_heads=self.n_heads,
                dropout=self.dropout,
                T=self.T,
                latent_dim=self.latent_dim,
                dhR=0 if (self.rope_ratio == 0 or i % self.rope_ratio == 0) else self.dhR,
                n_experts=self.n_experts,
                k=self.k,
                moe=self.moe,
            )(x, cache=layer_cache, train=train)
            out_cache.append(current_cache)

        x = x @ embed.embedding.embedding.T
        if not train:
            x = nn.softmax(x, axis=-1)

        return x, out_cache

    @classmethod
    def get_model(cls, model_config, init_key: jax.random.key):
        x = jnp.ones((1, model_config.T), dtype=jnp.int32)

        model = cls(model_config.model_dimension,
                        model_config.n_heads,
                        model_config.dhR,
                        model_config.rope_ratio,
                        model_config.T,
                        model_config.vocab_size,
                        model_config.dropout,
                        model_config.blocks,
                        model_config.n_experts,
                        model_config.k,
                        model_config.moe,
                        model_config.latent_dim,
                        )

        params = model.init(
            init_key,
            x,
            train=False,
        )['params']

        return model, params


if __name__ == "__main__":
    model = Decoder(
        model_dimension=64,
        n_heads=4,
        dhR=0,
        rope_ratio=0,
        T=32,
        vocab_size=10000,
        dropout=0.1,
        blocks=2,
        n_experts=4,
        k=2,
        moe=True,
        latent_dim=16
    )
    key = jax.random.key(0)
    key, init_key, dropout_key = jax.random.split(key, 3)

    x = jax.random.randint(init_key, (16, 32), 0, 32, dtype=jnp.int32)
    params = model.init(init_key, x, train=True)["params"]
    print("Model parameters initialized successfully.")
    x = model.apply({"params": params}, x, train=True, rngs={"dropout": dropout_key})
    print(x.shape)
