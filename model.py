import jax
import jax.numpy as jnp
from jax import random
import math
from typing import Callable
from einops import rearrange
import flax
from flax import linen as nn
import tiktoken

from jax.numpy import dtype
from config import parse_args


class Embeddings(nn.Module):
    model_dimension: int
    vocab_size: int
    model_dtype: jnp.dtype

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dimension,
            dtype=self.model_type,
        )

    @nn.compact
    def __call__(self, x):
        x = self.embedding(x) * math.sqrt(self.model_dimension)
        return x


class RoPE:
    def __init__(self, T, model_dim, model_dtype):
        self.T = T
        self.model_dim = model_dim
        self.model_dtype = model_dtype
        assert model_dim % 2 == 0, "model_dim must be even"


        freq = jnp.arange(self.T, dtype=jnp.float32)[:, None]
        pos = (
            jnp.arange(self.model_dim // 2, dtype=jnp.float32)[:, None]
            .repeat(2, axis=-1)
            .reshape(1, -1)
        )

        theta = 10000 ** (-2 * pos / self.model_dim)
        self.cos = jnp.cos(freq * theta, dtype=self.model_dtype)
        self.sin = jnp.sin(freq * theta, dtype=self.model_dtype)

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
    model_dtype: jnp.dtype
    grad_checkpoint: bool 

    def setup(self):
        self.W_down = nn.Dense(features=2 * self.latent_dim, dtype=self.model_dtype)
        self.W_uKV = nn.Dense(features=2 * self.model_dim, dtype=self.model_dtype)
        self.W_uQ = nn.Dense(features=self.model_dim, dtype=self.model_dtype)

        self.dk = self.model_dim // self.n_heads
        
        if self.grad_checkpoint: 
            self.output = nn.remat(nn.Dense)(features=self.model_dim, dtype=self.model_dtype)
        else:
            self.output = nn.Dense(features=self.model_dim, dtype=self.model_dtype)

        self.rope = None

        if self.dhR != 0:
            self.Wkr = nn.Dense(features=self.dhR, dtype=self.model_dtype)
            self.Wqr = nn.Dense(
                features=(self.dhR * self.n_heads), dtype=self.model_dtype
            )
            self.rope = RoPE(model_dim=self.dhR, T=self.T, dtype=self.model_dtype)

    def __call__(
        self, x, cKV_cache=None, kRT_cache=None, attention_mask=None, train=True
    ):
        B, T, C = x.shape
        t_idx = 0
        if train == False and cKV_cache is not None:
            t_idx = cKV_cache.shape[1]
        x = x[:, t_idx:, :]
        cKVt, cqt = jnp.split(self.W_down(x), 2, axis=-1)

        if self.rope:
            kRt = self.rope(self.Wkr(x)[:, None, ...], t_idx, T)
            kRt = kRt.repeat(self.n_heads, axis=1)

            qRt = rearrange(
                self.Wqr(x), "B T (nh d) -> B nh T d", nh=self.n_heads, d=self.dhR
            )
            qRt = self.rope(qRt, t_idx, T)

        if not train:
            if cKV_cache is None:
                cKV_cache = cKVt
            else:
                cKV_cache = jnp.concatenate([cKV_cache, cKVt], axis=1)
            cKVt = cKV_cache

            if self.rope:
                krt_head = kRt[:, 0, :, :]
                if kRT_cache is None:
                    kRT_cache = krt_head
                else:
                    kRT_cache = jnp.concatenate([kRT_cache, krt_head], axis=1)
                kRt = kRT_cache[:, None, ...].repeat(self.n_heads, axis=1)

            if cKV_cache.shape[1] >= self.T:
                cKV_cache = cKV_cache[:, -self.T + 1 :, :]
                if self.rope:
                    kRT_cache = kRT_cache[:, -self.T + 1 :, :]

        v_k = rearrange(
            self.W_uKV(cKVt), "B T (nh d) -> B nh T d", nh=self.n_heads, d=2 * self.dk
        )
        v, k = jnp.split(v_k, 2, axis=-1)

        if self.rope:
            k = jnp.concatenate([k, kRt], axis=-1)

        q = self.W_uQ(cqt)
        q = rearrange(q, "B T (nh dk) -> B nh T dk", nh=self.n_heads, dk=self.dk)

        if self.rope:
            q = jnp.concatenate([q, qRt], axis=-1)

        def scaledDotProd(q, k, v, mask):
            w = jnp.einsum("B n T d, B n t d -> B n T t", q, k) * (
                    1 / ((self.dk) ** 0.5)
            )

            w = jnp.where(mask==0, -9e15, w)
            w = nn.softmax(w, axis=-1)
            
            output = jnp.einsum("B n T t, B n t d -> B n T d", w, v)

            return output

        if self.grad_checkpoint:
            scaledDotProd = jax.remat(scaledDotProd)

        mask = jnp.ones((B, self.n_heads, T, T))
        if train:
            mask = jnp.tril(mask)

        output = scaledDotProd(q,k,v, mask)
        output = rearrange(output, "B nh T dk -> B T (nh dk)")

        output = self.output(output)

        return output, (cKV_cache, kRT_cache)


class LayerNorm(nn.Module):
    model_dimension: int
    model_dtype: jnp.dtype
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
        y = jnp.asarray(y, self.model_dtype)

        return y


class NoisyKGate(nn.Module):
    model_dimension: int
    n_experts: int
    k: int
    model_dtype: jnp.dtype

    def setup(self):
        self.centroids = nn.Dense(features=self.n_experts, dtype=self.model_dtype)

    def top(self, x):
        assert x.shape[0] == self.n_experts, "x must be of shape (n_experts, )"
        g_i, i = jax.lax.top_k(x, self.k)
        g = jnp.zeros((x.shape[0], ), dtype=x.dtype)
        g = g.at[i].set(g_i)
        g = g / jnp.sum(g, axis=-1)
        g = g[i]

        return g, i

    def __call__(self, x):
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
            features= self.model_dimension * self.n_shared,
        )
        self.experts = [
            FeedForward(
                model_dimension=self.model_dimension,
                ff_dim=4 * self.model_dimension,
                dropout=self.dropout,
                model_dtype=self.model_dtype,
                grad_checkpoint=self.grad_checkpoint
            )
            for i in range(self.n_experts)
        ]
        self.gate = NoisyKGate(
            model_dimension=self.model_dimension,
            n_experts=self.n_experts,
            k=self.k,
            model_dtype=self.model_dtype,
        )

    def get_gScores(self, scores, indices, x, train=True):
        expert_lambda = [
            lambda mdl, x: mdl.experts[i](x, train=train) for i in range(self.n_experts)
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

        B,T,C = x.shape

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
    def __call__(self, x):
        x = nn.Dense(features=self.ff_dim, dtype=self.model_dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.model_dimension, dtype=self.model_dtype)(x)
         
        return x

class FeedForward(nn.Module):
    model_dimension: int
    ff_dim: int
    dropout: float
    model_dtype: jnp.dtype
    grad_checkpoint: bool

    @nn.compact
    def __call__(self, x, train: bool = True):
        ff = FFBody
        if self.grad_checkpoint:
            ff = nn.remat(FFBody)

        ff = ff(
            model_dimension=self.model_dimension,
            ff_dim=4 * self.model_dimension,
            dropout=self.dropout,
            model_dtype=self.model_dtype
        )
        x_ff = nn.Dropout(rate=self.dropout,deterministic=not train)(ff(x))

        return x_ff


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
    def __call__(self, x, cache=(None, None), train=True):
        
        x_norm = LayerNorm(
            model_dimension=self.model_dimension,
            model_dtype=self.model_dtype,
        )(x)
        
        x_up, cache = MLA(
                model_dim=self.model_dimension,
                n_heads=self.n_heads,
                T=self.T,
                latent_dim=self.latent_dim,
                dhR=self.dhR,
                model_dtype=self.model_dtype,
                grad_checkpoint=self.grad_checkpoint
            )(x_norm, *cache,attention_mask=None, train=train)

        x = x + x_up
        
        x_norm =  LayerNorm(
                    model_dimension=self.model_dimension
                    model_dtype=self.model_dtype,
                  )(x)

        load = None
        if self.moe == True:
            x_ff, load = MoE(
                model_dimension=self.model_dimension,
                n_experts=self.n_experts,
                k=self.k,
                dropout=self.dropout,
                model_dtype=self.model_dtype,
                n_shared=self.n_shared,
                grad_checkpoint=self.grad_checkpoint
            )(x_norm, train=train)

        else:
            x_ff = FeedForward(
                model_dimension=self.model_dimension,
                ff_dim=4 * self.model_dimension,
                dropout=self.dropout,
                model_dtype=self.model_dtype,
                grad_checkpoint=self.grad_checkpoint
            )(x_norm, train=train) 

        x = x + x_ff

        return x, (cache, load)


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
    n_shared: int
    k: int
    moe: bool
    latent_dim: int
    model_dtype: jnp.dtype
    grad_checkpoint: bool

    @nn.compact
    def __call__(self, x, cache=None, train=True):
        embed = Embeddings(
            model_dimension=self.model_dimension,
            vocab_size=self.vocab_size,
            model_dtype=self.model_type,
        )
        x = embed(x)

        out_cache = []
        load = None
        for i in range(self.blocks):
            if cache is None:
                layer_cache = (None, None)
            else:
                layer_cache = cache[i]

            x, (current_cache, current_load) = Block(
                        model_dimension=self.model_dimension,
                        n_heads=self.n_heads,
                        dropout=self.dropout,
                        T=self.T,
                        latent_dim=self.latent_dim,
                        dhR=0
                        if (self.rope_ratio == 0 or i % self.rope_ratio == 0)
                        else self.dhR,
                        moe=self.moe,
                        n_experts=self.n_experts,
                        n_shared=self.n_shared,
                        k=self.k,
                        model_dtype=self.model_type,
                    )(x, cache=layer_cache, train=train)
            if load is None:
                load = current_load
            else:
                add_tree = lambda x,y: jax.tree.map(lambda a, b: a + b, x, y)
                load = (add_tree(load[0], current_load[0]), add_tree(load[1], current_load[1]))

            out_cache.append(current_cache)

        x = x @ embed.embedding.embedding.T
        x = jnp.asarray(x, dtype=jnp.float32)

        if load is not None:
            load = (load[0] * load[1]).mean(axis=0)

        return x, (out_cache, load)

    @classmethod
    def get_model(cls, model_config, init_key: jax.random.key):
        x = jnp.ones((1, model_config.T), dtype=jnp.int32)

        model = cls(
            model_dimension=model_config.model_dimension,
            n_heads=model_config.n_heads,
            dhR=model_config.dhR,
            rope_ratio=model_config.rope_ratio,
            T=model_config.T,
            vocab_size=model_config.vocab_size,
            dropout=model_config.dropout,
            blocks=model_config.blocks,
            n_experts=model_config.n_experts,
            k=model_config.k,
            moe=model_config.moe,
            latent_dim=model_config.latent_dim,
            n_shared=model_config.n_shared,
            model_type=jnp.bfloat16
            if (model_config.model_dtype == "bfloat16")
            else jnp.float32,
        )

        params = model.init(
            init_key,
            x,
            train=False,
        )["params"]

        return model, params

    def generate(
        self, params, key, x: str = "", B=1, k=10000, temperature=1, max_tokens=100
    ):
        enc = tiktoken.get_encoding("cl100k_base")
        cache = None

        start_of_text = jnp.array(
            [enc._special_tokens["<|endoftext|>"]], dtype=jnp.int32
        )
        x_encoded = jnp.array(enc.encode(x), dtype=jnp.int32)
        x = jnp.concatenate([start_of_text, x_encoded], axis=-1)
        x = jnp.repeat(x[None, :], B, axis=0)

        out = x

        for i in range(max_tokens):
            x = out[:, -self.T :]
            logits, (cache, _) = self.apply({"params": params}, x, cache, train=False)

            logits = logits[:, -1, :] / temperature
            k_scores, _ = jax.lax.top_k(logits, k)
            probs = nn.softmax(k_scores, axis=-1)

            key, sample_key = jax.random.split(key)
            out_next = jax.random.categorical(sample_key, probs, axis=-1)[:, None]
            out = jnp.concatenate([out, out_next], axis=-1)

        tokens = jax.device_get(out[:, 1:])
        decode_fn = lambda x: enc.decode(x)
        outputs = list(map(decode_fn, tokens))

        return outputs


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
        n_shared=2,
        k=2,
        moe=True,
        latent_dim=16,
    )
    key = jax.random.key(0)
    key, init_key, dropout_key = jax.random.split(key, 3)

    x = jax.random.randint(init_key, (16, 32), 0, 32, dtype=jnp.int32)
    params = model.init(init_key, x, train=True)["params"]
    print("Model parameters initialized successfully.")
    x = model.apply({"params": params}, x, train=True, rngs={"dropout": dropout_key})
    print(x.shape)
