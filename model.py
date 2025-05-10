import jax
import jax.numpy as jnp
from jax import random
import math
from typing import Callable
from einops import rearrange
import flax
from flax import linen as nn

class Embeddings(nn.Module):
    model_dimension : int
    vocab_size : int

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.model_dimension)
    
    def __call__(self, x):
        x = self.embedding(x) * math.sqrt(self.model_dimension)
        return x

class RoPE(nn.Module):
    model_dim : int
    t : int

    def setup(self):
        zeros = jnp.zeros((self.t,self.model_dim))
        self.array_m = jnp.arange(self.t)[:, None] + zeros
        self.pos = ((jnp.arange(self.model_dim)[None, :] + zeros) // 2)
        self.theta = 10000*(-2*(self.pos - 1)/self.model_dim)
        self.matrix1 = jnp.cos(self.array_m * self.theta)
        self.matrix2 = jnp.sin(self.array_m * self.theta)
        
    def __call__(self, x):
        a = jnp.reshape(x, (-1, 2, self.model_dim))
        a = a[:, ::-1, :]
        a = jnp.reshape(a, (a.shape[0] * a.shape[1], self.model_dim))
        result = (x * self.matrix1) + (a * self.matrix2)
        return result

class MLA(nn.Module):
    model_dim : int 
    n_heads : int
    max_tokens : int
    latent_dim : int 
    dhR : int
    t : int
    
    def setup(self):
        self.Wdkv = nn.Dense(features=self.latent_dim)
        self.Wuk = nn.Dense(features=self.model_dim)
        self.Wuv = nn.Dense(features=self.model_dim)
        self.Wdq = nn.Dense(feauters=self.latent_dim)
        self.Wuq = nn.Dense(features=self.model_dim)

        self.dk = self.model_dim // self.n_heads
        self.output = nn.Dense(features=self.model_dim) 
        self.cKV_cache = None
        self.kRT_cache = None
        self.cache_ind = 0

        self.Wkr = nn.Dense(features=self.dhR)
        self.Wqr = nn.Dense(features=(self.dhR*self.n_heads))
        self.rope = RoPE(model_dim=self.model_dim, t=self.t)

    def __call__(self, x, train=True):
        if train == False:
            x = x[:, -1:, :]
        
        cKVt = self.Wdkv(x)
        cqt = self.Wdq(x)
        
        if self.dhR != 0: 
            kRt = self.rope(self.Wkr(x))
            qrt = self.rope(self.Wqr(cqt))
            qrt = rearrange(qrt, 'B T C -> B nh T dk', nh=self.n_heads, dk=self.dk)

        B, T ,C = x.shape 
        if train == False:
            if self.cKV_cache == None:
                self.cKV_cache = nn.zeros((B, self.max_tokens, self.dc))
            
            if self.dhR != 0:
                if self.kRT_cache == None:
                    self.kRT_cache = nn.zeros((B, self.max_tokens, self.dhR))
                
                self.kRT_cache.at[:, self.cache_ind:self.cache_ind+T, :].set(kRt)
                kRt = self.kRT_cache[:, :self.cache_ind+T, :]

            self.cKV_cache.at[:, self.cache_ind:self.cache_ind+T, :].set(cKVt)

            cKVt = self.cKV_cache[:, :self.cache_ind+T, :]

            self.cache_ind = min(self.cache_ind + T, self.max_tokens-1)

            if (self.cache_ind == self.max_tokens-1):
                self.cKV_cache.at[:, :-1, :].set(self.cKV_cache[:, 1:, :])

                if self.dhR != 0:
                    self.kRT_cache.at[:, :-1, :].set(self.kRT_cache[:, 1:, :])

        k_c = self.Wuk(cKVt) 
        k_c = rearrange(k_c, 'B T C -> B nh T dk', nh=self.n_heads, dk=self.dk)
        if self.dhR != 0:
            k_r = kRt[:, None, ...].repeat(axis=1, total_repeat_length=self.n_heads)
            k = jnp.concatenate([k_c, kRt], axis=-1)
        else:
            k = k_c

        v = self.Wuv(cKVt)
        v = rearrange(v, 'B T C -> B nh T dk', nh=self.n_heads, dk=self.dk)

        q = self.Wuq(cqt)
        q_c = rearrange(q, 'B T C -> B nh T dk', nh=self.n_heads, dk=self.dk)

        if self.dhR != 0:
            q = jnp.concatenate([q, qrt], axis=-1)
       
        weights = jnp.einsum('B nh T dk, B nh t dk -> B nh T t', q, k) * (1/ ((self.dk) ** 0.5))

        if train == True:
            size = weights.shape[-1]
            mask = jnp.tril(jnp.ones((B, self.n_heads, size, size)))
            weights = jnp.where(mask == 0, -9e15, weights)
            
        logits = nn.softmax(weights, axis=-1)
        attention = jnp.einsum('B nh T t, B nh t dk -> B nh T dk', logits, v)
        attention = rearrange(attention, 'B nh T dk -> B T (nh dk)')
        output = self.output(attention)
        
        return output

class LayerNorm(nn.Module):
    model_dimension : int
    gamma_init : Callable = nn.initializers.lecun_normal()
    beta_init : Callable = nn.initializers.lecun_normal()

    def setup(self):
        self.gamma = self.param('gamma', self.gamma_init, self.model_dimension)
        self.beta = self.param('beta',self.beta_init, self.model_dimension)
        self.eps = 1e-05
    
    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        norm = ((x - mean)/jnp.sqrt(var + self.eps))
        y = jnp.einsum('B T C, C -> B T C', norm, self.gamma) + self.beta[None, None, :]
        
        return y

class Expert(nn.Module):
    model_dimension : int
    ff_dim : int
    dropout : float

    def setup(self):
        self.linear1 = nn.Dense(features=self.ff_dim)
        self.linear2 = nn.Dense(features=self.model_dimension)
        self.dropout = nn.Dropout(rate=self.dropout) 
    
    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class NoisyKGate(nn.Module):
    model_dimension : int
    n_experts : int
    k : int
    dropout : float

    def setup(self):
        self.rng = jax.random.PRNGKey(42)
        self.Wg = nn.Dense(features=self.n_experts)
        self.Wnoise = nn.Dense(features=self.n_experts)
    
    def top(self, x):
        k = self.k
        y,i = jax.lax.top_k(x, k)
        y = nn.softmax(y)
        return y, i
    
    def __call__(self, x):
        b = x.shape[0]
        t = x.shape[1]
        Hx = self.Wg(x) + ((jax.random.normal(self.rng, shape=(b, t, self.n_experts))) * nn.softplus(self.Wnoise(x)))
        g_scores, indices = jnp.apply_along_axis(func1d=self.top, axis=-1, arr=Hx)
        return g_scores, indices

class MoE(nn.Module):
    model_dimension : int
    n_experts : int
    k : int
    dropout : float
    
    def setup(self):
        self.experts = [Expert(model_dimension=self.model_dimension, ff_dim=4*self.model_dimension, dropout=self.dropout) for i in range(self.n_experts)]
        self.gate = NoisyKGate(model_dim=self.model_dimension, n_experts=self.n_experts, k=self.k, dropout=self.dropout)

    def gScores(self, scores, indices, x):
        expert = lambda i : self.experts[i](x) # (C) -> (C)
        expert_parallel = jax.vmap(fun=expert, in_axes=(0), out_axes=(0)) 

        experts = expert_parallel(indices) # (K) -> (K, C)
        gscore = scores[:, None] * experts #(K, 1), (K, C) -> (K, C)
        gscore = jnp.sum(gscore, axis=0) #(K, C) -> C
        return gscore
    
    def __call__(self, x):
        s, i= self.gate(x)
        gscore_parallel = jax.vmap(fun=jax.vmap(fun=self.gScores, in_axes=(0,0,0), out_axes=(0)), in_axis=(0,0,0), out_axes=(0))
        res = gscore_parallel(s, i, x)
        return res

class FeedForward(nn.Module):
    model_dimension : int
    ff_dim : int
    dropout : float

    def setup(self):
        self.linear1 = nn.Dense(features=self.ff_dim)
        self.linear2 = nn.Dense(features=self.model_dimension)
        self.dropout = nn.Dropout(rate=self.dropout) 
    
    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Block(nn.Module):
    model_dimension : int
    n_heads : int
    dropount : float
    max_tokens : int
    latent_dim : int 
    dhR : int
    t : int
    n_experts : int
    k : int
    moe : bool
    
    def setup(self):
        self.attention = MLA(model_dim=self.model_dimension, n_heads=self.n_heads, max_tokens=self.max_tokens, latent_dim=self.latent_dim, dhR=self.dhR, t=self.t)
        self.norm1 = LayerNorm(model_dimension=self.model_dimension)
        self.norm2 = LayerNorm(model_dimension=self.model_dimension)
        if (self.moe == True): 
            self.ff =  MoE(model_dimension=self.model_dimension, n_experts=self.n_experts, k=self.k, dropout=self.dropount)
        else: 
            self.ff = FeedForward(model_dimension=self.model_dimension, ff_dim=4*self.model_dimension, dropout=self.dropout)
        
    def __call__(self, x, train=True):
        x = self.norm1(x + self.attention(x, train=train))
        x = self.norm2(x + self.ff(x))
        return x
    
class Decoder(nn.Module):
    model_dimension : int
    n_heads : int
    seq_len : int
    vocab_size : int
    dropout : float
    blocks : int
    n_experts : int 
    k : int
    moe : bool
    max_tokens : int
    latent_dim : int
    dhR : int

    def setup(self):
        self.ratio = 4
        ind = 0
        self.embeddingTable = Embeddings(model_dimension=self.model_dimension, vocab_size=self.vocab_size)
        self.Blocks = [Block
                       (model_dimension=self.model_dimension, 
                        n_heads=self.n_heads, 
                        dropout=self.dropout, 
                        max_tokens=self.max_tokens, 
                        latent_dim=self.latent_dim, 
                        dhR=0 if i % 4 == 0 else self.dhR, 
                        t=self.seq_len, 
                        n_experts=self.n_experts, 
                        k=self.k, MoE=self.moe) 
                        for i in range(self.blocks)]
        self.linear = nn.Dense(features=self.vocab_size)

    def __call__(self, x, train=True):
        x = self.embeddingTable(x) #B,T,C
        x = [Block(x, train=train) for Block in self.blocks]
        x = self.linear(x)
        result = nn.softmax(x, axis=-1)
        return result