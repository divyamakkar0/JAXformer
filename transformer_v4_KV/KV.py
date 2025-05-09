import jax
import jax.numpy as jnp
from jax import random
import math
from typing import Callable
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
    
class PositionalEncoding(nn.Module):
    model_dimension : int
    seq_len : int

    def setup(self):
        zeros = jnp.zeros((self.seq_len, self.model_dimension))
        row = jnp.arange(0, self.seq_len, 1)
        pos_matrix = row.reshape(-1, 1)
        row2 = jnp.arange(0, self.model_dimension, 1)
        pos2 = zeros + row2
        j = pos2 // 2
        denom = 1/(10000 ** ((2*j)/self.model_dimension))
        inp = pos_matrix * denom
        pi2j = jnp.sin(inp)
        pi2j1 = jnp.cos(inp)
        zeros = zeros.at[:, 0::2].set(pi2j[:, ::2])
        zeros = zeros.at[:, 1::2].set(pi2j1[:, 1: :2])
        self.encodings = zeros
        
    def __call__(self, x):
        return x + self.encodings

class ScaledDotProduct(nn.Module):
    dk : int 
    max_token_len : int

    def setup(self):
        self.W = nn.Dense(features=3*self.dk)
        self.key_cache = None
        self.value_cache = None
        self.token_ind = 0

    def __call__(self, x, train:bool = True):
        x = x[:, -1:, :]
        B, T, C = x.shape
        qkv = self.W(x) #still works when q, k, v is t x dk  or b x t x dk
        q,k,v = jnp.split(qkv, 3, axis=-1)
        if train == False:
            if self.key_cache is None or self.value_cache is None:
                self.key_cache = jnp.zeros((B, self.max_token_len, self.dk))
                self.value_cache = jnp.zeros((B, self.max_token_len, self.dk))


            self.key_cache = self.key_cache.at[:, self.token_ind:self.token_ind + T, :].set(k)
            self.value_cache = self.value_cache.at[:, self.token_ind:self.token_ind + T, :].set(v)

            k = self.key_cache[:, :self.token_ind + T,:]
            v = self.value_cache[:, :self.token_ind + T, :]

            self.token_ind = min(self.token_ind + T, self.max_token_len-1)

            if self.token_ind == self.max_token_len - 1:
                self.key_cache.at[:,0:-1,:].set(self.key_cache[:, 1:, :])
                self.value_cache.at[:,0:-1,:].set(self.value_cache[:, 1:, :])
        
        weights = jnp.einsum('b t c, b T c -> b t T', q, k) / math.sqrt(self.dk)
        size = weights.shape[-1]
        mask = jnp.tril(jnp.ones((size, size)))
        logits = jnp.where(mask == 0, -9e15, weights)
        attention = nn.softmax(logits, axis=-1)
        values = jnp.einsum('b t T, b T c -> b t c', attention, v)

        return values

class MultiHeadAttention(nn.Module):
    n_heads :int 
    model_dim : int
    max_token_len : int

    def setup(self):
        self.dk = self.model_dim / self.n_heads
        self.SA_layers = [ScaledDotProduct(dk=self.dk, max_token_len=self.max_token_len) for i in range(self.n_heads)]
        self.WO = nn.Dense(features=self.model_dim)

    def __call__(self, x, train:bool = True):
        if train == True:
            scores = [layer(x, train=True) for layer in self.SA_layers] 
        else:
           scores = [layer(x, train=False) for layer in self.SA_layers]  
        mha = jnp.concatenate(scores, axis=-1)
        res = self.WO(mha)
        return res
    
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
        x = nn.dropout(x)
        x = self.linear2(x)
        return x

class SoftmaxGate(nn.Module):
    model_dimension : int
    n_experts : int
    
    def setup(self):
        self.Wg = nn.Dense(features=self.n_experts)
    
    def __call__(self, x):
        g = self.Wg(x)
        gate = nn.softmax(g, axis=-1)
        return gate

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
        # z = jnp.ones(x.shape) * -jnp.inf
        # z = z.at[i].set(y)
        return y, i
    
    def __call__(self, x):
        #B,T,N
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
        self.gate = NoisyKGate(model_dim=self.model_dimension, num_experts=self.n_experts, k=self.k)

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
        x = nn.dropout(x)
        x = self.linear2(x)
        return x

class Block(nn.Module):
    model_dimension : int
    n_heads : int
    dropount : float
    max_token_len : int
    
    def setup(self):
        self.attention = MultiHeadAttention(model_dim=self.model_dimension, n_heads=self.n_heads, max_token_len=self.max_token_len)
        self.norm1 = LayerNorm(model_dimension=self.model_dimension)
        self.norm2 = LayerNorm(model_dimension=self.model_dimension)
        self.feedForward = FeedForward(model_dimension=self.model_dimension, ff_dim=4*self.model_dimension, dropout=self.dropout)
    
    def __call__(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.feedForward(x))
        return x

class Decoder(nn.Module):
    model_dimension : int
    n_heads : int
    seq_len : int
    vocab_size : int
    dropout : float
    blocks : int
    max_token_len : int 

    def setup(self):
        self.embeddingTable = Embeddings(model_dimension=self.model_dimension, vocab_size=self.vocab_size)
        self.Blocks = [Block(model_dimension=self.model_dimension, n_heads=self.n_heads, dropount=self.dropout, max_token_len=self.max_token_len) for i in range(self.blocks)]
        self.encodings = PositionalEncoding(model_dimension=self.model_dimension, seq_len=self.seq_len)
        self.linear = nn.Dense(features=self.vocab_size)

    def __call__(self, x):
        #B,T
        x = self.embeddingTable(x) #B,T,C
        x = self.encodings(x) 
        x = [Block(x) for Block in self.blocks]
        x = self.linear(x)
        result = nn.softmax(x, axis=-1)
        return result