import os
import urllib.request
from urllib.error import HTTPError

# Github URL where python scripts are stored.
base_url = "https://raw.githubusercontent.com/phlippe/uvadlc_notebooks/master/docs/tutorial_notebooks/scaling/JAX/"
# Files to download.
python_files = ["single_gpu.py", "utils.py"]
# For each file, check whether it already exists. If not, try downloading it.
for file_name in python_files:
    if not os.path.isfile(file_name):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_name)
        except HTTPError as e:
            print(
                "Something went wrong. Please try to download the file directly from the GitHub repository, or contact the author with the full output including the following error:\n",
                e,
            )
from tensor_parallelism.utils import simulate_CPU_devices

simulate_CPU_devices()

import functools
from pprint import pprint
from typing import Any, Callable, Dict, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from ml_collections import ConfigDict
from flax.training import train_state
import time
from flax.training.train_state import TrainState

PyTree = Any
Metrics = Dict[str, Tuple[jax.Array, ...]]

class DPClassifier(nn.Module):
    config: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = nn.Dense(
            features=self.config.hidden_size,
            dtype=self.config.dtype,
            name="input_dense",
        )(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(
            features=self.config.num_classes,
            dtype=self.config.dtype,
            name="output_dense",
        )(x)
        x = x.astype(jnp.float32)
        return x
    
data_config = ConfigDict(
    dict(
        batch_size=16,
        num_classes=8,
        input_size=32,
    )
)
model_config = ConfigDict(
    dict(
        hidden_size=8,
        dropout_rate=0.1,
        dtype=jnp.bfloat16,
        num_classes=data_config.num_classes,
        data_axis_name="data",
    )
)
optimizer_config = ConfigDict(
    dict(
        learning_rate=1e-3,
        num_minibatches=4,
    )
)
config = ConfigDict(
    dict(
        model=model_config,
        optimizer=optimizer_config,
        data=data_config,
        data_axis_name=model_config.data_axis_name,
        seed=42,
    )
)

class KeyState:
    def __init__(self, base_key: jax.random.key):
        self.key = jax.random.key(base_key)

    def __call__(self, num: int = 2):
        self.key, rng = jax.random.split(self.key, num=num)
        return rng

model = DPClassifier(config=config.model)
optimizer = optax.adamw(
    learning_rate=1e-3,
)

key = KeyState(config.seed)

x=jax.random.normal(key(), (config.data.batch_size, config.data.input_size))
y = jax.random.randint(key(), (config.data.batch_size,), 0, config.data.num_classes)

params = model.init(key(), x, False)['params']

tx = optax.chain(
    optax.clip_by_global_norm(1),
    optax.inject_hyperparams(optax.adam)(learning_rate=1e-3),
)

state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)

def cross_entropy_loss(model, params, key, x, y, train=True):
        key, dropout_key = jax.random.split(key)
        B, T = x.shape
        pred = model.apply({'params': params}, x, train=train, rngs={'dropout': dropout_key})
        log_prob = jax.nn.log_softmax(pred, axis=-1)
        loss = -jnp.mean(log_prob[jnp.arange(B), y])
        return loss

def train_step(loss_fn, params, key, *args, **kwargs):
        loss_grad = jax.value_and_grad(
            loss_fn,
            argnums=0,
            has_aux=False
        )
        loss, grads = loss_grad(params, key, *args, **kwargs, train=True)
        # don't need cache in training

        metrics = {
            'loss': loss,
        }
        return grads, metrics

loss_fn = jax.tree_util.Partial(cross_entropy_loss, model)
train_step_jit = jax.jit(
        lambda key, params, x, y : train_step(loss_fn, params, key, x, y),
    )

train_loss = 0.0
for _ in range(100):
    grads = None
    grad_loss = 0.0
    grads_step, metrics = train_step_jit(key(), state.params, x, y)
    print(metrics)
    grads = grads_step if grads is None else jax.tree.map(
                lambda x, y: x + y, grads, grads_step
            )
    state = state.apply_gradients(grads=grads)
    train_loss += (grad_loss)

