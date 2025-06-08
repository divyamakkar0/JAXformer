# %%
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

# %%
from tensor_parallelism.utils import simulate_CPU_devices

simulate_CPU_devices()

# %%
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
from single_gpu import TrainState
import time

PyTree = Any
Metrics = Dict[str, Tuple[jax.Array, ...]]

# %%
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

# %%
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

# %%
class KeyState:
    def __init__(self, base_key: jax.random.key):
        self.key = jax.random.key(base_key)

    def __call__(self, num: int = 2):
        self.key, rng = jax.random.split(self.key, num=num)
        return rng

# %%
model = DPClassifier(config=config.model)
optimizer = optax.adamw(
    learning_rate=config.optimizer.learning_rate,
)
class TrainStateWithRNG(train_state.TrainState):
        rng: Any

# %%
key = KeyState(config.seed)
x=jax.random.normal(key(), (config.data.batch_size, config.data.input_size))
y = jax.random.randint(key(), (config.data.batch_size,), 0, config.data.num_classes)
variables = model.init({"params": key()}, x, train=False)
params = variables.pop("params")
device_array = np.array(jax.devices())
mesh = Mesh(device_array, ("x",))
print(jax.tree.reduce(lambda acc, current: acc + current.size, jax.tree.leaves(params), 0))


# %%
def init_device(params, rng, local_model, config):
        tx = optax.chain(
            optax.clip_by_global_norm(1),
            optax.inject_hyperparams(optax.adam)(learning_rate=1e-3),
        )
        state = TrainStateWithRNG.create(
            apply_fn=local_model.apply,
            params=params,
            tx=tx,
            rng=rng,
        )
        return state

# %%
sharded_init = shard_map(
            functools.partial(init_device, rng=key(), local_model=model, config=model_config),
            mesh,
            in_specs=(P()),
            out_specs=(P()),
        )

state_initialized = sharded_init(params)

# %%
def fold_key(key, axis):
        axis_index = jax.lax.axis_index(axis)
        return jax.random.fold_in(key, axis_index)

# %%
def cross_entropy_loss(model, params, key, x, y, train=True):
        dropout_key = fold_key(key, "x")
        B, T = x.shape
        pred = model.apply({'params': params}, x, train=train, rngs={'dropout': dropout_key})
        log_prob = jax.nn.log_softmax(pred, axis=-1)
        loss = -jnp.mean(log_prob[jnp.arange(B), y])
        loss = jax.lax.pmean(loss, axis_name="x")  
        return loss
#loss = cross_entropy_loss(model, params, key(), x, y)

# %%
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

# %%
def accumulate_grads(key, x, y, state):
        print("starting training")
        loss_fn = jax.tree_util.Partial(cross_entropy_loss, model)
        train_step_jit = lambda key, params, x, y : train_step(loss_fn, params, key, x, y)
   
        start = time.time()
        train_loss = 0.0

        grads = None
        acc_metrics = None
        for i in range(2):
            grads_step, metrics = train_step_jit(key, state.params, x, y)
            grads = grads_step if grads is None else jax.tree.map(
                lambda x, y: x + y, grads, grads_step
            )
            acc_metrics = metrics if acc_metrics is None else jax.tree.map(jnp.add, acc_metrics, metrics)

        grads = jax.tree.map(lambda x: x / 2, grads)
        
        return grads, acc_metrics

# %%
def train_step_device(state, x, y):
        key, step_key = jax.random.split(state.rng)
        grads, step_metrics = accumulate_grads(step_key, x, y, state)
        grads = jax.tree.map(lambda g: jax.lax.pmean(g, axis_name="x"), grads)
        new_state = state.apply_gradients(grads=grads, rng=key)
        step_metrics = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="x"), step_metrics)

        return new_state, step_metrics


# %%
train_step_dp_fn =  shard_map(
            train_step_device,
            mesh,
            in_specs=(P(), P("x",), P("x",)),
            out_specs=(P(), P()),
        )

# %%
# state, metrics = train_step_dp_fn(state_initialized, x, y)
# state

# # %%
# print("DP Parameters")
# pprint(jax.tree.map(lambda x: (x.shape, x.sharding), state_initialized.params))

# %%
state = state_initialized
for _ in range(100):
    state_dp, metrics_dp = train_step_dp_fn(state, x, y)
    state = state_dp
    print(metrics_dp)
state_dp, final_metrics_dp = train_step_dp_fn(state_dp, x, y)
print(final_metrics_dp)

# %%
# print(state)

# %%
# p = state.params['input_dense']['kernel']
# jax.debug.visualize_array_sharding(x)

# %%
print("DP Parameters")
pprint(jax.tree.map(lambda x: (x.shape, x.sharding), state.params))
print("Metrics")
pprint(jax.tree.map(lambda x: (x.shape, x.sharding), metrics))