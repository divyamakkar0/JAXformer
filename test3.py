# import os
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

import jax
import jax.numpy as jnp
from jax import NamedSharding
from jax.sharding import Mesh
from functools import partial
from jax.sharding import PartitionSpec as P
import numpy as np


# import sys
# proc_id = int(sys.argv[1])
# num_procs = int(sys.argv[2])
# jax.distributed.initialize('localhost:10000', num_procs, proc_id)

jax.distributed.initialize()

devices = np.array(jax.devices())
a1, a2 = devices.shape[0] // 2, 2
mesh = jax.make_mesh((a1, a2), ("dp", "pp"))


def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
    b = jax.random.normal(k2, (n_out,))
    return W, b


def init(key, layer_sizes, batch_size):
    key, *keys = jax.random.split(key, len(layer_sizes))
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

    key, *keys = jax.random.split(key, 3)
    inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
    targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))

    return params, (inputs, targets)


layer_sizes = [784, 128, 128, 128, 128, 128, 8]
batch_size = 128 * a1

params, batch = init(jax.random.key(0), layer_sizes, batch_size)


L = len(params) - 2  # num layers, excluding first and last
N = batch_size // a1  # batch size
F = params[0][0].shape[1]  # num features

# choose some pipeline parameters
S = a2  # number of stages
B = 16  # size of each microbatch
assert L % S == 0, "S (number of stages) must divide L (number of inner layers)"

# compute some useful quantities
M, ragged = divmod(N, B)  # M is number of microbatches
assert not ragged, "B (size of each microbatch) must divide total batch size"
K, ragged = divmod(M, S)  # K is microbatches per stage
assert not ragged, "S (number of stages) must divide number of microbatches"
print(f"{S} stages, {L // S} layer(s) per stage, {L} pipelined layers total")
print(f"{B} examples per microbatch, {M} microbatches total")


def predict_pp(params, inputs):
    (W_first, b_first), inner_params, (W_last, b_last) = params
    inputs = jax.nn.tanh(jnp.dot(inputs, W_first) + b_first)

    layer_fn = lambda Wb, x: jax.nn.tanh(x @ Wb[0] + Wb[1])

    def fwd_fn(Wb, x):
        def grad_fn(stop_grad):
            return (
                lambda *args: jax.lax.stop_gradient(layer_fn(*args))
                if stop_grad
                else layer_fn(*args)
            )

        return jax.lax.cond(
            jnp.any(jnp.isnan(x)),
            grad_fn(stop_grad=True),
            grad_fn(stop_grad=False),
            Wb,
            x,
        )

    inputs = spmd_pipeline(fwd_fn, inner_params, inputs)
    outputs = jnp.dot(inputs, W_last) + b_last
    return outputs


@partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=((P(), P("pp"), P()), P("pp", "dp")),
    out_specs=P(),
    check_vma=False,
)
def loss_pp(params, batch):
    inputs, targets = batch
    predictions = predict_pp(params, inputs.reshape(K, B, -1)).reshape(K * B, -1)
    targets = targets.reshape(K * B, -1)
    local_loss = jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))
    return jax.lax.pmean(jax.lax.pmean(local_loss, "pp"), "dp")


def spmd_pipeline(fn, stage_params, inputs):
    stage = jax.lax.axis_index("pp")
    n_devices = jax.lax.axis_size("pp")
    layers_per_device = stage_params[0].shape[0]
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
    perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    for i in range(microbatches + layers - 1):
        batch_idx = i % microbatch_per_device
        layer_idx = (i - layers + 1) % microbatch_per_device
        state = state.at[0].set(jnp.where(stage == 0, inputs[batch_idx], state[0]))
        state = jax.vmap(fn)(stage_params, state)
        outputs = outputs.at[layer_idx].set(
            jnp.where(stage == n_devices - 1, state[-1], outputs[layer_idx])
        )

        state = (
            jnp.roll(state, 1, axis=0)
            .at[0]
            .set(jax.lax.ppermute(state[-1], "pp", perm))
        )

        if batch_idx == microbatch_per_device - 1:
            inputs = jax.lax.ppermute(inputs, axis_name="pp", perm=perm)

        if layer_idx == microbatch_per_device - 1:
            outputs = jax.lax.ppermute(outputs, axis_name="pp", perm=perm)

    outputs = jax.lax.ppermute(outputs, "pp", perm)
    return outputs


# def spmd_pipeline(fn, stage_params, inputs):
#   stage = jax.lax.axis_index('pp')
#   outputs = jax.array_ref(jnp.zeros_like(inputs))
#   state = jax.array_ref(jnp.zeros((L // S, B, F)))
#   for i in range(M+L-1):
#     state[0] = jnp.where(stage == 0, inputs[i % K], state[0])
#     state[...] = jax.vmap(fn)(stage_params, state[...])
#     outputs[(i-L+1)%K] = jnp.where(stage == S-1, state[-1], outputs[(i-L+1) % K])
#     # outputs = outputs.at[(i-L+1) % K].set(jnp.where(stage == S-1, state[-1], outputs[(i-L+1) % K]))
#     state[...], inputs, outputs[...] = shift(i, state[...], inputs, outputs[...])
#   outputs[...] = jax.lax.ppermute(outputs[...], 'pp', [(i, (i+1) % S) for i in range(S)])
#   return outputs[...]

# def spmd_pipeline(fn, stage_params, inputs):
#   stage = jax.lax.axis_index('pp')
#   outputs = jnp.zeros_like(inputs) * jnp.nan
#   state = jnp.zeros((L // S, B, F)) * jnp.nan
#   for i in range(M+L-1):
#     state = jnp.where(stage == 0, jnp.concat([inputs[i % K][None, ...], state[1:]], axis=0), state)
#     state = jax.vmap(fn)(stage_params, state)
#     outputs = jnp.where(stage == S-1, jnp.concat([
#       outputs[:(i-L+1)%K],
#       state[-1][None, ...],
#       outputs[(i-L+1)%K + 1:]
#     ], axis=0), outputs)
#     state, inputs, outputs = shift(i, state, inputs, outputs)
#   outputs = jax.lax.ppermute(outputs, 'pp', [(i, (i+1) % S) for i in range(S)])
#   return outputs

# def shift(i, state, inputs, outputs):
#   sh = lambda x, d: jax.lax.ppermute(x, 'pp', [(i, (i+d) % S) for i in range(S)])
#   state = jnp.roll(state, +1, axis=0).at[0].set(sh(state[-1], +1))
#   if (i % K) == (-1 % K):
#     inputs = sh(inputs, +1)
#   if ((i-L+1) % K) == (-1 % K):
#     outputs = sh(outputs, +1)
#   return state, inputs, outputs


first_params, *inner_params, last_params = params
Ws, bs = zip(*inner_params)
params_stacked = jnp.stack(Ws), jnp.stack(bs)
first_params = jax.device_put(first_params, NamedSharding(mesh, P()))
params_stacked = jax.device_put(params_stacked, NamedSharding(mesh, P("pp")))
last_params = jax.device_put(last_params, NamedSharding(mesh, P()))
params_ = first_params, params_stacked, last_params

batch_ = (batch[0], batch[1])
x, y = batch_
x = x.reshape(M, B * a1, -1)
y = y.reshape(M, B * a1, -1)
batch_ = (x, y)
batch_ = jax.device_put(batch_, NamedSharding(mesh, P("pp", "dp")))

print(jax.jit(loss_pp)(params_, batch_))
grad_a = jax.jit(jax.grad(loss_pp))(params_, batch_)


def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jax.nn.tanh(outputs)
    return outputs


def loss(params, batch):
    inputs, targets = batch
    predictions = predict(params, inputs)
    return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))


print(jax.jit(loss)(params, batch))
grad_b = jax.jit(jax.grad(loss))(params, batch)
grad_b_layers = grad_b[1:-1]

grad_b_layers = jax.tree.map(lambda *x: jnp.stack((*x,), axis=0), *grad_b_layers)

print(jax.tree.map(lambda x, y: jnp.max(jnp.abs(x - y)), grad_a[1], grad_b_layers))


# p_spec = P('dp', 'pp')
# @partial(
#     jax.shard_map,
#     mesh=mesh,
#     in_specs=p_spec,
#     out_specs=p_spec,
# )
# def test_fn(x):
#     a = jax.lax.axis_index("pp")
#     x = jax.lax.select(a == 0, jnp.ones_like(x), jnp.zeros_like(x))
#     return x

# x = jnp.ones((a1, a2, 1), dtype=jnp.float32)
# out = test_fn(x)
# out = jax.experimental.multihost_utils.process_allgather(out)
# print(out)
