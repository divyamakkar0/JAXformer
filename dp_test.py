import os

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
)

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

# jax.distributed.initialize()

devices = np.array(jax.devices())
mesh = jax.make_mesh((devices.shape[0],), ("dp",))


W = jax.random.normal(jax.random.key(0), (2, 2))
W = jax.device_put(W, NamedSharding(mesh, P()))
x = jax.random.normal(jax.random.key(1), (8, 2))
x = jax.device_put(x, NamedSharding(mesh, P("dp")))


@partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(P(), P("dp")),
    out_specs=(P(), P()),
)
def matmul(W, x):
    def loss_fn(W, x):
        loss = (x @ W).sum()
        # loss = jax.lax.psum(loss, "dp")

        return loss

    W = jax.lax.pvary(W, axis_name="dp")
    grad_fn = jax.value_and_grad(loss_fn)

    loss, grad = grad_fn(W, x)
    print(loss, grad)
    # loss = jax.lax.psum(loss, "dp")

    return loss, grad


loss, grad = matmul(W, x)
naive_loss, naive_grad = jax.value_and_grad(lambda W, x: (x @ W).sum())(W, x)


# compare the losses difference
print("loss diff:", jnp.abs(loss - naive_loss))
