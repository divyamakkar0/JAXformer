import jax
import jax.numpy as jnp
from functools import partial
from jax.sharding import PartitionSpec as P
import numpy as np

print("initing ... ")
jax.distributed.initialize()
print("init done")

devices = np.array(jax.devices())
if jax.process_index() == 0:
    for idx in np.ndindex(devices.shape):
        d = devices[idx]
        print(
            f"  {idx} ID: {d.id}, Process: {d.process_index}, "
            f"Coords: {d.coords}, Core: {d.core_on_chip}"
        )

mesh = jax.make_mesh((32,), ('dp'))
print(mesh)
a = jnp.arange(32)
a = jax.device_put(a, jax.NamedSharding(mesh, P('dp')))

@partial(
    jax.shard_map,
    in_specs=(P('dp'),),
    out_specs=P(),
)
def all_gather(x):
    return jax.lax.all_gather(x, axis_name='dp', axis=0, tiled=True)

b = all_gather(a)

print(b)
jax.debug.visualize_array_sharding(b)
