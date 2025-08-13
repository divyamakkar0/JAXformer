import jax
import jax.numpy as jnp
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
mesh = jax.make_mesh(jax.device_count(), ('dp'))
a = jnp.arange(32)
a = jax.device_put(a, jax.NamedSharding(mesh, jax.sharding.PartitionSpec('dp')))

