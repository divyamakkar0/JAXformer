import jax
import jax.numpy as jnp



mesh = jax.make_mesh((1,), ('dp',))
x = jnp.arange(10)
x = jax.device_put(x, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('dp')))


print(x.sharding.spec)
