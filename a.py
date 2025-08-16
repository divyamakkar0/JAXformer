import jax
import jax.numpy as jnp

a = jax.lax.iota(jnp.int32, 40)
print(a)
