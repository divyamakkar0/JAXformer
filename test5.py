import jax
import jax.numpy as jnp


@jax.jit
def f():
    x_ref = jax.array_ref(
        jnp.zeros(3)
    )  # new array ref, with initial value [0., 0., 0.]
    x_ref[1] = 1.0  # indexed add-update
    return x_ref


x = f()
print(x)  # ArrayRef([0., 2., 0.])
