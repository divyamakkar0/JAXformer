x = jnp.arange(24)
x = x.reshape(2, 3, 4)
x


y, i = jax.lax.top_k(x, 2)


def top(x):
    k = 2
    y, i = jax.lax.top_k(x, k)
    z = jnp.ones(x.shape) * -jnp.inf
    z = z.at[i].set(y)
    return z


x2 = jnp.apply_along_axis(func1d=top, axis=-1, arr=x)
x2

x1 = jnp.arange(4)
y1, ind = jax.lax.top_k(x, 2)
z = jnp.ones(x1.shape) * -jnp.inf
x = z.at[ind].set(y1)
x

z = jnp.ones(x.shape)
z = z * -jnp.inf
z = z.at[i].set(y)
