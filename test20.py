import jax
import jax.numpy as jnp
import timeit


# Define a function to sample a large matrix
def sample_data(key):
    return jax.random.normal(key, shape=(20000, 20000))


# JIT-compile the function
jitted_sample_data = jax.jit(sample_data)

# Create an initial key
key = jax.random.PRNGKey(0)

# --- Performance Comparison ---

# 1. Warm up the JIT-compiled function to trigger compilation
print("Running JITted function for the first time to compile it...")
jitted_sample_data(key).block_until_ready()
print("Compilation complete.\n")

# 2. Time the JITted function (now using the cached version)
# We use a lambda function to pass the code to timeit
jit_time = timeit.timeit(
    lambda: jitted_sample_data(key).block_until_ready(), number=100
)
print(f"Time for JITted function (averaged over 10 runs): {jit_time / 10:.6f} seconds")

# 3. Time the original, non-JITted function
vanilla_time = timeit.timeit(lambda: sample_data(key).block_until_ready(), number=100)
print(
    f"Time for non-JITted function (averaged over 10 runs): {vanilla_time / 10:.6f} seconds"
)
