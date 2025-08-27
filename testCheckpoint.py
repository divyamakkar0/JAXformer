import jax
import jax.numpy as jnp

import orbax
import orbax.checkpoint as ocp
import os

jax.distributed.initialize()
OUTPUT_DIR = "./testSave"
# NAME = "testRun"
# checkpoint_dir = os.path.join(
#         os.path.abspath(OUTPUT_DIR), NAME, "checkpoints"
#   )
checkpoint_dir = "gs://jaxformer-test-bucket2/checkpoints/testRun"
# if not load:
#     os.makedirs(checkpoint_dir)
checkpoint_dir = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

options = ocp.CheckpointManagerOptions(max_to_keep=1)
checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)

test_pytree = {
        "a": jnp.array([1, 2, 3]),
        "b": {"c": jnp.array([4, 5, 6]),
        "d": jnp.array([7, 8, 9])}
    }

mesh = jax.make_mesh((jax.device_count(), ), ('dp',))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
test_pytree = jax.tree.map(lambda x: jax.device_put(x, sharding), test_pytree)

step = 0
print("Saving checkpoint...")
checkpoint_manager.save(step, args=ocp.args.StandardSave(test_pytree))
print("kicked off")
checkpoint_manager.wait_until_finished()
print("done")
