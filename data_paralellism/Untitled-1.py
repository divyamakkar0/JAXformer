"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the “Software”), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.sharding as shd
import jaxtyping as jt
import optax
from absl.testing import parameterized
from flax.training.train_state import TrainState

from . import fsdp


def _mock_data_mesh(n_devices: int) -> shd.Mesh:
    """Mocking a mesh with jax.device_count() > 1 using the cpu."""
    chex.set_n_cpu_devices(n_devices)
    jax.config.update("jax_platform_name", "cpu")
    return jax.make_mesh((jax.device_count(),), ("data",))


class FSDPTest(chex.TestCase):
    @parameterized.parameters(
        dict(shape=(8, 129, 13, 64), n_devices=8, spec=(None, None, None, "data")),
        dict(shape=(13, 17), n_devices=8, spec=(None, None)),
        dict(shape=(16,), n_devices=8, spec=()),
        dict(shape=(), n_devices=8, spec=()),
    )
    def test_fsdp_partition_spec(
        self, shape: tuple, n_devices: int, spec: tuple
    ) -> None:
        assert fsdp._fsdp_partition_spec(shape, n_devices) == shd.PartitionSpec(*spec)

    def test_shard_pytree(self) -> None:
        mesh = _mock_data_mesh(8)
        params = dict(
            layer1=dict(w=jnp.ones((1, 2, 8)), b=jnp.ones((8,))),
            layer2=dict(w=jnp.ones((1, 8)), s=jnp.ones(())),
        )
        shardings = fsdp.infer_fsdp_sharding(params, mesh)
        params = fsdp.shard_pytree(params, shardings)

        self.assertEqual(
            params["layer1"]["w"].sharding.spec, shd.PartitionSpec(None, None, "data")
        )

    def test_shard_state(self) -> None:
        mesh = _mock_data_mesh(8)
        module = nn.Sequential((nn.Dense(32), nn.LayerNorm(), nn.Dense(8)))

        def init_train_state(rng: jt.PRNGKeyArray, inputs: jax.Array) -> TrainState:
            params = module.init(rng, inputs)
            return TrainState.create(
                apply_fn=module.apply,
                params=params["params"],
                tx=optax.adam(1e-3),
            )

        rng = jax.random.key(0)
        inputs = jax.ShapeDtypeStruct(shape=(16, 7), dtype=jnp.bfloat16)
        state = jax.eval_shape(init_train_state, rng, inputs)
        shardings = fsdp.infer_fsdp_sharding(state, mesh)
        state = jax.jit(init_train_state, out_shardings=shardings)(rng, inputs)

        self.assertEqual(
            state.params["layers_0"]["kernel"].sharding.spec,
            shd.PartitionSpec(None, "data"),
        )
        self.assertEqual(
            state.params["layers_1"]["bias"].sharding.spec, shd.PartitionSpec()
        )
        self.assertEqual(
            state.params["layers_1"]["scale"].sharding.spec, shd.PartitionSpec()
        )
