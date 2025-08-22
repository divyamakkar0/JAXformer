import os

from matplotlib.pyplot import step

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "
)

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

import optax
from jax.sharding import PartitionSpec as P
import numpy as np
from test2 import ModelConfig, shardedModel
from dataset import Dataset
from utils import parse_args, config
import time
from typing import Tuple

def log(msg: str):
    if jax.process_index() == 0:
        print(msg)

def init_devices(axes: Tuple[int,...], axes_name: Tuple[str,...]) -> jax.sharding.Mesh:
    jax.distributed.initialize()

    devices = np.array(jax.devices())
    for idx in np.ndindex(devices.shape):
        d = devices[idx]
        log(
            f"  {idx} ID: {d.id}, Process: {d.process_index}, "
            f"Coords: {d.coords}, Core: {d.core_on_chip}"
        )

    assert devices.size == np.prod(axes), (
        f"Expected {np.prod(axes)} devices, got {devices.shape[0]}"
    )

    mesh = jax.make_mesh((*axes, ), (*axes_name, ))
    return mesh

def main(cfg: config):

    DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL = cfg.device_config.n_device_axis

    axes = (*cfg.device_config.n_device_axis,)
    axes_name = ("dp", "pp", "tp")

    mesh = init_devices(
        axes, axes_name
    )

    data_partition = jax.sharding.NamedSharding(
        mesh,
        P(None, "pp", "dp", "tp"),
    )

    train_dataset, val_dataset = Dataset.getDataset(
        cfg.data_config,
        partition=data_partition,
        dp=DATA_PARALLEL,
    )

    model = shardedModel(cfg.model_config)

    log("creating sharded model ...")
    params = model.init_weights(jax.random.PRNGKey(0), mesh)

    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.lr.min_lr,
        peak_value=config.lr.max_lr,
        warmup_steps=config.lr.warmup_steps,
        decay_steps=config.lr.end_steps,
        end_value=config.lr.end_lr,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.inject_hyperparams(optax.adamw)(lr_scheduler),
    )


    default_sharding = jax.sharding.NamedSharding(mesh, P())
    opt_state = jax.tree.map(
        lambda x: x if np.ndim(x) != 0 else jax.device_put(x, default_sharding),
        tx.init(params),
    )


    param_count = jax.tree.reduce(
        lambda x, y: x + y.size,
        params,
        0,
    )
    log(f"Total parameters: {param_count:,}")


    def step(params, x, y, key, train=True):
        def loss_fn(params, x, y, key):
            logits, _ = model.pipe_step(
                params,
                x,
                key=key,
                train=train,
            )
            log_probs = jax.nn.log_softmax(logits, axis=-1)

            M, B, T, V = logits.shape
            y = y.reshape(-1)
            log_probs = log_probs.reshape(M * B * T, V)

            loss_idx = lambda x, idx: jax.lax.dynamic_slice(x, (idx,), (1,))
            loss = -(jax.vmap(loss_idx, in_axes=(0, 0))(log_probs, y)).mean()
            loss = jax.lax.pmean(loss, axis_name="pp")
            loss = jax.lax.pmean(loss, axis_name="tp")
            loss = jax.lax.pmean(loss, axis_name="dp")

            return loss

        if train:
            loss_fn = jax.value_and_grad(loss_fn)

        key = key.reshape(2,)
        val = loss_fn(params, x, y, key)
        loss, grads = val if train else (val, None)

        return loss, grads

    def update_params(params, opt_state, x,y, key):
        loss, grads = step(params, x, y, key, train=True)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    param_spec = shardedModel.get_p_spec([model.embedding, model.block], mesh, cfg.model_config)
    opt_spec = jax.tree.map(
        lambda x: x.sharding.spec,
        opt_state
    )
    data_spec = P("pp", "dp", "tp")
    key_spec = P("dp", "pp", "tp")

    train_step = jax.jit(
        jax.shard_map(
            lambda params, opt_state, x, y, key: update_params(params, opt_state, x, y, key),
            mesh=mesh,
            in_specs=(param_spec, opt_spec, data_spec, data_spec, key_spec),
            out_specs=(param_spec, opt_spec, P()),
            check_vma=False
        )
    )

    eval_step = jax.jit(
        jax.shard_map(
            lambda params, x, y, key: step(params, x, y, key=key, train=False)[0],
            mesh=mesh,
            in_specs=(param_spec, data_spec, data_spec, key_spec),
            out_specs=P(),
            check_vma=False
        ),
    )

    total_steps = cfg.training_steps
    total_tokens = train_dataset.tokens_per_step

    jax.experimental.multihost_utils.sync_global_devices("sync")
    log(f"Total parameters: {param_count:,}")
    log(f"Total steps: {total_steps}")
    log(f"Total tokens per step: {total_tokens:,}")

    key = jax.random.PRNGKey(0)
    key, sample_key = jax.random.split(key, 2)
    init_step = 0
    start = time.time()

    for i in range(init_step, total_steps):
        key, train_key, eval_key = jax.random.split(key, 3)
        train_key = jax.random.split(train_key, DATA_PARALLEL * LAYER_PARALLEL * TENSOR_PARALLEL)
        train_key = jnp.asarray(train_key).reshape((DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL, 2))
        eval_key = jax.random.split(eval_key, DATA_PARALLEL * LAYER_PARALLEL * TENSOR_PARALLEL)
        eval_key = jnp.asarray(eval_key).reshape((DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL, 2))

        x, y = train_dataset()
        params, opt_state, loss = train_step(params, opt_state, x, y, train_key)

        eval_x, eval_y = val_dataset()
        eval_loss = eval_step(params, eval_x, eval_y, eval_key)

        loss, eval_loss = loss.item(), eval_loss.item()
        jax.experimental.multihost_utils.sync_global_devices("sync")
        time_per_batch = time.time() - start
        tokens_per_second = 2 * total_tokens / time_per_batch
        log_string = f"Step {i + 1}, Loss: {loss:.4f}, Eval Loss: {eval_loss:.4f}, tk/s: {tokens_per_second:,.2f}"
        log(log_string)
        start = time.time()

    outputs = model.generate(
        params,
        cfg.model_config,
        key=sample_key,
        x="hello world",
        B=1,
        k=10000,
        temperature=1.0,
        n_devices=1,
        use_cache=True,
    )

    log("Generated outputs:")
    for output in outputs:
        log(f"\t{output}")


if __name__ == "__main__":
    cfg = parse_args()