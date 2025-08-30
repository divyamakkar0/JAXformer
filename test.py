import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "
)

import jax
import jax.numpy as jnp

#TODO: switch to gcs path
jax.config.update("jax_compilation_cache_dir", "gs://jaxformer-cache/")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

import optax
import numpy as np
import time
import json
import wandb
import orbax.checkpoint as ocp

from dataclasses import asdict
from jax.sharding import PartitionSpec as P
from functools import partial
from typing import Tuple
from test2 import shardedModel
from dataset import Dataset
from utils import parse_args, config


def log(msg: str):
    if jax.process_index() == 0:
        print(msg)

def init_devices(
    axes: Tuple[int, ...], axes_name: Tuple[str, ...]
) -> jax.sharding.Mesh:
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
    try:
        mesh = jax.make_mesh(axes, axes_name)
    except:
        log("Failed to create mesh with make_mesh, falling back to sharding.Mesh")
        mesh = jax.sharding.Mesh(devices.reshape(axes), axes_name)
    return mesh


def main(cfg: config):
    key = jax.random.PRNGKey(cfg.seed)
    DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL = cfg.device_config.n_device_axis

    axes = (*cfg.device_config.n_device_axis,)
    axes_name = ("dp", "pp", "tp")

    mesh = init_devices(axes, axes_name)
    log(mesh)

    checkpoint_dir = cfg.output_dir + cfg.name
    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)
    load = checkpoint_manager.latest_step() is not None

    data_spec = P(None, "pp", "dp", "tp")
    data_partition = jax.sharding.NamedSharding(mesh, data_spec)

    train_dataset, val_dataset = Dataset.getDataset(
        cfg.data_config,
        partition=data_partition,
        dp=DATA_PARALLEL,
        pp=LAYER_PARALLEL,
        tp=TENSOR_PARALLEL,
    )

    model = shardedModel(cfg.model_config)

    log("creating sharded model ...")
    key, init_key = jax.random.split(key, 2)
    params = model.init_weights(init_key, mesh)

    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=cfg.lr.min_lr,
        peak_value=cfg.lr.max_lr,
        warmup_steps=cfg.lr.warmup_steps,
        decay_steps=cfg.lr.end_steps,
        end_value=cfg.lr.end_lr,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.inject_hyperparams(optax.adamw)(learning_rate=lr_scheduler),
    )

    default_sharding = jax.sharding.NamedSharding(mesh, P())
    opt_state = jax.tree.map(
        lambda x: x if jnp.ndim(x) != 0 else jax.device_put(x, default_sharding),
        tx.init(params),
    )

    init_step = 0
    use_wandb = cfg.wandb is True and jax.process_index() == 0
    wandb_id = None

    def make_save_tree(step):
        model_state = {
            "params": params,
            "opt_state": opt_state,
        }
        save_tree = {
            "state": model_state,
            "key": jax.device_get(key),
            "train_step_idx": train_dataset.step_idx,
            "train_shard_idx": (train_dataset.shard_idx - 1) % len(train_dataset.data),
            "val_step_idx": val_dataset.step_idx,
            "val_shard_idx": (val_dataset.shard_idx - 1) % len(val_dataset.data),
            "step": step,
            "wandb_id": wandb_id,
        }
        return save_tree

    def save_checkpoint(
        step,
    ):
        save_tree = make_save_tree(step)
        checkpoint_manager.save(step, args=ocp.args.StandardSave(save_tree))

    if load:
        abstract_tree_map = jax.tree.map(
            ocp.utils.to_shape_dtype_struct, make_save_tree(init_step)
        )
        tree_state = checkpoint_manager.restore(
            checkpoint_manager.latest_step(),
            args=ocp.args.StandardRestore(abstract_tree_map),
        )

        init_step = tree_state["step"]
        log(f"loading checkpoint @ step {init_step}")

        key.key = tree_state["key"]
        params = tree_state["state"]["params"]
        opt_state = tree_state["state"]["opt_state"]

        train_dataset.step_idx = tree_state["train_step_idx"]
        train_dataset.shard_idx = tree_state["train_shard_idx"]
        train_dataset.load_next_shard()

        val_dataset.step_idx = tree_state["val_step_idx"]
        val_dataset.shard_idx = tree_state["val_shard_idx"]
        val_dataset.load_next_shard()

        wandb_id = tree_state["wandb_id"]
        if use_wandb:
            assert wandb_id is not None, "wandb_id is None"
            wandb.init(
                entity="waterloo2",
                project="jaxformer",
                name=config.name,
                resume="must",
                id=wandb_id,
                config=asdict(config),
            )

    else:
        log("no checkpoint found, saving init copy")
        save_checkpoint(init_step)
        if use_wandb:
            wandb.init(
                entity="waterloo2",
                project="jaxformer",
                name=cfg.name,
                resume="allow",
                config=asdict(cfg),
            )
            wandb_id = wandb.run.id

    if use_wandb:
        table = wandb.Table(
            columns=["step"]
            + [
                f"tokens_{i}"
                for i in range(
                    cfg.inference_config.batch_size
                    * cfg.inference_config.n_devices
                    * jax.process_count()
                )
            ],
            log_mode="INCREMENTAL",
        )

    param_count = jax.tree.reduce(
        lambda x, y: x + y.size,
        params,
        0,
    )
    log(f"Total parameters: {param_count:,}")

    def step(params, x, y, key, train):
        def loss_fn(params, x, y, key):
            logits, (_, moe_stat) = model.pipe_step(
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
            loss_cross = -(jax.vmap(loss_idx, in_axes=(0, 0))(log_probs, y)).mean()

            loss_cross = jax.lax.pmean(loss_cross, axis_name="dp")
            loss_cross = jax.lax.pmean(loss_cross, axis_name="tp")
            loss_cross = jax.lax.pmean(loss_cross, axis_name="pp")

            loss_balance = 0.0

            # TODO: change to moe stat
            # if load is not None:
            #     load = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="dp"), load)
            #     load = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="tp"), load)
            #     load = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="pp"), load)

            #     f, p = load["f"], load["p"]
            #     loss_balance = (cfg.model_config.n_experts / cfg.model_config.k) * (
            #         f * p
            #     ).sum()

            loss = loss_cross + cfg.alpha * loss_balance

            metrics = {
                "loss": loss,
                "loss_cross": loss_cross,
                "loss_balance": loss_balance,
                "load_expert": None,  # TODO: fix this
                "moe_stat": moe_stat
            }
            return loss, metrics

        return loss_fn(params, x, y, key)

    param_spec = shardedModel.get_p_spec(
        [model.embedding, model.block], mesh, cfg.model_config
    )
    opt_spec = jax.tree.map(lambda x: x.sharding.spec, opt_state)
    key_spec = P("dp", "pp", "tp")

    @jax.jit
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(param_spec, opt_spec, data_spec, data_spec, key_spec),
        out_specs=(param_spec, opt_spec, P()),
        check_vma=False,
    )
    def train_step(params, opt_state, x, y, key):
        step_fn = jax.value_and_grad(step, has_aux=True)

        def single_step(grads, batch):
            (_, metrics), grads_current = step_fn(params, *batch, train=True)
            grads = jax.tree.map(lambda x, y: x + y, grads, grads_current)
            return grads, metrics

        grads = jax.tree.map(lambda x: jnp.zeros_like(x), params)
        key = key.reshape(cfg.grad_step, 2)

        grads, metrics = jax.lax.scan(
            single_step,
            grads,
            (x, y, key),
        )

        grads = jax.tree.map(lambda x: x / cfg.grad_step, grads)
        metrics = jax.tree.map(lambda x: x.mean(), metrics)

        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, metrics

    @jax.jit
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(param_spec, data_spec, data_spec),
        out_specs=P(),
        check_vma=False,
    )
    def eval_step(params, x, y):
        def single_step(_, batch):
            loss, metrics = step(
                params, *batch, key=jax.random.PRNGKey(0), train=False
            )  # Key does not matter
            return loss, metrics

        _, metrics = jax.lax.scan(single_step, 0, (x, y))
        metrics = jax.tree.map(lambda x: x.mean(), metrics)
        return metrics

    total_steps = cfg.training_steps
    total_tokens = train_dataset.tokens_per_step

    jax.experimental.multihost_utils.sync_global_devices("sync")
    log(f"Total steps: {total_steps}")
    log(f"Total tokens per step: {total_tokens:,}")

    key, sample_key = jax.random.split(key, 2)
    start = time.time()
    train_loss = []

    @partial(jax.jit, static_argnames=["steps"])
    def make_sharded_key(key, steps=1):
        key = jax.random.split(
            key, DATA_PARALLEL * LAYER_PARALLEL * TENSOR_PARALLEL * steps
        )
        key = jnp.asarray(key).reshape(
            (DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL, steps, 2)
        )
        return key

    for current_step in range(init_step, total_steps):
        key, train_key = jax.random.split(key)
        train_key = make_sharded_key(train_key, steps=cfg.grad_step)

        x, y = train_dataset(step=cfg.grad_step)

        params, opt_state, metrics = train_step(params, opt_state, x, y, train_key)
        breakpoint()
        train_loss.append(metrics["loss"])

        if use_wandb:
            wandb_log = {
                "step": current_step,
                "loss/train_loss": metrics["loss"],
                "loss/train_cross_entropy_loss": metrics["loss_cross"],
                "lr": opt_state[1].hyperparams["learning_rate"],
            }
            if cfg.model_config.moe:
                wandb_log["loss/load_loss"] = metrics["loss_balance"]
                for h in range(cfg.model_config.n_experts):
                    wandb_log[f"load/head_{h}"] = metrics[f"load_expert"][h]

        if current_step % cfg.checkpoint_steps == 0:
            time_per_batch = time.time() - start
            eval_x, eval_y = val_dataset(step=cfg.eval_steps)
            val_metrics = eval_step(params, eval_x, eval_y)

            if use_wandb:
                wandb_log["loss/val_loss"] = val_metrics["loss"]
                wandb_log["loss/val_cross_entropy_loss"] = val_metrics["loss_cross"]
                if cfg.model_config.moe:
                    wandb_log["loss/val_load_loss"] = val_metrics["loss_balance"]
                    for h in range(cfg.model_config.n_experts):
                        wandb_log[f"load/head_{h}"] = val_metrics[f"load_expert"][h]

            jax.experimental.multihost_utils.sync_global_devices("sync")

            tokens_per_second = cfg.checkpoint_steps * total_tokens / time_per_batch
            train_loss = jnp.array(train_loss).mean().item()
            eval_loss = val_metrics["loss"].item()
            log_string = f"Step {current_step + 1}, Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, tk/s: {tokens_per_second:,.2f}"
            log(log_string)

            start = time.time()
            train_loss = []

        if current_step % (10 *cfg.checkpoint_steps) == 0:
            outputs = model.generate(
                params,
                cfg.model_config,
                key=sample_key,
                x=cfg.inference_config.prompt,
                B=cfg.inference_config.batch_size,
                k=cfg.inference_config.top_k,
                temperature=cfg.inference_config.temperature,
                n_devices=cfg.inference_config.n_devices,
                use_cache=cfg.inference_config.use_cache,
            )

            log("Generated outputs:")
            for output in outputs:
                log(f"\t{output}")

            if jax.process_index() == 0:
                save_path = os.path.join(os.path.abspath("./samples"), cfg.name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(
                    os.path.join(save_path, "tokens.txt"),
                    "a",
                ) as f:
                    f.write(f"{current_step} | {outputs}\n")

            if use_wandb:
                table.add_data(current_step, *outputs)
                wandb_log["inference_tokens"] = table

            save_checkpoint(current_step)
            gen_end = time.time()
            print(f"Generation time: {gen_end:.4f} seconds")

            start = time.time()

        if use_wandb:
            wandb.log(data=wandb_log, step=current_step)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    jax.distributed.initialize()
    cfg = parse_args()
    print(json.dumps(cfg.__dict__, indent=4, default=lambda o: o.__dict__))
    main(cfg)
