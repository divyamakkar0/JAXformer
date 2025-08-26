from functools import partial
import os

from main import step

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "
)

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "~/ax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

import optax
from jax.sharding import PartitionSpec as P
import numpy as np
from test2 import Transformer
from dataset import Dataset
from utils import parse_args, config
import time
from typing import Tuple
import json
import wandb
from dataclasses import asdict
import orbax.checkpoint as ocp
from einops import rearrange


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

    mesh = jax.make_mesh((*axes,), (*axes_name,))
    return mesh


def main(cfg: config):
    key = jax.random.PRNGKey(0)
    DATA_PARALLEL = cfg.device_config.n_device_axis

    axes = (*cfg.device_config.n_device_axis,)
    axes_name = ("dp", )

    mesh = init_devices(axes, axes_name)
    log(mesh)

    checkpoint_dir = os.path.join(
        os.path.abspath(cfg.output_dir), cfg.name, "checkpoints"
    )
    load = os.path.exists(checkpoint_dir)
    if not load:
        os.makedirs(checkpoint_dir)
        checkpoint_dir = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)

    data_spec = P(None, "dp")
    data_partition = jax.sharding.NamedSharding(mesh, data_spec)

    train_dataset, val_dataset = Dataset.getDataset(
        cfg.data_config,
        partition=data_partition,
        dp=DATA_PARALLEL,
    )

    model = Transformer.get_model(cfg.model_config)

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
        lambda x: x if np.ndim(x) != 0 else jax.device_put(x, default_sharding),
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
        if use_wandb:
            wandb.init(
                entity="waterloo2",
                project="jaxformer",
                name=cfg.name,
                resume="allow",
                config=asdict(cfg),
            )
            wandb_id = wandb.run.id
        # save_checkpoint(init_step)

    if use_wandb:
        table = wandb.Table(
            columns=["step"]
            + [
                f"tokens_{i}"
                for i in range(
                    cfg.inference_batch
                    * cfg.model.blocks
                    * (jax.device_count() // cfg.model.blocks)
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
            logits, _ = model.pipe_step(
                params,
                x,
                key=key,
                train=train,
            )
            logits = logits.astype(jnp.float32)
            log_probs = jax.nn.log_softmax(logits, axis=-1)

            M, B, T, V = logits.shape
            y = y.reshape(-1)
            log_probs = log_probs.reshape(M * B * T, V)

            loss_idx = lambda x, idx: jax.lax.dynamic_slice(x, (idx,), (1,))
            loss = -(jax.vmap(loss_idx, in_axes=(0, 0))(log_probs, y)).mean()
            return loss

        loss = loss_fn(params, x, y, key)

        loss = jax.lax.pmean(loss, axis_name="pp")
        loss = jax.lax.pmean(loss, axis_name="tp")
        loss = jax.lax.pmean(loss, axis_name="dp")

        return loss

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
        step_fn = jax.value_and_grad(step)

        def single_step(batch):
            x, y, key = batch
            loss, grads = step_fn(params, x, y, key, train=True)
            return grads, loss


        # grads = jax.tree.map(lambda x: jnp.zeros_like(x), params)
        # loss = 0.0
        key = key.reshape(
            2,
        )
        batch = (x[0], y[0], key)
        grads, loss = single_step(batch)

        # for i in range(cfg.grad_step):
        #     key, subkey = jax.random.split(key)
        #     batch = (x[i], y[i], subkey)
        #     grads_step, loss_step = single_step(batch)
        #     grads = jax.tree.map(lambda a, b: a + b, grads, grads_step)
        #     loss += loss_step

        grads = jax.tree.map(lambda x: x / cfg.grad_step, grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        loss = loss.mean()
        return params, opt_state, loss

    @jax.jit
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(param_spec, data_spec, data_spec, key_spec),
        out_specs=P(),
        check_vma=False,
    )
    def eval_step(params, x, y, key):
        def single_step(loss, x, y, key):
            loss += step(params, x, y, key, train=False)
            return loss, None

        loss = jax.lax.scan(single_step, 0, (x, y, key))
        loss = loss / config.eval_steps
        return loss

    total_steps = cfg.training_steps
    total_tokens = train_dataset.tokens_per_step

    jax.experimental.multihost_utils.sync_global_devices("sync")
    log(f"Total steps: {total_steps}")
    log(f"Total tokens per step: {total_tokens:,}")

    key, sample_key = jax.random.split(key, 2)
    start = time.time()
    train_loss = []

    @jax.jit
    def make_sharded_key(key):
        key = jax.random.split(key, DATA_PARALLEL * LAYER_PARALLEL * TENSOR_PARALLEL)
        key = jnp.asarray(key).reshape(
            (DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL, 2)
        )
        return key


    for current_step in range(init_step, total_steps):
        key, train_key, eval_key = jax.random.split(key, 3)
        train_key = make_sharded_key(train_key)
        eval_key = make_sharded_key(eval_key)

        x, y = train_dataset(step=cfg.grad_step)

        params, opt_state, loss = train_step(params, opt_state, x, y, train_key)
        loss.block_until_ready()
        jax.experimental.multihost_utils.sync_global_devices("sync")
        end_time = time.time()
        tks_per_second = total_tokens / (end_time - start)

        log(
            f"step: {current_step + 1} \t loss: {loss.item()} \t tks/s: {tks_per_second:.2f}s"
        )
        start = time.time()
        continue

        if use_wandb:
            wandb_log = {
                "step": current_step,
                "loss/train_loss": metrics["loss"],
                "loss/train_cross_entropy_loss": metrics["loss_cross"],
                "lr": opt_state[1].hyperparams["learning_rate"],
            }
            if cfg.model.moe:
                wandb_log["loss/load_loss"] = metrics["loss_load"]
                for h in range(cfg.model.n_experts):
                    wandb_log[f"load/head_{h}"] = metrics[f"load/head_{h}"]

        if current_step % cfg.checkpoint_steps == 0:
            time_per_batch = time.time() - start
            eval_x, eval_y = val_dataset()
            eval_loss = eval_step(params, eval_x, eval_y, eval_key)

            eval_loss = eval_loss.item()
            train_loss = np.mean(jax.device_get(jnp.array(train_loss))).item()

            if use_wandb:
                wandb_log["loss/val_loss"] = metrics_val["loss"]
                wandb_log["loss/val_cross_entropy_loss"] = metrics_val["loss_cross"]
                if cfg.model.moe:
                    wandb_log["loss/val_load_loss"] = metrics_val["loss_load"]
                    for h in range(cfg.model.n_experts):
                        wandb_log[f"load/head_{h}"] = metrics_val[f"load/head_{h}"]

            jax.experimental.multihost_utils.sync_global_devices("sync")

            tokens_per_second = cfg.checkpoint_steps * total_tokens / time_per_batch
            log_string = f"Step {current_step + 1}, Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, tk/s: {tokens_per_second:,.2f}"
            log(log_string)
            # save_checkpoint(current_step)

            start = time.time()
            train_loss = []

        if use_wandb:
            wandb.log(data=wandb_log, step=current_step)

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

    if use_wandb:
        table = wandb.Table(
            columns=["step"]
            + [
                f"tokens_{i}"
                for i in range(
                    config.inference_batch
                    * config.model.blocks
                    * (jax.device_count() // config.model.blocks)
                )
            ],
        )
        wandb.Table.MAX_ROWS = total_steps // config.checkpoint_steps
        with open(
            os.path.join(os.path.abspath(config.output_dir), config.name, "tokens.txt"),
            "r",
        ) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            tokens = line.split("|")[1]
            tokens = ast.literal_eval(tokens)
            table.add_data(i, *tokens)

        wandb.log({"inference_tokens": table})
        wandb.finish()


if __name__ == "__main__":
    jax.distributed.initialize()
    cfg = parse_args()
    print(json.dumps(cfg.__dict__, indent=4, default=lambda o: o.__dict__))
    main(cfg)
