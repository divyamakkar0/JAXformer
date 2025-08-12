import os

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

DEBUG = os.getenv("DEBUG")
if DEBUG is not None and int(DEBUG) == 1:
    print(f" --------- DEBUGGING MODE ON -----------")
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_disable_jit", True)
else:
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_disable_jit", False)

import time
import ast
import json
import optax

from utils import parse_args, config
from model import shardedModel
from dataset import Dataset

import wandb
from dataclasses import asdict

import orbax.checkpoint as ocp

from jax.sharding import PartitionSpec as P
from typing import Optional
from jaxtyping import PyTree
from einops import rearrange
import numpy as np


class KeyState:
    def __init__(self, seed: int):
        self.key = jax.random.PRNGKey(seed)

    def __call__(self, num: int = 1):
        self.key, *rng = jax.random.split(self.key, num=num + 1)
        return rng[0] if num == 1 else jnp.array(rng)


class TrainState:
    def __init__(self, params, tx, opt_state: Optional[PyTree] = None):
        self.params = params
        self.tx = tx
        self.opt_state = opt_state
        if opt_state is None:
            self.opt_state = tx.init(params)

    def apply_gradients(self, grads):
        updates, self.opt_state = self.tx.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)

        return self

    @classmethod
    def restore(cls, restored: PyTree, tx, model_spec, mesh):
        params = jax.tree.map(
            lambda x, y: jax.device_put(x, jax.sharding.NamedSharding(mesh, y)),
            (*restored["params"],),
            model_spec,
        )

        opt_state_init = tx.init(params)

        opt_state = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(opt_state_init),
            jax.tree_util.tree_leaves(restored["opt_state"]),
        )

        default_shard = jax.sharding.NamedSharding(mesh, P())

        def shard_opt_leaf(init_leaf, loaded_leaf):
            dim = np.ndim(loaded_leaf)
            sharding = init_leaf.sharding if dim != 0 else default_shard
            return jax.device_put(loaded_leaf, sharding)

        opt_state = jax.tree.map(shard_opt_leaf, opt_state_init, opt_state)

        return cls(
            params=params,
            tx=tx,
            opt_state=opt_state,
        )

    @property
    def n_params(self):
        param_count = sum(x.size for x in jax.tree.leaves(self.params))
        return param_count

def log(msg: str):
    if jax.process_index() == 0:
        print(msg)


def setup_devices(cfg: config):
    device_cfg = cfg.device_config
    assert 3 == len(device_cfg.n_device_axis)

    jax.distributed.initialize()
    devices = np.array(jax.devices())[:, None, None]

    assert devices.shape[0] == np.prod(device_cfg.n_device_axis)
    devices = devices.reshape(*device_cfg.n_device_axis)


    mesh = jax.make_mesh((*device_cfg.n_device_axis,), axis_names=("fsdp", "model", "tensor"))

    count = devices.shape
    count = {
        "fsdp": count[0],
        "model": count[1],
        "tensor": count[2],
    }

    platform = jax.devices()[0].platform
    if platform == "tpu":
        log("Available TPU Devices:")
        log(f"Device array shape: {devices.shape}")
        for idx in np.ndindex(devices.shape):
            d = devices[idx]
            log(
                f"  {idx} ID: {d.id}, Process: {d.process_index}, "
                f"Coords: {d.coords}, Core: {d.core_on_chip}"
            )
    log(f"Mesh: {mesh}")

    return mesh, count


def loss(model, cfg: config, params: PyTree, key: jax.random.PRNGKey, x, y, train):
    M, B, T = x.shape
    pred, (_, load) = shardedModel.pipe_step(model, params, x, key=key, train=train)

    total_tokens = M * B * T
    log_prob = jax.nn.log_softmax(pred, axis=-1).reshape(total_tokens, -1)
    y = y.reshape(total_tokens)

    loss_idx = lambda x, idx: jax.lax.dynamic_slice(x, (idx,), (1,))
    loss_cross = -(jax.vmap(loss_idx, in_axes=(0, 0))(log_prob, y)).mean()

    loss_balance = 0.0
    if load is not None:
        load = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="fsdp"), load)
        load = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="model"), load)
        load = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="tensor"), load)

        f = load["f"]
        p = load["p"]

        loss_balance = cfg.model.n_experts / cfg.model.k * jnp.sum(f * p)

    loss_cross = jax.lax.pmean(loss_cross, axis_name="model")
    loss_cross = jax.lax.pmean(loss_cross, axis_name="fsdp")
    loss_cross = jax.lax.pmean(loss_cross, axis_name="tensor")

    loss = loss_cross + cfg.alpha * loss_balance
    aux_stat = (load if load is None else load["tokens_per_expert"], loss_cross, loss_balance)

    return loss, aux_stat


def step(loss_fn, grad_steps, params, key, x, y, train):
    if train:
        loss_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    def step_fn(grads, batch):
        x, y, key = batch
        val = loss_fn(params, key, x, y, train=train)

        if train:
            val, grads_new = val
            grads = jax.tree.map(
                lambda x, y: x + y,
                grads,
                grads_new,
            )

        loss, (load, loss_cross, loss_balance) = val

        metrics = {
            "loss": loss,
            "loss_cross": loss_cross,
        }

        if load is not None:
            metrics["loss_load"] = loss_balance

            for h in range(load.shape[0]):
                metrics[f"load/head_{h}"] = load[h]

        return grads, metrics

    x = rearrange(x, "1 m (g b) t -> g m b t", g=grad_steps)
    y = rearrange(y, "1 m (g b) t -> g m b t", g=grad_steps)
    key = rearrange(key, "1 1 1 g s -> g s", g=grad_steps, s=2)

    grads = None
    if train:
        embed_grads, layer_grads = jax.tree.map(lambda x: jnp.zeros_like(x), params)

        join_fn = lambda path: ' '.join(i.key for i in path).lower()

        def init_vary_layer(key, x):
            path = join_fn(key)
            axes = ["model"]
            if 'moe' in path and 'feedforward' in path:
                if x.ndim == 4:
                    axes.extend(["fsdp", "tensor"])
            elif 'gamma' in path or 'beta' in path:
                axes.extend(["tensor"])
            elif x.ndim == 3:
                axes.extend(["fsdp", "tensor"])
            x = jax.lax.pvary(x, axis_name=(*axes,))
            return x

        layer_grads = jax.tree_util.tree_map_with_path(
            init_vary_layer,
            layer_grads,
        )

        grads = (embed_grads, layer_grads)

    grads, metrics = jax.lax.scan(step_fn, init=grads, xs=(x, y, key), unroll=1)
    metrics = jax.tree.map(lambda x: x.mean(), metrics)

    if grads is not None:
        grads = jax.tree.map(lambda x: x / grad_steps, grads)
        return grads, metrics

    return metrics


def main(config: config):

    mesh, count = setup_devices(config)

    log(json.dumps(cfg.__dict__, indent=4, default=lambda o: o.__dict__))

    checkpoint_dir = os.path.join(
        os.path.abspath(config.output_dir), config.name, "checkpoints"
    )
    load = os.path.exists(checkpoint_dir)

    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)

    key = KeyState(config.seed)
    model = shardedModel.get_model(cfg=config.model)

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

    total_steps = config.training_steps
    config.data.train_batch_size *= count["fsdp"] * config.grad_step
    config.data.val_batch_size *= count["fsdp"] * config.eval_steps

    log("setting up dataset")

    data_partition = jax.sharding.NamedSharding(
        mesh, P(None, "fsdp", "model", None, "tensor")
    )
    (
        train_dataset,
        val_dataset,
    ) = Dataset.getDataset(
        config.data, partition=data_partition, dp=count["fsdp"], pp=count["model"]
    )

    log(f"train steps: {len(train_dataset)} | val steps: {len(val_dataset)}")

    def save_checkpoint(step, wandb_id):
        if jax.process_index() != 0:
            return
        model_state = {
            "params": jax.device_get(state.params),
            "opt_state": jax.device_get(state.opt_state),
        }
        save_tree = {
            "state": model_state,
            "key": key.key,
            "train_step_idx": train_dataset.step_idx,
            "train_shard_idx": (train_dataset.shard_idx - 1) % len(train_dataset.data),
            "val_step_idx": val_dataset.step_idx,
            "val_shard_idx": (val_dataset.shard_idx - 1) % len(val_dataset.data),
            "step": step,
            "wandb_id": wandb_id,
        }
        checkpoint_manager.save(step, save_tree)

    init_step = 0
    use_wandb = config.wandb is True and jax.process_index() == 0
    wandb_id = None

    model_spec = shardedModel.get_p_spec(
        model=model,
        mesh=mesh,
        config=config.model,
    )

    if load:
        tree_state = checkpoint_manager.restore(checkpoint_manager.latest_step())
        init_step = tree_state["step"]
        log(f"loading checkpoint @ step {init_step}")

        key.key = tree_state["key"]

        state = TrainState.restore(tree_state["state"], tx, model_spec, mesh)

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
            table = wandb.Table(
                columns=["step"] + [f"tokens_{i}" for i in range(config.inference_batch * config.model.blocks * (jax.device_count() // config.model.blocks))],
                log_mode="INCREMENTAL",
            )
    else:
        log("No checkpoint found, starting from scratch")
        log("Creating model ...")

        params = shardedModel.get_params(
            cfg=config.model, model=model, mesh=mesh, key=key()
        )
        state = TrainState(params=params, tx=tx)

        if use_wandb:
            wandb.init(
                entity="waterloo2",
                project="jaxformer",
                name=config.name,
                resume="allow",
                config=asdict(config),
            )
            wandb_id = wandb.run.id
            table = wandb.Table(
                columns=["step"] + [f"tokens_{i}" for i in range(config.inference_batch * config.model.blocks * (jax.device_count() // config.model.blocks))],
                log_mode="INCREMENTAL",
            )

        save_checkpoint(0, wandb_id)

    log(f"Model parameter count: {state.n_params:,d} ")
    loss_fn = jax.tree_util.Partial(loss, model, config)

    key_spec = P("fsdp", "model", "tensor")
    data_spec = P("fsdp", "model", None, "tensor")
    train_step = jax.jit(
        jax.shard_map(
            lambda key, params, x, y: step(
                loss_fn, config.grad_step, params, key, x, y, train=True
            ),
            mesh=mesh,
            in_specs=(
                key_spec,
                model_spec,
                data_spec,
                data_spec,
            ),
            out_specs=(model_spec, P()),
        )
    )

    eval_step = jax.jit(
        jax.shard_map(
            lambda key, params, x, y: step(
                loss_fn, config.eval_steps, params, key, x, y, train=False
            ),
            mesh=mesh,
            in_specs=(
                key_spec,
                model_spec,
                data_spec,
                data_spec,
            ),
            out_specs=P(),
        )
    )

    log("start training")

    start = time.time()
    train_loss = 0.0
    sample_key = key()

    for current_step in range(init_step, total_steps):
        with jax.named_scope("train_step"):
            key_count = (
                count["fsdp"] * count["model"] * count["tensor"] * config.grad_step
            )
            if key_count == 1:
                keys = jnp.array([key()]).reshape(1, 1, 1, 1, 2)
            else:
                keys = key(key_count).reshape(
                    count["fsdp"], count["model"], count["tensor"], config.grad_step, 2
                )

            keys = jax.device_put(keys, jax.sharding.NamedSharding(mesh, key_spec))
            x, y = train_dataset()

            grads, metrics = train_step(keys, state.params, x, y)

        state = state.apply_gradients(grads=grads)
        train_loss += metrics["loss"]

        if use_wandb:
            wandb_log = {
                "step": current_step,
                "loss/train_loss": metrics["loss"],
                "loss/train_cross_entropy_loss": metrics["loss_cross"],
                "lr": state.opt_state[1].hyperparams["learning_rate"],
            }
            if config.model.moe:
                wandb_log["loss/load_loss"] = metrics["loss_load"]
                for h in range(config.model.n_experts):
                    wandb_log[f"load/head_{h}"] = metrics[f"load/head_{h}"]

        if current_step % config.checkpoint_steps == 0:
            end = time.time()
            total_time = end - start
            tokens_per_second = (
                config.data.train_batch_size * config.model.T * config.checkpoint_steps
            ) / total_time
            train_loss = (
                (train_loss / config.checkpoint_steps)
                if current_step > 0
                else train_loss
            )

            with jax.named_scope("eval_step"):
                key_count = (
                    count["fsdp"] * count["model"] * count["tensor"] * config.eval_steps
                )
                if key_count == 1:
                    keys = jnp.array([key()]).reshape(1, 1, 1, 1, 2)
                else:
                    keys = key(key_count).reshape(
                        count["fsdp"],
                        count["model"],
                        count["tensor"],
                        config.eval_steps,
                        2,
                    )

                x, y = val_dataset()
                metrics_val = eval_step(keys, state.params, x, y)

            if use_wandb:
                wandb_log["loss/val_loss"] = metrics_val["loss"]
                wandb_log["loss/val_cross_entropy_loss"] = metrics_val["loss_cross"]
                if config.model.moe:
                    wandb_log["loss/val_load_loss"] = metrics_val["loss_load"]
                    for h in range(config.model.n_experts):
                        wandb_log[f"load/head_{h}"] = metrics_val[f"load/head_{h}"]

            log_string = (
                f"step: {current_step} "
                + f" | val_loss: {float(metrics_val['loss']):.4f} "
                + f" | train_loss: {float(train_loss):.4f} "
                + f" | tokens/s: {float(tokens_per_second):.2f} "
                + f" | time: {float(end - start):.2f}s"
            )

            log(log_string)

            samples = shardedModel.generate(
                config.model, state.params, sample_key, x="hello", use_cache=False
            )
            log("sample tokens: \n")
            for tokens in samples:
                log(f"\t {tokens}\n")

            if jax.process_index() == 0:
                with open(
                    os.path.join(
                        os.path.abspath(config.output_dir), config.name, "tokens.txt"
                    ),
                    "a",
                ) as f:
                    f.write(f"{current_step} | {tokens}\n")

            if use_wandb:
                table.add_data(current_step, *samples)
                wandb_log["inference_tokens"] = table


            save_checkpoint(current_step, wandb_id)
            start = time.time()
            train_loss = 0.0

        if use_wandb:
            wandb.log(
                data=wandb_log,
                step=current_step,
            )

    if use_wandb:
        table = wandb.Table(
            columns=["step"] + [f"tokens_{i}" for i in range(config.inference_batch * config.model.blocks * (jax.device_count() // config.model.blocks))],
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
    cfg = parse_args()
    main(cfg)
