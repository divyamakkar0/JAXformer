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
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_disable_jit", True)
else:
    jax.config.update("jax_debug_nans", True)
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

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from functools import partial
from typing import Optional, Tuple, Type, TypeVar
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
    def restore(cls, restored: PyTree, tx, mesh):
        params = shardedModel.shard_params((*restored["params"],), mesh=mesh)

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


def setup_devices(cfg: config):
    device_cfg = cfg.device_config
    assert 2 == len(device_cfg.n_device_axis)

    jax.distributed.initialize()
    devices = np.array(jax.devices())[:, None]

    assert devices.shape[0] == np.prod(device_cfg.n_device_axis)
    devices = devices.reshape(*device_cfg.n_device_axis)

    platform = jax.devices()[0].platform
    if platform == "tpu":
        print("Available TPU Devices:")
        print(f"Device array shape: {devices.shape}")
        for idx in np.ndindex(devices.shape):
            d = devices[idx]
            print(
                f"  {idx} ID: {d.id}, Process: {d.process_index}, "
                f"Coords: {d.coords}, Core: {d.core_on_chip}"
            )

    mesh = Mesh(devices, axis_names=("data", "model"))
    print(f"Mesh: {mesh}")
    count = devices.shape
    count = {
        "data": count[0],
        "model": count[1],
    }

    return mesh, count


def loss(model, alpha, params, key, x, y, train):
    M, B, T = x.shape

    pred, (_, load) = shardedModel.pipe_step(model, params, x, key=key, train=train)

    total_tokens = M * B * T
    log_prob = jax.nn.log_softmax(pred, axis=-1).reshape(total_tokens, -1)
    y = y.reshape(total_tokens)

    loss_idx = lambda x, idx: jax.lax.dynamic_slice(x, (idx,), (1,))
    loss_cross = -(jax.vmap(loss_idx, in_axes=(0, 0))(log_prob, y)).mean()

    loss_balance = 0.0
    if load is not None:
        loss_balance = model.n_experts / (model.k * T**2) * load.sum(axis=0)

    loss = loss_cross + alpha * loss_balance
    aux_stat = (load, loss_cross, loss_balance)

    loss = jax.lax.pmean(loss, axis_name="model")
    aux_stat = jax.lax.pmean(aux_stat, axis_name="model")

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
            load = load.mean(axis=0)
            metrics["loss_load"] = loss_balance
            for h in range(load.shape[0]):
                metrics[f"load/head_{h}"] = load[h]

        return grads, metrics

    x = rearrange(x, "1 m (g b) t -> g m b t", g=grad_steps)
    y = rearrange(y, "1 m (g b) t -> g m b t", g=grad_steps)
    key = rearrange(key, "1 m g s -> g m s", g=grad_steps, s=2)

    grads = None
    if train:
        grads_init = lambda x: jax.lax.pvary(jnp.zeros_like(x), "model")
        grads = jax.tree.map(grads_init, params)

    grads, metrics = jax.lax.scan(step_fn, init=grads, xs=(x, y, key), unroll=1)
    metrics = jax.tree.map(lambda x: x.mean(), metrics)
    metrics = jax.lax.pmean(metrics, axis_name="data")

    if grads is not None:
        grads = jax.tree.map(lambda x: x / grad_steps, grads)
        grads = jax.lax.pmean(grads, axis_name="data")

        embed_grads, layer_grads = grads
        embed_grads = jax.lax.pmean(embed_grads, axis_name="model")
        grads = (embed_grads, layer_grads)

        return grads, metrics

    return metrics


def main(config: config):
    print("setting up devices")
    mesh, count = setup_devices(config)

    checkpoint_dir = os.path.join(
        os.path.abspath(config.output_dir), config.name, "checkpoints"
    )
    load = os.path.exists(checkpoint_dir)

    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)

    key = KeyState(config.seed)

    print("setting up state")
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
    config.data.train_batch_size *= count["data"] * config.grad_step
    config.data.val_batch_size *= count["data"] * config.eval_steps

    print("setting up dataset")
    data_partition = jax.sharding.NamedSharding(
        mesh,
        P(None, "data", "model", None, None),
    )
    (
        train_dataset,
        val_dataset,
    ) = Dataset.getDataset(
        config.data, partition=data_partition, dp=count["data"], pp=count["model"]
    )

    print(f"train steps: {len(train_dataset)} | val steps: {len(val_dataset)}")

    def save_checkpoint(step, wandb_id):
        model_state = {
            "params": jax.device_get(state.params),
            "opt_state": jax.device_get(state.opt_state),
        }
        save_tree = {
            "state": model_state,
            "key": key.key,
            "train_step_idx": train_dataset.step_idx,
            "train_shard_idx": (train_dataset.shard_idx - 1)
            % len(train_dataset.data_path),
            "val_step_idx": val_dataset.step_idx,
            "val_shard_idx": (val_dataset.shard_idx - 1) % len(val_dataset.data_path),
            "step": step,
            "wandb_id": wandb_id,
        }
        checkpoint_manager.save(step, save_tree)

    init_step = 0
    use_wandb = config.wandb is True
    wandb_id = None

    if load:
        print(f"loading checkpoint @ step {init_step}")

        tree_state = checkpoint_manager.restore(checkpoint_manager.latest_step())
        init_step = tree_state["step"]
        key.key = tree_state["key"]

        state = TrainState.restore(tree_state["state"], tx, mesh)

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
        print("No checkpoint found, starting from scratch")
        print("Creating model ...")

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
        save_checkpoint(0, wandb_id)

    print(f"Model parameter count: {state.n_params:,d} ")
    out = shardedModel.generate(
        model,
        state.params,
        key(),
        mesh,
        x="hello"
    )

    breakpoint()

    loss_fn = jax.tree_util.Partial(
        loss,
        model,
        config.alpha,
    )

    train_step = jax.jit(
        jax.shard_map(
            lambda key, params, x, y: step(
                loss_fn, config.grad_step, params, key, x, y, train=True
            ),
            mesh=mesh,
            in_specs=(
                P("data", "model"),
                (P(), P("model")),
                P("data", "model"),
                P("data", "model"),
            ),
            out_specs=((P(), P("model")), P()),
        )
    )

    eval_step = jax.jit(
        jax.shard_map(
            lambda key, params, x, y: step(
                loss_fn, config.eval_steps, params, key, x, y, train=False
            ),
            mesh=mesh,
            in_specs=(
                P("data", "model"),
                (P(), P("model")),
                P("data", "model"),
                P("data", "model"),
            ),
            out_specs=P(),
        )
    )

    print("start training")

    start = time.time()
    train_loss = 0.0
    sample_key = key()

    for current_step in range(init_step, total_steps):
        with jax.named_scope("train_step"):
            key_count = count["data"] * count["model"] * config.grad_step
            if key_count == 1:
                keys = jnp.array([key()])[None, :]
            else:
                keys = key(key_count).reshape(
                    count["data"], count["model"], config.grad_step, 2
                )

            keys = jax.device_put(
                keys, jax.sharding.NamedSharding(mesh, P("data", "model"))
            )

            grads, metrics = train_step(
                keys,
                state.params,
                *train_dataset(),
            )

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
                key_count = count["data"] * count["model"] * config.eval_steps
                if key_count == 1:
                    keys = jnp.array([key()])[None, :]
                else:
                    keys = key(key_count).reshape(
                        count["data"], count["model"], config.eval_steps, 2
                    )

                metrics_val = eval_step(
                    keys,
                    state.params,
                    *val_dataset(),
                )

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

            print(log_string)

            # params_host_device = jax.device_put(
            #     jax.device_get(state.params), mesh.devices[0]
            # )
            # tokens = model.generate(
            #     params_host_device,
            #     sample_key,
            #     "",
            #     B=config.inference_batch,
            #     k=10000,
            #     max_tokens=30,
            #     temperature=1,
            # )

            # print("tokens: ", tokens)
            # with open(
            #     os.path.join(
            #         os.path.abspath(config.output_dir), config.name, "tokens.txt"
            #     ),
            #     "a",
            # ) as f:
            #     f.write(f"{current_step} | {tokens}\n")

            # save_checkpoint(current_step, wandb_id)
            start = time.time()
            train_loss = 0.0

        if use_wandb:
            wandb.log(
                data=wandb_log,
                step=current_step,
            )

    if use_wandb:
        table = wandb.Table(
            columns=[f"tokens_{i}" for i in range(config.inference_batch)]
        )
        wandb.Table.MAX_ROWS = total_steps // config.checkpoint_steps
        with open(
            os.path.join(os.path.abspath(config.output_dir), config.name, "tokens.txt"),
            "r",
        ) as f:
            lines = f.readlines()
        for line in lines:
            tokens = line.split("|")[1]
            tokens = ast.literal_eval(tokens)
            table.add_data(tokens)

        wandb.log({"inference_tokens": table})
        wandb.finish()


if __name__ == "__main__":
    cfg = parse_args()
    print(json.dumps(cfg.__dict__, indent=4, default=lambda o: o.__dict__))
    main(cfg)
