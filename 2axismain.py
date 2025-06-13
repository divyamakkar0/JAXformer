import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "
)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", False)

import flax
from flax.training import train_state
import time
import ast
import json
import optax

from utils import parse_args, config
from model import Decoder
from dataset import Dataset

import wandb
from dataclasses import asdict

import orbax.checkpoint as ocp

from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from functools import partial
import numpy as np


def setup_devices(cfg: config):
    device_cfg = cfg.device_config
    assert device_cfg.n_axis == len(device_cfg.n_device_axis)
    assert device_cfg.n_axis == len(device_cfg.n_device_name)

    jax.distributed.initialize()
    devices = np.array(jax.devices())
    n_devices = devices.shape[0]

    assert n_devices == np.prod(device_cfg.n_device_axis)

    while devices.ndim < device_cfg.n_axis:
        devices = devices[..., None]
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

    mesh = Mesh(devices, axis_names=device_cfg.n_device_name)
    count = devices.shape

    return mesh, count


#TODO: init model
def init_state(mesh, config, model, params, *, step=0, opt_state=None):
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

    if opt_state is None:
        opt_state = tx.init(params)

    @jax.jit
    @partial(jax.shard_map, mesh=mesh, in_specs=(P(), P()), out_specs=(P()))
    def state_fn(params, opt_state):
        state = train_state.TrainState(
            step=step,
            apply_fn=model.apply,
            params=params,
            tx=tx,
            opt_state=opt_state
        )
        return state

    state_init = state_fn(params, opt_state)
    return state_init


class KeyState:
    def __init__(self, seed: int):
        self.key = jax.random.PRNGKey(seed)

    def __call__(self, num: int = 1):
        self.key, *rng = jax.random.split(self.key, num=num + 1)
        if num == 1:
            return rng[0]
        else:
            return jnp.array(rng)


def loss(model, alpha, params, key, x, y, train):
    B, T = x.shape
    pred, (_, load) = model.apply(
        {"params": params}, x, train=train, rngs={"dropout": key}
    )
    log_prob = jax.nn.log_softmax(pred, axis=-1).reshape(B * T, -1)
    loss_idx = lambda x, idx: jax.lax.dynamic_slice(x, (idx,), (1,))
    y = y.reshape(B * T)
    loss_cross = -(jax.vmap(loss_idx, in_axes=(0, 0))(log_prob, y)).mean()

    loss_balance = 0.0
    if load is not None:
        loss_balance = model.n_experts / (model.k * T**2) * load.sum(axis=0)

    loss = loss_cross + alpha * loss_balance

    return loss, (load, loss_cross, loss_balance)


def step(loss_fn, grad_steps, params, key, x, y, train=True):
    if train:
        loss_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    def step_fn(grads, batch):
        *data, key = batch
        val = loss_fn(params, key, *data, train=train)

        if train:
            val, grads_new = val
            grads = jax.tree.map(
                lambda x, y: x + y,
                grads,
                grads_new,
            )
        else:
            grads = None

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

    B, T = x.shape

    x = x.reshape(grad_steps, B // grad_steps, T)
    y = y.reshape(grad_steps, B // grad_steps, T)

    grads = None
    if train:
        grads = jax.tree.map(lambda x: jnp.zeros_like(x), params)

    grads, metrics = jax.lax.scan(step_fn, init=grads, xs=(x, y, key), unroll=1)

    if grads is not None:
        grads = jax.tree.map(lambda x: x / grad_steps, grads)
        grads = jax.lax.pmean(grads, axis_name="data")

    metrics = jax.tree.map(lambda x: x.mean(), metrics)
    metrics = jax.lax.pmean(metrics, axis_name="data")

    return grads, metrics


def main(config: config):
    print("setting up devices")
    mesh, count = setup_devices(config)

    key = KeyState(config.seed)

    print("setting up model")
    model, global_params = Decoder.get_model(model_config=config.model, init_key=key())
    param_count = sum(x.size for x in jax.tree.leaves(global_params))
    print(f"Model parameter count: {param_count:,d} ")

    print("setting up state & dataset")
    state = init_state(mesh, config, model, global_params)

    total_steps = config.training_steps
    config.data.train_batch_size *= count * config.grad_step
    config.data.val_batch_size *= count * config.eval_steps
    print("setting up dataset")
    (
        train_dataset,
        val_dataset,
    ) = Dataset.getDataset(config.data)

    print(f"train steps: {len(train_dataset)} | val steps: {len(val_dataset)}")

    loss_fn = jax.tree_util.Partial(loss, model, config.alpha)

    train_step = jax.jit(
        jax.shard_map(
            lambda key, params, x, y: step(
                loss_fn, config.grad_step, params, key, x, y, train=True
            ),
            mesh=mesh,
            in_specs=(P("data"), P(), P("data"), P("data")),
            out_specs=(P(), P()),
        )
    )

    eval_step = jax.jit(
        jax.shard_map(
            lambda key, params, x, y: step(
                loss_fn, config.eval_steps, params, key, x, y, train=False
            )[1],
            mesh=mesh,
            in_specs=(P("data"), P(), P("data"), P("data")),
            out_specs=(P()),
        )
    )

    checkpoint_dir = os.path.join(
        os.path.abspath(config.output_dir), config.name, "checkpoints"
    )
    load = os.path.exists(checkpoint_dir)

    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)

    def save_checkpoint(step, wandb_id):
        save_tree = {
            "state": flax.serialization.to_state_dict(state),
            "key": jax.device_get(key.key),
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
        tree_state = checkpoint_manager.restore(checkpoint_manager.latest_step())

        init_step = tree_state["step"]
        key.key = tree_state["key"]
        unsharded_state = flax.serialization.from_state_dict(state, tree_state["state"])
        state = init_state(
            mesh,
            config,
            model,
            params=unsharded_state.params,
            step=init_step,
            opt_state=unsharded_state.opt_state,
        )

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
        print(f"loading checkpoint @ step {init_step}")
    else:
        print("No checkpoint found, starting from scratch")
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

    print("start training")

    start = time.time()
    train_loss = 0.0
    sample_key = key()

    for current_step in range(init_step, total_steps):
        with jax.named_scope("train_step"):
            if count * config.grad_step == 1:
                keys = jnp.array([key()])
            else:
                keys = key(count * config.grad_step)

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
                if count * config.grad_step == 1:
                    keys = jnp.array([key()])
                else:
                    keys = key(count * config.grad_step)
                metrics_val = eval_step(
                    key(count * config.eval_steps),
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

            params_host_device = jax.device_put(
                jax.device_get(state.params), mesh.devices[0]
            )
            tokens = model.generate(
                params_host_device,
                sample_key,
                "",
                B=config.inference_batch,
                k=10000,
                max_tokens=30,
                temperature=1,
            )

            print("tokens: ", tokens)
            with open(
                os.path.join(
                    os.path.abspath(config.output_dir), config.name, "tokens.txt"
                ),
                "a",
            ) as f:
                f.write(f"{current_step} | {tokens}\n")

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
