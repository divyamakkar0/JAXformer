import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_autotune_level=3"
)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

import flax
from flax.training import train_state
import time
import ast
import json
import optax

from utils import parse_args, config, Metrics
from model import Decoder
from dataset import Dataset

from typing import Tuple, Any, Optional
import orbax.checkpoint as ocp
import wandb
from dataclasses import asdict


class KeyState:
    def __init__(self, base_key: jax.random.key):
        self.key = jax.random.key(base_key)

    def __call__(self, num: int = 2):
        self.key, rng = jax.random.split(self.key, num=num)
        return rng


def loss(model, alpha, params, key, x, y, train=True):
    B, T = x.shape
    pred, (_, load) = model.apply(
        {"params": params}, x, train=train, rngs={"dropout": key}
    )

    log_prob = jax.nn.log_softmax(pred, axis=-1).reshape(B * T, -1)
    y = y.reshape(B * T)
    loss_idx = lambda x, idx: jax.lax.dynamic_slice(x, (idx,), (1,))
    loss_cross = -(jax.vmap(loss_idx, in_axes=(0, 0))(log_prob, y)).mean()

    loss_balance = 0.0
    if load is not None:
        loss_balance = model.n_experts / (model.k * T**2) * load.sum(axis=0)

    loss = loss_cross + alpha * loss_balance
    return loss, (load, loss_cross, loss_balance)


def train_step(loss_fn, params, key, *args, **kwargs):
    loss = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    val, grads = loss(params, key, *args, **kwargs, train=True)
    loss, load, loss_cross, loss_balance = (
        val[0],
        *val[1],
    )

    metrics = {
        "loss": loss,
        "load": load,
        "loss_cross": loss_cross,
        "loss_load": loss_balance,
    }

    return grads, metrics


def eval_step(loss_fn, params, key, *args, **kwargs):
    loss, (load, load_cross, loss_balance) = loss_fn(
        params, key, *args, **kwargs, train=False
    )
    metrics = {
        "loss": loss,
        "load": load,
        "loss_cross": load_cross,
        "loss_load": loss_balance,
    }
    return metrics


def main(config: config):
    key = KeyState(config.seed)

    print("setting up dataset")
    (
        train_dataset,
        val_dataset,
    ) = Dataset.getDataset(cfg.data)

    print(f'train steps: {len(train_dataset)} | val steps: {len(val_dataset)}')
    print("setting up model")

    model, params = Decoder.get_model(model_config=config.model, init_key=key())

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameter count: {param_count:,d} ")
    total_steps = config.training_steps

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

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    loss_fn = jax.tree_util.Partial(loss, model, config.alpha)
    train_step_jit = jax.jit(
        lambda key, params, x, y: train_step(loss_fn, params, key, x, y),
    )
    eval_step_jit = jax.jit(
        lambda key, params, x, y: eval_step(loss_fn, params, key, x, y)
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
            "key": key.key,
            "train_step_idx": train_dataset.step_idx,
            "train_shard_idx": (train_dataset.shard_idx - 1) % len(train_dataset.data_path),
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

        key.key = tree_state["key"]
        state = flax.serialization.from_state_dict(state, tree_state["state"])

        train_dataset.step_idx = tree_state["train_step_idx"]
        train_dataset.shard_idx = tree_state["train_shard_idx"]
        train_dataset.load_next_shard()

        val_dataset.step_idx = tree_state["val_step_idx"]
        val_dataset.shard_idx = tree_state["val_shard_idx"]
        val_dataset.load_next_shard()

        init_step = tree_state["step"]
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

    metrics_step = Metrics(config.model.n_experts if config.model.moe else None)
    metrics_val = Metrics(config.model.n_experts if config.model.moe else None)

    print("start training")

    start = time.time()
    train_loss = 0.0
    sample_key = key()

    for current_step in range(init_step, total_steps):
        grads = None
        metrics_step.reset()
        for i in range(config.grad_step):
            grads_step, metrics = train_step_jit(key(), state.params, *train_dataset())
            grads = (
                grads_step
                if grads is None
                else jax.tree.map(lambda x, y: x + y, grads, grads_step)
            )
            metrics_step = metrics_step + metrics

        grads = jax.tree_util.tree_map(lambda x: x / config.grad_step, grads)
        state = state.apply_gradients(grads=grads)
        metrics_step = metrics_step / config.grad_step

        train_loss += metrics_step["loss"]

        if use_wandb:
            wandb_log = {
                "step": current_step,
                "loss/val__loss": metrics_step["loss"],
                "loss/val_cross_entropy_loss": metrics_step["loss_cross"],
                "lr": state.opt_state[1].hyperparams["learning_rate"],
            }
            if config.model.moe:
                wandb_log["loss/val_load_loss"] = metrics_step["loss_load"]
                for h in range(config.model.n_experts):
                    wandb_log[f"load/val_head_{h}"] = metrics_step[f"load/head_{h}"]

        if current_step % config.checkpoint_steps == 0:
            end = time.time()
            total_time = end - start
            tokens_per_second = (
                config.data.batch_size
                * config.grad_step
                * config.model.T
                * config.checkpoint_steps
            ) / total_time
            train_loss = (
                (train_loss / config.checkpoint_steps)
                if current_step > 0
                else train_loss
            )

            metrics_val = metrics_val.reset()
            for i in range(config.eval_steps):
                metrics = eval_step_jit(key(), state.params, *val_dataset())
                metrics_val = metrics_val + metrics
            metrics_val = metrics_val / config.checkpoint_steps
            if use_wandb:
                wandb_log["val_loss"] = metrics_val["loss"]

            log_string = (
                f"step: {current_step} "
                + f" | val_loss: {float(metrics_val['loss']):.4f} "
                + f" | train_loss: {float(train_loss):.4f} "
                + f" | tokens/s: {float(tokens_per_second):.2f} "
                + f" | time: {float(end - start):.2f}s"
            )

            print(log_string)

            tokens = model.generate(
                state.params,
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
