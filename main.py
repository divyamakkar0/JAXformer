import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)


import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

from flax.training import train_state
import time
import math
import optax

from config import parse_args, config
from model import Decoder
from dataset import Dataset

from typing import Tuple, Any
import orbax.checkpoint
import shutil
from flax.training import orbax_utils

#key gen
class KeyState:
    def __init__(self, base_key: jax.random.key):
        self.key = jax.random.key(base_key)

    def __call__(self, num: int = 2):
        self.key, rng = jax.random.split(self.key, num=num)
        return rng

def cross_entropy_loss(model, params, key, x, y, train=True):

    B, T = x.shape
    pred, cache = model.apply({'params': params}, x, train=train, rngs={'dropout': key})
    log_prob = jax.nn.log_softmax(pred, axis=-1).reshape(B * T, -1)
    y = y.reshape(B * T)
    loss = -jnp.mean(log_prob[jnp.arange(B * T), y])
    return loss, (pred, cache)

#MoE Loss

def train_step(loss_fn, params, key, *args, **kwargs):
    loss = jax.value_and_grad(
        loss_fn,
        argnums=0,
        has_aux=True
    )
    val, grads = loss(params, key, *args, **kwargs, train=True)
    loss, pred, _ = val[0], *val[1] # don't need cache in training

    metrics = {
        'loss': loss,
        'pred': pred,
    }

    return grads, metrics

def eval_step(loss_fn, params, key, *args, **kwargs):
    loss, (pred, cache) = loss_fn(params, key, *args, **kwargs, train=False)
    pred = jax.nn.softmax(pred, axis=-1)
    metrics = {
        'loss': loss,
        'pred': pred,
        'cache': cache,
    }
    return metrics

def main(config: config):
    """
    main function
    """


    key = KeyState(config.seed) #fix this line

    print("setting up dataset")
    train_dataset, val_dataset, = Dataset.getDataset(cfg.data, key())
    print(len(train_dataset), len(val_dataset))

    print("setting up model")
    model, params = Decoder.get_model(model_config=config.model, init_key=key())

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameter count: {param_count:,d} ")
    total_steps = config.training_steps

    #cosine scheduler
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.lr.min_lr,
        peak_value=config.lr.max_lr,
        warmup_steps=config.lr.warmup_steps,
        decay_steps=config.lr.end_steps,
        end_value=config.lr.end_lr,
    )

    #optax adam optimizer
    tx = optax.adam(lr_scheduler)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    print("starting training")

    loss_fn = jax.tree_util.Partial(cross_entropy_loss, model)
    train_step_jit = jax.jit(
        lambda key, params, x, y : train_step(loss_fn, params, key, x, y),
        donate_argnames=("params")
        )
    eval_step_jit = jax.jit(lambda key, params, x, y : eval_step( loss_fn, params, key, x, y))



    checkpoint_dir = os.path.abspath(config.checkpoint_dir)
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
    checkpoint_dir, checkpointer, options)

    start = time.time()
    train_loss = 0.0
    for current_step in range(total_steps):
        grads = None
        grad_loss = 0.0
        for i in range(config.grad_step):
            x_t, label_t = train_dataset()
            grads_step, metrics = train_step_jit(key(), state.params, x_t, label_t)
            # wandb.log({"step": current_step, "train_loss": metrics['loss']})
            if grads is None:
                grads = grads_step
            else:
                grads = jax.tree.map(lambda x, y: x + y, grads, grads_step)
            grad_loss += metrics['loss']

        grads = jax.tree_util.tree_map(lambda x: x / config.grad_step, grads)
        state = state.apply_gradients(grads=grads)
        train_loss += grad_loss / config.grad_step

        if current_step > 0 and current_step % cfg.checkpoint_steps == 0:

            # checkpoint_manager.save(current_step, state)
            val_loss = 0.0
            for i in range(config.checkpoint_steps):
                x_t, label_t = val_dataset()
                metrics = eval_step_jit(key(), state.params, x_t, label_t)
                # wandb.log({"step": current_step, "val_loss": metrics['loss']})
                val_loss += metrics['loss']
            val_loss /= config.checkpoint_steps
            print(f"step: {current_step}, val_loss: {val_loss:.4f}, train_loss: {train_loss / cfg.checkpoint_steps:.4f}, time: {time.time() - start:.2f}s")

            start = time.time()
            train_loss = 0.0

if __name__ == "__main__":
    cfg = parse_args()
    print(cfg)
    main(cfg)
