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

    pred = model.apply({'params': params}, x, training=train, rngs={'dropout': key})
    log_prob = jax.nn.log_softmax(pred, axis=-1)
    targets = jnp.sum(log_prob * jax.nn.one_hot(y, pred.shape[-1]), axis=-1)
    loss = -jnp.mean(targets)
    return loss, pred

#MoE Loss

def train_step(loss_fn, params, key, *args, **kwargs):
    loss = jax.value_and_grad(
        loss_fn,
        argnums=0,
        has_aux=True
    )
    val, grads = loss(params, key, *args, **kwargs, train=True)
    loss, pred = val

    metrics = {
        'loss': loss,
        'pred': pred,
    }

    return grads, metrics

def eval_step(loss_fn, params, key, *args, **kwargs):
    loss, pred = loss_fn(params, key, *args, **kwargs, train=False)
    metrics = {
        'loss': loss,
        'pred': pred,
    }
    return metrics

def learning_rate(time_step, warmup_steps, total_steps, min_rate, max_rate):
    if time_step < warmup_steps:
        return min_rate + (max_rate - min_rate) * (time_step/warmup_steps)
    elif time_step <= total_steps:
        decay_steps = total_steps - warmup_steps
        decay_time = time_step - warmup_steps
        cosine_decay = 0.5 * (1 + jnp.cos(math.pi * (decay_time / decay_steps)))
        return min_rate + (max_rate - min_rate)*cosine_decay
    else:
        return min_rate

def main(config: config):
    """
    main function
    """


    key = KeyState(config.seed) #fix this line

    print("setting up dataset")
    train_dataset, val_dataset, = Dataset.getDataset(cfg.data, key())

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
            lambda key, params, x, y : train_step(loss_fn, params, key, x, y)
        )
    eval_step_jit = jax.jit(
        lambda key, params, x, y : eval_step( loss_fn, params, key, x, y)
        )

    start = time.time()
    train_loss = 0.0
    checkpoint_dir = config.checkpoint_dir
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
    checkpoint_dir, checkpointer, options)

    for current_step in range(total_steps):
        x_t, label_t = train_dataset()
        breakpoint()
        grads, metrics = train_step_jit(key(), state.params, x_t, label_t)
        # wandb.log({"step": current_step, "train_loss": metrics['loss']})
        state = state.apply_gradients(grads=grads)
        train_loss += metrics['loss']

        if current_step % cfg.checkpoint_steps == 0:
            checkpoint_manager.save(current_step, orbax_utils.from_train_state(state))

            print(f"step: {current_step}, train_loss: {metrics['loss']:.4f}, time: {time.time() - start:.2f}s")
            start = time.time()

if __name__ == "__main__":
    cfg = parse_args()
    print(cfg)
    main(cfg)
