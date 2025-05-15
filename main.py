import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from flax import linen as nn
from flax.training import train_state
import time
import math
import optax

from config import parse_args
from model import Decoder
from dataset import Dataset

from typing import Tuple, Any

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

def main(config):


    key = KeyState(config.seed) #fix this line

    print("setting up dataset")
    train_dataset, val_dataset, = Dataset.getDataset(cfg.data, key())

    print("setting up model")
    model, params = get_model(model_config=config.model, init_key=key())

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameter count: {param_count:,d} ")
    total_steps = config.training_steps

    #cosine scheduler
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.min_lr,
        peak_value=config.max_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.end_steps,
        end_value=config.end_lr,
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

    for current_step in range(total_steps):
        #get batch needs to be fixed
        x_t, label_t = get_batch(key=key(), scheduler=scheduler, data=train_dataset())
        grads, metrics = train_step_jit(key(), state.params, x, label_t, t, noise)
        wandb.log({"step": current_step, "train_loss": metrics['loss']})
        state = state.apply_gradients(grads=grads)
        train_loss += metrics['loss']

if __name__ == "__main__":
    cfg = parse_args()
    print(cfg)
    main(cfg)
