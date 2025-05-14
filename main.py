import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from flax import linen as nn
from flax.training import train_state
import time

from config import parse_args
from model import Decoder
from dataset import Dataset


config = parse_args()

#key gen
class KeyState:
    def __init__(self, base_key: jax.random.key):
        self.key = jax.random.key(base_key)

    def __call__(self, num: int = 2):
        self.key, rng = jax.random.split(self.key, num=num)
        return rng


def get_model(model_config, init_key: jax.random.key, b, t, c) -> Tuple[nn.Module, Any]:

    x = jnp.ones((b, t, c))

    model = Decoder(model_config)
    params = model.init(init_key, x, training=False)['params']
    return model, params

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

def main(config):
   key = KeyState(config.seed) #fix this line

   print("setting up dataset")
   train_dataset = Dataset(config.dataset, config.batch_size, None, key())
   
   print("setting up model")
   model, params = get_model(model_config=config.model, init_key=key(), b=config.batch_size, t=config.T, c=config.model_dimension)
   
   param_count = sum(x.size for x in jax.tree.leaves(params))
   print(f"Model parameter count: {param_count:,d} ")
   total_steps = config.training_steps
   
   #cosine scheduler
   #optax adam optimizer

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
        x_t, label_t, t, noise = get_batch(key=key(), scheduler=scheduler, data=train_dataset())
        grads, metrics = train_step_jit(key(), state.params, x, label_t, t, noise)
        wandb.log({"step": current_step, "train_loss": metrics['loss']})
        state = state.apply_gradients(grads=grads)
        train_loss += metrics['loss']


    

   




