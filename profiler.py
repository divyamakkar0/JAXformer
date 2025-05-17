import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

import flax
from flax.training import train_state
import time
import ast
import optax

from config import parse_args, config
from model import Decoder
from dataset import Dataset

from typing import Tuple, Any
import orbax.checkpoint as ocp
import wandb
from dataclasses import asdict

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
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=lr_scheduler),
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    print("starting training")

    loss_fn = jax.tree_util.Partial(cross_entropy_loss, model)
    train_step_jit = jax.jit(
        lambda key, params, x, y : train_step(loss_fn, params, key, x, y),
        )
    eval_step_jit = jax.jit(lambda key, params, x, y : eval_step( loss_fn, params, key, x, y))

    checkpoint_dir = os.path.join(os.path.abspath(config.output_dir), config.name, "checkpoints")
    load = os.path.exists(checkpoint_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)

    def save_checkpoint(step):
        save_tree = {
                'state': flax.serialization.to_state_dict(state),
                'key': key.key,
                'train_idx': train_dataset.idx,
                'val_idx': val_dataset.idx,
                'step': step,
            }
        checkpoint_manager.save(step, save_tree)

    init_step = 0

    if load:
        print("loading checkpoint")
        tree_state = checkpoint_manager.restore(checkpoint_manager.latest_step())
        key.key = tree_state['key']
        state = flax.serialization.from_state_dict(state, tree_state['state'])
        train_dataset.idx = tree_state['train_idx']
        val_dataset.idx = tree_state['val_idx']
        init_step = tree_state['step']
    else:
        print("No checkpoint found, starting from scratch")
        save_checkpoint(0)


    use_wandb = config.wandb is True
    if use_wandb:
        wandb.init(
            entity="waterloo2",
            project="jaxformer",
            name=config.name,
            id=config.name,
            resume="allow",
            config=asdict(config),
        )

    start = time.time()
    train_loss = 0.0

    for current_step in range(init_step, total_steps):
        grads = None
        grad_loss = 0.0
        for i in range(config.grad_step):
            grads_step, metrics = train_step_jit(key(), state.params, *train_dataset())
            grads = grads_step if grads is None else jax.tree.map(
                lambda x, y: x + y, grads, grads_step
            )
            grad_loss += metrics['loss']

        grads = jax.tree_util.tree_map(lambda x: x / config.grad_step, grads)
        state = state.apply_gradients(grads=grads)
        train_loss += (grad_loss/config.grad_step)
        if use_wandb:
            wandb_log = {
                "step": current_step,
                "train_loss": grad_loss,
                "lr": state.opt_state.hyperparams["learning_rate"],
            }

        if current_step > 0 and current_step % config.checkpoint_steps == 0:
            end = time.time()
            total_time = end - start
            tokens_per_second = (config.data.batch_size * config.grad_step)/ total_time
            val_loss = 0.0
            for i in range(config.checkpoint_steps):
                metrics = eval_step_jit(key(), state.params, *val_dataset())
                val_loss += metrics['loss']
            val_loss /= config.checkpoint_steps
            if use_wandb:
                wandb_log['val_loss'] = val_loss
            print(f"step: {current_step} | val_loss: {val_loss:.4f} | train_loss: {train_loss / cfg.checkpoint_steps:.4f} | tokens/s: {tokens_per_second:.2f} | time: {end - start:.2f}s")

            save_checkpoint(current_step)

            tokens = model.generate(
                state.params,
                key(),
                "hello",
                B=config.inference_batch,
                k=10000,
                max_tokens=30,
                temperature=1,
            )
            print("tokens: ", tokens)
            with open(os.path.join(os.path.abspath(config.output_dir), config.name, "tokens.txt"), "a") as f:
                f.write(f"{current_step} | {tokens}\n")

            start = time.time()
            train_loss = 0.0

        if use_wandb:
                    wandb.log(wandb_log)

    if use_wandb:
        table = wandb.Table(columns=[f"tokens_{i}" for i in range(config.inference_batch)])
        wandb.Table.MAX_ROWS = total_steps // config.checkpoint_steps
        with open(
            os.path.join(os.path.abspath(config.output_dir), config.name, "tokens.txt"), "r"
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
    print(cfg)
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        main(cfg)
