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

import flax
from flax.training import train_state
import time
import ast
import optax

from config import parse_args, modelConfig
from model import Decoder

from typing import Tuple, Any
from dataclasses import asdict

from main import cross_entropy_loss, train_step


def main(config: modelConfig):
    """
    main function
    """

    key = jax.random.key(0)
    print("setting up model")
    key, init_key = jax.random.split(key)
    model, params = Decoder.get_model(model_config=config.model, init_key=init_key)

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameter count: {param_count:,d} ")

    # cosine scheduler
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.lr.min_lr,
        peak_value=config.lr.max_lr,
        warmup_steps=config.lr.warmup_steps,
        decay_steps=config.lr.end_steps,
        end_value=config.lr.end_lr,
    )

    # optax adam optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=lr_scheduler),
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    print("starting profiling")

    loss_fn = jax.tree_util.Partial(cross_entropy_loss, model)
    train_step_jit = jax.jit(
        lambda key, params, x, y: train_step(loss_fn, params, key, x, y),
    )

    B = 100
    T = config.model.T
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        key, x_init, y_init, dropout_key = jax.random.split(key, num=4)

        x = jax.random.randint(x_init, (B, T), 0, config.model.vocab_size)
        y = jax.random.randint(y_init, (B, T), 0, config.model.vocab_size)

        grad, metrics = train_step_jit(dropout_key, state.params, x, y)

        # tokens = model.generate(
        #     state.params,
        #     key,
        #     "hello",
        #     B=config.inference_batch,
        #     k=10000,
        #     max_tokens=30,
        #     temperature=1,
        # )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)


# get hte model
# init, wrap one step in the train
