import jax
import jax.numpy as jnp
from functools import partial
import optax
from jax.sharding import PartitionSpec as P
import numpy as np
from test2 import ModelConfig, shardedModel
from dataset import Dataset
from utils import dataConfig
import time


MIN_LR = 0.0
MAX_LR = 4e-3
END_LR = 4e-4
WARMUP_STEPS = 0
END_STEPS = 6000
GRAD_CLIP_NORM = 1.0

MODEL_DIM = 12
VOCAB_SIZE = 100277
BLOCKS = 4
LAYERS_PER_BLOCK = 1
NUM_HEADS = 2
SEQ_LEN = 6
DROPOUT_RATE = 0.1
BATCH_SIZE = 20
MICRO_BATCH_SIZE = 4
LATENT_DIM = 8
DHR = 8
MODEL_DTYPE = jnp.bfloat16

DATA_PARALLEL = 4
LAYER_PARALLEL = 4
TENSOR_PARALLEL = 2

def log(msg: str):
    if jax.process_index() == 0:
        print(msg)


jax.distributed.initialize()

devices = np.array(jax.devices())
for idx in np.ndindex(devices.shape):
    d = devices[idx]
    log(
        f"  {idx} ID: {d.id}, Process: {d.process_index}, "
        f"Coords: {d.coords}, Core: {d.core_on_chip}"
    )

assert devices.size == DATA_PARALLEL * LAYER_PARALLEL * TENSOR_PARALLEL, (
    f"Expected {DATA_PARALLEL * LAYER_PARALLEL * TENSOR_PARALLEL} devices, got {devices.shape[0]}"
)

mesh = jax.make_mesh((DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL), ("dp", "pp", "tp"))

data_cfg = dataConfig(
    bucket_name="10bt_gpt4",
    process_path="./bucket_downloads/processShard",
    train_folder_name="train",
    val_folder_name="val",
    train_batch_size=DATA_PARALLEL * BATCH_SIZE,
    T=SEQ_LEN,
    val_batch_size=DATA_PARALLEL * BATCH_SIZE,
    micro_batch_size=MICRO_BATCH_SIZE,
)

data_partition = jax.sharding.NamedSharding(
    mesh,
    P(None, "pp", "dp", "tp"),
)
train_dataset, val_dataset = Dataset.getDataset(
    data_cfg,
    partition=data_partition,
    dp=DATA_PARALLEL,
)

modelCfg = ModelConfig(
    model_dimension=MODEL_DIM,
    vocab_size=VOCAB_SIZE,
    n_head=NUM_HEADS,
    blocks=BLOCKS,
    layers_per_block=LAYERS_PER_BLOCK,
    T=SEQ_LEN,
    latent_dim=LATENT_DIM,
    dhR=DHR,
    dropout_rate=DROPOUT_RATE,
    model_dtype=MODEL_DTYPE,
)

model = shardedModel(modelCfg)

log("creating sharded model ...")
params = model.init_weights(jax.random.PRNGKey(0), mesh)

lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=MIN_LR,
        peak_value=MAX_LR,
        warmup_steps=WARMUP_STEPS,
        decay_steps=END_STEPS,
        end_value=END_LR,
)
tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.inject_hyperparams(optax.adamw)(lr_scheduler),
)

default_sharding = jax.sharding.NamedSharding(mesh, P())
opt_state = jax.tree.map(
    lambda x: x if np.ndim(x) != 0 else jax.device_put(x, default_sharding),
    tx.init(params),
)


param_count = jax.tree.reduce(
    lambda x, y: x + y.size,
    params,
    0,
)
log(f"Total parameters: {param_count:,}")


def step(params, x, y, key, train=True):
    def loss_fn(params, x, y, key):
        logits, _ = model.pipe_step(
            params,
            x,
            key=key,
            train=train,
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        M, B, T, V = logits.shape
        y = y.reshape(-1)
        log_probs = log_probs.reshape(M * B * T, V)

        loss_idx = lambda x, idx: jax.lax.dynamic_slice(x, (idx,), (1,))
        loss = -(jax.vmap(loss_idx, in_axes=(0, 0))(log_probs, y)).mean()
        loss = jax.lax.pmean(loss, axis_name="pp")
        loss = jax.lax.pmean(loss, axis_name="tp")
        loss = jax.lax.pmean(loss, axis_name="dp")

        return loss

    if train:
        loss_fn = jax.value_and_grad(loss_fn)

    key = key.reshape(2,)
    val = loss_fn(params, x, y, key)
    loss, grads = val if train else (val, None)

    return loss, grads


param_spec = shardedModel.get_p_spec([model.embedding, model.block], mesh, modelCfg)
opt_spec = jax.tree.map(
    lambda x: x.sharding.spec,
    opt_state
)
data_spec = P("pp", "dp", "tp")
key_spec = P("dp", "pp", "tp")

@jax.jit
def update_params(params, opt_state, x,y, key):
    loss, grads = step(params, x, y, key, train=True)
    grads = jax.tree.map(
        lambda g: jax.lax.pmean(g, axis_name="pp"),
        grads,
    )
    grads = jax.tree.map(
        lambda g: jax.lax.pmean(g, axis_name="tp"),
        grads,
    )
    grads = jax.tree.map(
        lambda g: jax.lax.pmean(g, axis_name="dp"),
        grads,
    )
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

train_step = jax.jit(
    jax.shard_map(
        lambda params, opt_state, x, y, key: update_params(params, opt_state, x, y, key),
        mesh=mesh,
        in_specs=(param_spec, opt_spec, data_spec, data_spec, key_spec),
        out_specs=(param_spec, opt_spec, P()),
        check_vma=False
    )
)

eval_step = jax.jit(
    jax.shard_map(
        lambda params, x, y, key: step(params, x, y, key=key, train=False)[0],
        mesh=mesh,
        in_specs=(param_spec, data_spec, data_spec, key_spec),
        out_specs=P(),
        check_vma=False
    ),
)

MAX_STEPS = 10
total_tokens = BATCH_SIZE * DATA_PARALLEL * SEQ_LEN
lr = 4e-3

jax.experimental.multihost_utils.sync_global_devices("sync")
log(f"Total parameters: {param_count:,}")
log(f"Total steps: {MAX_STEPS}")
log(f"Total tokens per step: {total_tokens:,}")
log(f"Learning rate: {lr}")

key = jax.random.PRNGKey(0)
key, sample_key = jax.random.split(key, 2)
start = time.time()

for i in range(MAX_STEPS):
    key, train_key, eval_key = jax.random.split(key, 3)
    train_key = jax.random.split(train_key, DATA_PARALLEL * LAYER_PARALLEL * TENSOR_PARALLEL)
    train_key = jnp.asarray(train_key).reshape((DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL, 2))
    eval_key = jax.random.split(eval_key, DATA_PARALLEL * LAYER_PARALLEL * TENSOR_PARALLEL)
    eval_key = jnp.asarray(eval_key).reshape((DATA_PARALLEL, LAYER_PARALLEL, TENSOR_PARALLEL, 2))

    x, y = train_dataset()
    params, opt_state, loss = train_step(params, opt_state, x, y, train_key)

    eval_x, eval_y = val_dataset()
    eval_loss = eval_step(params, eval_x, eval_y, eval_key)

    loss, eval_loss = loss.item(), eval_loss.item()
    jax.experimental.multihost_utils.sync_global_devices("sync")
    time_per_batch = time.time() - start
    tokens_per_second = 2 * total_tokens / time_per_batch
    log_string = f"Step {i + 1}, Loss: {loss:.4f}, Eval Loss: {eval_loss:.4f}, tk/s: {tokens_per_second:,.2f}"
    log(log_string)
    start = time.time()

outputs = model.generate(
    params,
    modelCfg,
    key=sample_key,
    x="hello world",
    B=1,
    k=10000,
    temperature=1.0,
    n_devices=1,
    use_cache=True,
)

log("Generated outputs:")
for output in outputs:
    log(f"\t{output}")