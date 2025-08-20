import jax
import jax.numpy as jnp
from functools import partial
from jax.sharding import PartitionSpec as P
import numpy as np
from test2 import ModelConfig, shardedModel
from dataset import Dataset
from utils import dataConfig
import time

MODEL_DIM = 512
VOCAB_SIZE = 100277
BLOCKS = 2
LAYERS_PER_BLOCK = 2
NUM_HEADS = 2
DATA_PARALLEL = 16
LAYER_PARALLEL = 2
SEQ_LEN = 1024
DROPOUT_RATE = 0.1
BATCH_SIZE = 20
MICRO_BATCH_SIZE = 4
LATENT_DIM = 64
DHR = 64
MODEL_DTYPE = jnp.bfloat16

jax.distributed.initialize()

devices = np.array(jax.devices())
if jax.process_index() == 0:
    for idx in np.ndindex(devices.shape):
        d = devices[idx]
        print(
            f"  {idx} ID: {d.id}, Process: {d.process_index}, "
            f"Coords: {d.coords}, Core: {d.core_on_chip}"
        )

assert devices.shape == (DATA_PARALLEL * LAYER_PARALLEL,), (
    f"Expected {DATA_PARALLEL * LAYER_PARALLEL} devices, got {devices.shape[0]}"
)

mesh = jax.make_mesh((DATA_PARALLEL, LAYER_PARALLEL), ("dp", "pp"))

data_partition = jax.sharding.NamedSharding(
    mesh,
    P(None, None, "dp", None),
)

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

train_dataset, val_dataset = Dataset.getDataset(
    data_cfg,
    partition=data_partition,
    dp=DATA_PARALLEL,
)

# modelCfg = ModelConfig(
#     model_dimension=MODEL_DIM,
#     vocab_size=VOCAB_SIZE,
#     n_head=NUM_HEADS,
#     blocks=BLOCKS,
#     layers_per_block=LAYERS_PER_BLOCK,
#     T=SEQ_LEN,
#     latent_dim=LATENT_DIM,
#     dhR=DHR,
#     dropout_rate=DROPOUT_RATE,
#     model_dtype=MODEL_DTYPE,
# )

modelCfg = ModelConfig(
    model_dimension=128,
    vocab_size=100277,
    n_head=8,
    blocks=4,
    layers_per_block=2,
    T=128,
    latent_dim=64,
    dhR=32,
    dropout_rate=0.2,
    model_dtype=jnp.bfloat16,
)



model = shardedModel(modelCfg)

print("creating sharded model ...")
params = model.init_weights(jax.random.PRNGKey(0), mesh)
param_count = jax.tree.reduce(
    lambda x, y: x + y.size,
    params,
    0,
)
print(f"Total parameters: {param_count:,}")


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

        return loss

    if train:
        loss_fn = jax.value_and_grad(loss_fn)
    key = key.reshape(
        2,
    )

    val = loss_fn(params, x, y, key)
    grads = None
    if train:
        loss, grads = val
        grads = jax.tree.map(
            lambda g: jax.lax.pmean(g, axis_name="dp"),
            grads,
        )
    else:
        loss = val
    loss = jax.lax.pmean(loss, axis_name="dp")
    return loss, grads


out_spec = shardedModel.get_p_spec([model.embedding, model.block], mesh, modelCfg)
data_spec = P("pp", "dp")
key_spec = P("dp", "pp")

train_step = jax.jit(
    jax.shard_map(
        lambda params, x, y, key: step(params, x, y, key=key, train=True),
        mesh=mesh,
        in_specs=(out_spec, data_spec, data_spec, key_spec),
        out_specs=(P(), out_spec),
        check_vma=False,
    )
)

eval_step = jax.jit(
    jax.shard_map(
        lambda params, x, y, key: step(params, x, y, key=key, train=False)[0],
        mesh=mesh,
        in_specs=(out_spec, data_spec, data_spec, key_spec),
        out_specs=P(),
        check_vma=False,
    ),
)

MAX_STEPS = 10
total_tokens = BATCH_SIZE * DATA_PARALLEL * SEQ_LEN
lr = 4e-3

jax.experimental.multihost_utils.sync_global_devices("sync")
if jax.process_index() == 0:
    print(f"Total parameters: {param_count:,}")
    print(f"Total steps: {MAX_STEPS}")
    print(f"Total tokens per step: {total_tokens:,}")
    print(f"Learning rate: {lr}")

key = jax.random.PRNGKey(0)
key, sample_key = jax.random.split(key, 2)
if jax.process_index() == 0:
    start = time.time()

# for i in range(MAX_STEPS):
#     key, train_key, eval_key = jax.random.split(key, 3)
#     train_key = jax.random.split(train_key, DATA_PARALLEL * LAYER_PARALLEL)
#     train_key = jnp.asarray(train_key).reshape((DATA_PARALLEL, LAYER_PARALLEL, 2))
#     eval_key = jax.random.split(eval_key, DATA_PARALLEL * LAYER_PARALLEL)
#     eval_key = jnp.asarray(eval_key).reshape((DATA_PARALLEL, LAYER_PARALLEL, 2))

#     x, y = train_dataset()
#     loss, grads = train_step(params, x, y, train_key)
#     eval_x, eval_y = val_dataset()
#     eval_loss = eval_step(params, eval_x, eval_y, eval_key)

#     params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

#     loss, eval_loss = loss.item(), eval_loss.item()
#     jax.experimental.multihost_utils.sync_global_devices("sync")
#     if jax.process_index() == 0:
#         time_per_batch = time.time() - start
#         tokens_per_second = 2 * total_tokens / time_per_batch
#         log_string = f"Step {i + 1}, Loss: {loss:.4f}, Eval Loss: {eval_loss:.4f}, tk/s: {tokens_per_second:,.2f}"
#         print(log_string)
#         start = time.time()

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

if jax.process_index() == 0:
    print("Generated outputs:")
    for output in outputs:
        print(output)