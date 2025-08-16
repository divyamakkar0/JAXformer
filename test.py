import jax
import jax.numpy as jnp
from functools import partial
from jax.sharding import PartitionSpec as P
import numpy as np
from test2 import Transformer
from dataset import Dataset
from utils import dataConfig
import time

MODEL_DIM = 512
VOCAB_SIZE = 100277
BLOCKS = 2
LAYERS_PER_BLOCK = 5
NUM_HEADS = 2
DATA_PARALLEL = 32
SEQ_LEN = 1024
DROPOUT_RATE = 0.1
DATA_FACTOR = 20
LATENT_DIM = 64
DHR = 64
MODEL_DTYPE=jnp.bfloat16

jax.distributed.initialize()

devices = np.array(jax.devices())
if jax.process_index() == 0:
    for idx in np.ndindex(devices.shape):
        d = devices[idx]
        print(
            f"  {idx} ID: {d.id}, Process: {d.process_index}, "
            f"Coords: {d.coords}, Core: {d.core_on_chip}"
        )

mesh = jax.make_mesh((DATA_PARALLEL,), ('dp',))

data_partition = jax.sharding.NamedSharding(
    mesh,
    P(None, 'dp', None, None, None),
)

data_cfg = dataConfig(
    bucket_name="10bt_gpt4",
    process_path="./bucket_downloads/processShard",
    train_folder_name="train",
    val_folder_name="val",
    train_batch_size=DATA_PARALLEL * DATA_FACTOR,
    T=SEQ_LEN,
    val_batch_size=DATA_PARALLEL * DATA_FACTOR,
    micro_batch_size=1,
)

train_dataset, val_dataset = Dataset.getDataset(
    data_cfg,
    partition=data_partition,
    dp=DATA_PARALLEL,
)

model = Transformer(
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

@jax.jit
@partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P('dp')),
        out_specs=P(),
)
def init_weights(x):
    params = model.init(jax.random.PRNGKey(1), x, train=False)['params']
    sharding = jax.NamedSharding(mesh, P())
    params = jax.tree.map(
        lambda x: jax.device_put(x, sharding),
        params,
    )
    return params

x_init = jnp.ones((32, 1, SEQ_LEN), dtype=jnp.int32)
params = init_weights(x_init)
param_count = jax.tree.reduce(
    lambda x, y: x + y.size,
    params,
    0,
)

def step(params, x, y, key, train=True):
    def loss_fn(params, x, y, key):
        logits, _ = model.apply({'params': params}, x, rngs={"dropout": key})
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        M, B, T, V = logits.shape
        y = y.reshape(-1)
        log_probs = log_probs.reshape(M * B * T, V)

        loss_idx = lambda x, idx: jax.lax.dynamic_slice(x, (idx,), (1,))
        loss = -(jax.vmap(loss_idx, in_axes=(0, 0))(log_probs, y)).mean()
        loss = jax.lax.pmean(loss, axis_name='dp')

        return loss
    if train:
        loss_fn = jax.value_and_grad(loss_fn)
    x,y = x[0], y[0]
    val = loss_fn(params, x, y, key)
    loss, grads = val if train else (val, None)
    return loss, grads

train_step = jax.jit(
    jax.shard_map(
        lambda params, x, y, key: step(params, x, y, key=key[0], train=True),
        mesh=mesh,
        in_specs=(P(), P('dp'), P('dp'), P('dp')),
        out_specs=P(),
    ),
)
eval_step = jax.jit(
    jax.shard_map(
        lambda params, x, y, key: step(params, x, y, key=key[0], train=False),
        mesh=mesh,
        in_specs=(P(), P('dp'), P('dp'), P('dp')),
        out_specs=P(),
    ),
)

MAX_STEPS = 10
total_tokens = DATA_FACTOR * DATA_PARALLEL * SEQ_LEN
lr = 4e-3

jax.experimental.multihost_utils.sync_global_devices('sync')
if jax.process_index() == 0:
    print(f"Total parameters: {param_count:,}")
    print(f"Total steps: {MAX_STEPS}")
    print(f"Total tokens per step: {total_tokens:,}")
    print(f"Learning rate: {lr}")

key = jax.random.PRNGKey(0)
if jax.process_index() == 0:
    start = time.time()
for i in range(MAX_STEPS):
    key, train_key, eval_key = jax.random.split(key, 3)
    train_key = jax.random.split(train_key, DATA_PARALLEL)
    train_key = jnp.asarray(train_key)
    eval_key = jax.random.split(eval_key, DATA_PARALLEL)
    eval_key = jnp.asarray(eval_key)

    x, y = train_dataset()
    loss, grads = train_step(params, x, y, train_key)
    eval_x, eval_y = val_dataset()
    eval_loss, _ = eval_step(params, eval_x, eval_y, eval_key)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)


    loss, eval_loss = loss.item(), eval_loss.item()
    jax.experimental.multihost_utils.sync_global_devices('sync')
    if jax.process_index() == 0:
        time_per_batch = time.time() - start
        tokens_per_second = 2 * total_tokens / time_per_batch
        log_string = f"Step {i+1}, Loss: {loss:.4f}, Eval Loss: {eval_loss:.4f}, tk/s: {tokens_per_second:,.2f}"
        print(log_string)
        start = time.time()
