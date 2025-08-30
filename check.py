import orbax


import jax
import jax.numpy as jnp

#TODO: switch to gcs path
jax.config.update("jax_compilation_cache_dir", "gs://jaxformer-cache/")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

import optax
import numpy as np
import orbax.checkpoint as ocp

from dataclasses import asdict
from jax.sharding import PartitionSpec as P
from functools import partial
from typing import Tuple
from google.cloud import storage
from test2 import shardedModel
from dataset import Dataset
from utils import parse_args, config

checkpoint_dir = "gs://results_jaxformer/newEmbed5`"
options = ocp.CheckpointManagerOptions(max_to_keep=1)
checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)
load = checkpoint_manager.latest_step() is not None

print(load)