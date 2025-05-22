import os
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from config import dataConfig
import math


class Dataset:
    def __init__(
        self,
        data_path: str | List[str],
        T: int,
        batch_size: int,
        key: jax.random.key = jax.random.key(0),
    ):

        self.T = T
        self.batch_size = batch_size
        self.key = key
        assert len(data_path) > 0, "data should not be empty"
        if isinstance(data_path, str):
            data_path = [data_path]
        self.data_path = data_path

        self.load_shared(self.data_path[0], display=True)

        self.steps = len(self)
        self.idx = 0

    def load_shared(self, shard: str, display: bool = False):

        data = jnp.load(shard)
        self.dataset = data[: -self.T]
        self.labels = data[1 : -self.T + 1]
        if display:
            print(f"Loaded shard {shard} with {self.dataset.shape[0]} tokens")

    def __len__(self):
        return int(
            math.ceil((self.dataset.shape[0] - self.T) / (self.batch_size * self.T))
        )

    def __call__(self):
        self.key, sample_key = jax.random.split(self.key)
        idx = jax.random.permutation(sample_key, self.dataset.shape[0] - self.T)[
            : self.batch_size
        ]

        stack_data = lambda dataset:  jax.vmap(
            lambda i: jax.lax.dynamic_slice(dataset, (i,), (self.T,)), in_axes=(0)
        )(idx)

        x = stack_data(self.dataset)
        y = stack_data(self.labels)

        self.idx += 1

        if self.idx >= self.steps:

            self.idx = 0
            dp = self.data_path.pop(0)
            self.load_shared(dp, display=True)
            self.data_path.append(dp)

        return x, y

    @classmethod
    def getDataset(cls, cfg: dataConfig, key: jax.random.key):
        train_dataset_path = os.path.abspath(cfg.train_dataset_path)
        if os.path.isdir(train_dataset_path):
            train_dataset_path = [
                os.path.join(train_dataset_path, f)
                for f in os.listdir(train_dataset_path)
                if f.endswith(".npy")
            ]

        val_dataset_path = os.path.abspath(cfg.val_dataset_path)
        if os.path.isdir(val_dataset_path):
            val_dataset_path = [
                os.path.join(val_dataset_path, f)
                for f in os.listdir(val_dataset_path)
                if f.endswith(".npy")
            ]

        train_dataset = cls(train_dataset_path, cfg.T, cfg.batch_size, key=key)
        val_dataset = cls(val_dataset_path, cfg.T, cfg.batch_size, key=key)

        return train_dataset, val_dataset


if __name__ == "__main__":
    test_cfg = dataConfig(
        train_dataset_path="./trainSetShards",
        val_dataset_path="./valSetShards",
        T=1024,
        batch_size=64,
    )

    import time

    start = time.time()
    train, test = Dataset.getDataset(test_cfg, jax.random.key(0))
    end = time.time()
    print(f"time taken to load dataset: {(end - start):.2f} seconds")
    print(len(train), len(test))
    breakpoint()
