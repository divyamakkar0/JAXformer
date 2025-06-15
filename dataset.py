import os
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np
from typing import Optional, Tuple, List
from utils import dataConfig
import math


class Dataset:
    def __init__(
        self,
        data_path: str | List[str],
        T: int,
        batch_size: int,
        dataPartition: Optional[NamedSharding] = None,
    ):
        self.T = T
        self.batch_size = batch_size
        assert len(data_path) > 0, "data should not be empty"
        if isinstance(data_path, str):
            data_path = [data_path]
        self.data_path = data_path

        self.shard_idx = 0
        self.step_idx = 0
        self.partition = dataPartition

        self.load_next_shard(display=True)

    def load_next_shard(self, display: bool = False):
        data = np.load(self.data_path[self.shard_idx])
        self.dataset = data[: -self.T]
        self.labels = data[1 : -self.T + 1]

        len_dataset = self.dataset.shape[0]
        max_batches = len_dataset // (self.batch_size * self.T)

        self.dataset = self.dataset[: max_batches * self.batch_size * self.T].reshape(
            max_batches, self.batch_size, self.T
        )
        self.labels = self.labels[: max_batches * self.batch_size * self.T].reshape(
            max_batches, self.batch_size, self.T
        )

        if self.partition is not None:
            self.dataset = jax.make_array_from_callback(
                self.dataset.shape,
                lambda idx: self.dataset[idx],
                sharding=self.partition,
            )
            self.labels = jax.make_array_from_callback(
                self.labels.shape,
                lambda idx: self.labels[idx],
                sharding=self.partition,
            )
        else:
            self.dataset = jax.device_put(self.dataset)
            self.labels = jax.device_put(self.labels)

        if display:
            print(
                f"Loaded shard {self.data_path[self.shard_idx]} with {self.dataset.shape[0]:,d} tokens"
            )

        self.shard_idx = (self.shard_idx + 1) % len(self.data_path)

    def __len__(self):
        return self.dataset.shape[0]

    def __call__(self):
        x = self.dataset[self.step_idx]
        y = self.labels[self.step_idx]
        self.step_idx += 1

        if self.step_idx >= self.dataset.shape[0]:
            self.step_idx = 0
            self.load_next_shard(display=True)

        return x, y

    @classmethod
    def getDataset(
        cls, cfg: dataConfig, partition: Optional[NamedSharding] = None
    ) -> Tuple["Dataset", "Dataset"]:
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

        train_dataset = cls(
            train_dataset_path, cfg.T, cfg.train_batch_size, partition=None
        )
        val_dataset = cls(val_dataset_path, cfg.T, cfg.val_batch_size, partition=None)

        return train_dataset, val_dataset


if __name__ == "__main__":
    test_cfg = dataConfig(
        train_dataset_path="./trainSetShards",
        val_dataset_path="./valSetShards",
        T=1024,
        batch_size=16,
    )

    import time

    start = time.time()
    train, test = Dataset.getDataset(test_cfg, jax.random.key(0))
    end = time.time()
    print(f"time taken to load dataset: {(end - start):.2f} seconds")
    print(len(train), len(test))
    breakpoint()
