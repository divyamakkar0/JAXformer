import os
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from utils import dataConfig
import math


class Dataset:
    def __init__(
        self,
        data_path: str | List[str],
        T: int,
        batch_size: int,
    ):
        self.T = T
        self.batch_size = batch_size
        assert len(data_path) > 0, "data should not be empty"
        if isinstance(data_path, str):
            data_path = [data_path]
        self.data_path = data_path

        self.shard_idx = 0
        self.step_idx = 0

        self.load_next_shard(display=True)

    def load_next_shard(self, display: bool = False):
        data = jnp.load(self.data_path[self.shard_idx])
        self.dataset = data[: -self.T]
        self.labels = data[1 : -self.T + 1]
        if display:
            print(
                f"Loaded shard {self.data_path[self.shard_idx]} with {self.dataset.shape[0]:,d} tokens"
            )
        self.shard_idx = (self.shard_idx + 1) % len(self.data_path)

    def __len__(self):
        return int(
            math.ceil((self.dataset.shape[0] - self.T) / (self.batch_size * self.T))
        )

    def __call__(self):
        if self.step_idx + self.batch_size * self.T <= self.dataset.shape[0]:
            x = self.dataset[
                self.step_idx : self.step_idx + self.batch_size * self.T
            ].reshape(self.batch_size, self.T)

            y = self.labels[
                self.step_idx : self.step_idx + self.batch_size * self.T
            ].reshape(self.batch_size, self.T)
            self.step_idx += self.batch_size * self.T

        else:
            x_current_shard = self.dataset[self.step_idx :]
            y_current_shard = self.labels[self.step_idx :]

            self.step_idx = 0
            self.load_next_shard(display=True)

            self.step_idx = self.batch_size * self.T - x_current_shard.shape[0]
            x_next_shard = self.dataset[: self.step_idx]
            y_next_shard = self.labels[: self.step_idx]
            x = jnp.concatenate([x_current_shard, x_next_shard], axis=0).reshape(
                self.batch_size, self.T
            )
            y = jnp.concatenate([y_current_shard, y_next_shard], axis=0).reshape(
                self.batch_size, self.T
            )

        return x, y

    @classmethod
    def getDataset(cls, cfg: dataConfig) -> Tuple["Dataset", "Dataset"]:
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

        train_dataset = cls(train_dataset_path, cfg.T, cfg.train_batch_size)
        val_dataset = cls(val_dataset_path, cfg.T, cfg.val_batch_size)

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
