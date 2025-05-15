import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from config import dataConfig
"""
TODO: Change the data_path to array and then make a load option which loads
this way we can make it on a larger dataset
for shakespeare this will work for now
"""

class Dataset:
    def __init__(
        self,
        data,
        T: int,
        batch_size: int,
        shuffle: bool = True,
        key: jax.random.key = jax.random.key(0)
    ):
        """
        data_path: where to load the shard from
        T: the max tokens in a sequence
        batch_size: the batch size
        idx: Tuple of the start and end index of the shard
        key: jax PRNG key to manage random
        """

        self.T = T
        self.batch_size = batch_size
        self.key = key

        self.data = data

        print("loading dataset ... ")

        inx = jnp.arange(self.data.shape[0] - T, dtype=jnp.int32)

        slice_fn = lambda i: jax.lax.dynamic_slice(self.data, (i,), (T,))
        self.dataset = jax.vmap(slice_fn, in_axes=(0))(inx)
        self.labels = jax.vmap(slice_fn, in_axes=(0))(inx + 1)


        if shuffle:
            self.key, shuffle_key = jax.random.split(self.key)
            idx = jax.random.permutation(shuffle_key, self.dataset.shape[0])
            self.dataset = self.dataset[idx]
            self.labels = self.labels[idx]

        self.idx = 0

    @classmethod
    def getDataset(cls, cfg: dataConfig, key: jax.random.key):

        data = jnp.load(cfg.dataset_path)
        idx = int(data.shape[0] * cfg.val_spilt)
        val_data, train_data = data[:idx], data[idx:]
        train_key, val_key = jax.random.split(key)
        train_dataset = cls(train_data, cfg.T, cfg.batch_size, shuffle=cfg.shuffle, key=train_key)
        val_dataset = cls(val_data, cfg.T, cfg.batch_size, shuffle=cfg.shuffle, key=val_key)

        return train_dataset, val_dataset

    def __len__(self):
        return self.dataset.shape[0]  # therotically should be ceil

    def __call__(self):

        if self.idx + self.batch_size < self.dataset.shape[0]:
            x = self.dataset[self.idx: self.idx + self.batch_size]
            y = self.labels[self.idx: self.idx + self.batch_size]
            self.idx += self.batch_size
        else:
            x = self.dataset[self.idx:]
            y = self.labels[self.idx:]
            self.idx = self.batch_size - x.shape[0]
            x = jnp.concat([x, self.dataset[:self.idx]])
            y = jnp.concat([y, self.labels[:self.idx]])


        return x, y

if __name__ == "__main__":

    test_cfg = dataConfig(
        dataset_path="./tokens.npy",
        val_spilt=0.1,
        T=1024,
        batch_size=3,
        shuffle=True
    )
    import time
    start = time.time()
    shake_dataset = Dataset.getDataset(test_cfg, jax.random.key(0))
    end = time.time()
    print(f"time taken to load dataset: {(end - start):.2f} seconds")
    print(len(shake_dataset))
    breakpoint()
