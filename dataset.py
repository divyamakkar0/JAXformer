import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
"""
TODO: Change the data_path to array and then make a load option which loads
this way we can make it on a larger dataset
for shakespeare this will work for now
"""

class Dataset:
    def __init__(
        self,
        data_path: str,
        T: int,
        batch_size: int,
        idx: Optional[Tuple] = None,
        key: jax.random.key = jax.random.key(0)
    ):
        """
        data_path: where to load the shard from
        T: the max tokens in a sequence
        batch_size: the batch size
        idx: Tuple of the start and end index of the shard
        key: jax PRNG key to manage random
        """

        self.data_path = data_path
        self.T = T
        self.batch_size = batch_size

        self.data = jnp.load(data_path, allow_pickle=True)
        self.key = key

        if idx is not None:
            self.data = self.data[idx[0]:idx[1]]

        self.dataset = jnp.stack([self.data[i: i + T] for i in range(0, self.data.shape[0] - T)])
        self.labels = jnp.stack([self.data[i + 1: i + T + 1] for i in range(0, self.data.shape[0] - T)])
        self.idx = 0

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



shake_dataset = Dataset("tokens.npy", 6, 3, (0,10) ,  key=jax.random.key(0))
print(len(shake_dataset))
breakpoint()
