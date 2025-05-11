import jax
import jax.numpy as jnp


class Dataset:
    def __init__(self, data_path, batch_size, idx, key):
        self.data_path = data_path
        self.batch_size = batch_size
        self.data = jnp.load(data_path, allow_pickle=True)

        self.start = 0
        self.end = len(self.data)

        self.key = key

        if idx is not None:
            self.start = idx[0]
            self.end = idx[1]

    def __len__(self):
        return (self.end - self.start) // self.batch_size

    def __call__(self):
        self.key, data_key = jax.random.split(self.key)
        idx = jax.random.randint(
            data_key, (1,), self.start, self.end - self.batch_size
        )[0]
        batch = self.data[idx : idx + self.batch_size]

        return batch


shake_dataset = Dataset("tokens.npy", 32, None, jax.random.key(0))
print(len(shake_dataset))
breakpoint()
