import os
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np
from typing import Optional, Tuple
from utils import dataConfig
from google.cloud import storage
import time


def log(out: str):
    if jax.process_index() == 0:
        print(out)


class Dataset:
    def __init__(
        self,
        process_path: str,
        T: int,
        batch_size: int,
        microbatch: int,
        dp: int,
        pp: int,
        bucket_name: str,
        id: str,
        partition: Optional[NamedSharding] = None,
    ):
        assert ((batch_size // dp) % microbatch) == 0, (
            "microbatch should divide batch size per data axis"
        )
        assert (microbatch % pp) == 0, "pp should divide microbatch size"

        self.T = T
        self.batch_size = batch_size
        self.dp = dp
        self.microbatch = microbatch

        self.step_idx = 0
        self.shard_idx = 0
        self.partition = partition

        self.bucket_name = bucket_name
        self.base_process_path = process_path
        self.process_path = process_path
        self.id = id
        self.data = self.return_blobs(bucket_name, self.id)
        self.dir_name = "bucket_downloads"
        try:
            os.mkdir(self.dir_name)
        except OSError as e:
            log(f"{self.dir_name} already exists")

        self.load_next_shard()

    def return_blobs(self, bucket_name, prefix, delimiter=None):
        res = []
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(
            bucket_name, prefix=prefix, delimiter=delimiter
        )
        for blob in blobs:
            res.append(blob.name)

        return res[1:]

    def download_blob_to_stream(self, bucket_name, source_blob_name, file_obj):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(source_blob_name)
        blob.download_to_file(file_obj)
        log(f"Downloaded blob {source_blob_name} to file-like object.")

        return file_obj

    def download_bucket(self, bucket_name, source_name, f):
        while True:
            try:
                result = self.download_blob_to_stream(bucket_name, source_name, f)
                return result
            except Exception as e:
                log("Failed to download due to exception")
                time.sleep(5)

    def download_next(self):
        log("Started downloading")
        source_name = self.data[self.shard_idx % len(self.data)]
        self.shard_idx += 1
        log(f" Downloading: {source_name} | Shard_idx: {self.shard_idx}")

        self.process_path = f"{self.base_process_path}_{self.id}_{self.shard_idx}"
        with open(self.process_path, "wb") as f:
            result = self.download_bucket(self.bucket_name, source_name, f)
            log(f"Done downloading {result}")

    def load_next_shard(self):
        self.download_next()

        def process_prev():
            log(f"Processing shard at {self.process_path}\n\n")
            try:
                data = np.load(self.process_path)
            except:
                log(f"couldn't load data\n\n")
            self.dataset = data[:-1]
            self.labels = data[1:]

            len_dataset = self.dataset.shape[0]
            max_batches = len_dataset // (self.batch_size * self.T)

            self.dataset = self.dataset[
                : max_batches * self.batch_size * self.T
            ].reshape(
                max_batches,
                self.microbatch,
                self.batch_size // self.microbatch,
                self.T,
            )
            self.labels = self.labels[: max_batches * self.batch_size * self.T].reshape(
                max_batches,
                self.microbatch,
                self.batch_size // self.microbatch,
                self.T,
            )

            self.dataset = jax.device_put(self.dataset, self.partition)
            self.labels = jax.device_put(self.labels, self.partition)

        process_prev()

        os.remove(self.process_path)

    def __len__(self):
        return self.dataset.shape[0]

    def __call__(self):
        x = self.dataset[self.step_idx]
        y = self.labels[self.step_idx]
        self.step_idx += 1

        if self.step_idx >= self.dataset.shape[0]:
            self.step_idx = 0
            self.load_next_shard()

        return x, y

    @classmethod
    def getDataset(
        cls,
        cfg: dataConfig,
        partition: Optional[NamedSharding] = None,
        dp: int = 1,
        pp: int = 1,
    ) -> Tuple["Dataset", "Dataset"]:
        train_dataset = cls(
            cfg.process_path,
            cfg.T,
            cfg.train_batch_size,
            cfg.micro_batch_size,
            partition=partition,
            dp=dp,
            pp=pp,
            bucket_name=cfg.bucket_name,
            id=cfg.train_folder_name,
        )
        val_dataset = cls(
            cfg.process_path,
            cfg.T,
            cfg.val_batch_size,
            cfg.micro_batch_size,
            partition=partition,
            dp=dp,
            pp=pp,
            bucket_name=cfg.bucket_name,
            id=cfg.val_folder_name,
        )

        return train_dataset, val_dataset


if __name__ == "__main__":
    test_cfg = dataConfig(
        bucket_name="10bt_gpt4",
        process_path="./bucket_downloads/processShard",
        train_folder_name="train",
        val_folder_name="val",
        T=1024,
        train_batch_size=16,
    )
    train, test = Dataset.getDataset(test_cfg, None)

    train_step_id = 20
    train_shard_id = ((train.shard_idx - 1) % len(train.data),)
    train.load_next_shard()

    val_step_id = 20
    val_shard_id = ((test.shard_idx - 1) % len(test.data),)
    test.load_next_shard()

    # start = time.time()
    # train, test = Dataset.getDataset(test_cfg, None)
    # jax.random.key(0)
    # end = time.time()
    # print(f"time taken to load dataset: {(end - start):.2f} seconds")
    # for i in range(6104 * 4):
    #     x,y = train()

    # print(len(train), len(test))
    # breakpoint()

    if os.path.exists(train.process_path):
        os.remove(train.process_path)
    if os.path.exists(test.process_path):
        os.remove(test.process_path)
