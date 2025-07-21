import os
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np
from typing import Optional, Tuple, List
from utils import dataConfig
import math

from google.cloud import storage
import shutil
import time


bucket_name = "10bt_gpt4"
source_blob_name_train = "edufineweb_train_"
source_blob_name_val = "edufineweb_val_"
folder = "./bucket_downloads/"
os.makedirs(folder, exist_ok=True)
prefix_train = "edufineweb_train"
prefix_val = "edufineweb_val"

def download_blob_to_stream(bucket_name, source_blob_name, file_obj):
    """Downloads a blob to a stream or other file-like object."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(source_blob_name)
    blob.download_to_file(file_obj)
    print(f"Downloaded blob {source_blob_name} to file-like object.")

    return file_obj

def download_bucket(bucket_name, source_name, f):
    result = download_blob_to_stream(bucket_name, source_name, f)
    while isinstance(result, Exception):
        print("Failed to download due to exception")
        time.sleep(5)
        result = download_bucket(bucket_name, source_name, f)
     
    print("Downloaded")
    return result

def len_blobs(bucket_name, prefix, delimiter=None):
    len = 0
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    for blob in blobs:
        len += 1
    
    return len



class Dataset:
    def __init__(
        self,
        data_path: List[str],
        T: int,
        batch_size: int,
        microbatch: int,
        dp: int,
        pp: int,
        train: bool,
        partition: Optional[NamedSharding] = None,
        count: int = 0,
        train_size: int = 0,
        val_size: int = 0,
    ):
        assert (batch_size % microbatch) == 0, "microbatch should divide batch size"
        assert (microbatch % pp) == 0, "pp should divide microbatch size"
        # assert len(data_path) > 0, "data should not be empty"

        self.T = T
        self.batch_size = batch_size
        self.dp = dp
        self.microbatch = microbatch
        self.train = train 

        if isinstance(data_path, str):
            data_path = [data_path]
        self.data_path = data_path

        self.shard_idx = 0
        self.step_idx = 0
        self.val_count = 0
        self.partition = partition
        self.count = 1
        self.train_size = len_blobs(bucket_name, prefix_train)
        self.val_size = len_blobs(bucket_name, prefix_val)
        self.load_next_shard(display=True)

    def load_next_shard(self, display: bool = False):
        size = self.train_size if self.train else self.val_size
        count = self.count if self.train else self.val_count
        full_path = ""

        if count <= size:
            padded = str(count)
            padded = padded.zfill(6)
            source_name = source_blob_name_train if self.train else source_blob_name_val 
            source_name += padded + ".npy"
            destination_file_name = "./BucketDownload" + padded
            if self.train:
                self.count += 1
            else:
                self.val_count += 1

            with open(destination_file_name, "wb") as f:
                result = download_bucket(bucket_name, source_name, f)
                print(result)
            
            shutil.move(destination_file_name, folder)
            full_path += folder + destination_file_name

            data = np.load(full_path)
            self.dataset = data[:-1]
            self.labels = data[1:]

            len_dataset = self.dataset.shape[0]
            max_batches = len_dataset // (self.batch_size * self.T)

            self.dataset = self.dataset[: max_batches * self.batch_size * self.T].reshape(
                max_batches,
                self.dp,
                self.microbatch,
                self.batch_size // (self.dp * self.microbatch),
                self.T,
            )
            self.labels = self.labels[: max_batches * self.batch_size * self.T].reshape(
                max_batches,
                self.dp,
                self.microbatch,
                self.batch_size // (self.dp * self.microbatch),
                self.T,
            )

            if self.partition is not None:
                self.dataset = jax.make_array_from_callback(
                    self.dataset.shape,
                    sharding=self.partition,
                    data_callback=lambda idx: self.dataset[idx],
                )
                self.labels = jax.make_array_from_callback(
                    self.labels.shape,
                    sharding=self.partition,
                    data_callback=lambda idx: self.labels[idx],
                )
            else:
                self.dataset = jax.device_put(self.dataset)
                self.labels = jax.device_put(self.labels)

            os.remove(full_path)
            print(f'removed {full_path} successfully')


    def __len__(self):
        return self.dataset.shape[0]

    def __call__(self):
        size = self.train_size if self.train else self.val_size
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
            train_dataset_path,
            cfg.T,
            cfg.train_batch_size,
            cfg.micro_batch_size,
            partition=partition,
            dp=dp,
            pp=pp,
            train=True,
        )
        val_dataset = cls(
            val_dataset_path,
            cfg.T,
            cfg.val_batch_size,
            cfg.micro_batch_size,
            partition=partition,
            dp=dp,
            pp=pp,
            train=False,
        )

        return train_dataset, val_dataset


if __name__ == "__main__":
    test_cfg = dataConfig(
        train_dataset_path="./dataset/test", 
        val_dataset_path="./dataset/test",
        T=1024,
        train_batch_size=16,
    )

    import time

    start = time.time()
    #jax.random.key(0)
    train, test = Dataset.getDataset(test_cfg, None)
    end = time.time()
    print(f"time taken to load dataset: {(end - start):.2f} seconds")
    print(len(train), len(test))
    for i in range(6104):
        train()
    breakpoint()
