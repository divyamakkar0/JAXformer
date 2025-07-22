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


class Dataset:
    def __init__(
        self,
        download_path: List[str],
        T: int,
        batch_size: int,
        microbatch: int,
        dp: int,
        pp: int,
        train: bool,
        bucket_name: str,
        source_blob_name: str,
        partition: Optional[NamedSharding] = None,
    ):
        assert (batch_size % microbatch) == 0, "microbatch should divide batch size"
        assert (microbatch % pp) == 0, "pp should divide microbatch size"
        assert len(download_path) > 0, "data should not be empty"

        self.T = T
        self.batch_size = batch_size
        self.dp = dp
        self.microbatch = microbatch
        self.train = train 
        self.step_idx = 0
        self.train_idx = 1
        self.val_idx = 0
        self.partition = partition
        self.source_blob_name = source_blob_name

        self.prefix_train = "train/" + self.source_blob_name + "train"
        self.prefix_val = "val/" + self.source_blob_name + "val"
        self.bucket_name = bucket_name
        self.download_path = download_path
        

        self.train_size = self.len_blobs(self.bucket_name, self.prefix_train)
        self.val_size = self.len_blobs(self.bucket_name, self.prefix_val)

        self.load_next_shard(display=True)

    def len_blobs(self, bucket_name, prefix, delimiter=None):
        len = 0
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
        for blob in blobs:
            len += 1
        
        return len
    
    def download_blob_to_stream(self, bucket_name, source_blob_name, file_obj):
        """Downloads a blob to a stream or other file-like object."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        blob = bucket.blob(source_blob_name)
        blob.download_to_file(file_obj)
        print(f"Downloaded blob {source_blob_name} to file-like object.")

        return file_obj

    def download_bucket(self, bucket_name, source_name, f):
        result = self.download_blob_to_stream(bucket_name, source_name, f)
        while isinstance(result, Exception):
            print("Failed to download due to exception")
            time.sleep(5)
            result = self.download_bucket(bucket_name, source_name, f)
        
        print("Downloaded")
        return result

    def load_next_shard(self, display: bool = False):
        source_blob_name_train = "train/edufineweb_train_"
        source_blob_name_val = "val/edufineweb_val_"
        if self.train:
            size = self.train_size
            count = self.train_idx
        else:
            size = self.val_size
            count = self.val_idx

        if count <= size:
            padded = str(count).zfill(6)
            source_name = source_blob_name_train if self.train else source_blob_name_val 
            source_name += padded + ".npy"
            destination_file_name = self.download_path + padded
            
            if self.train:
                self.train_idx += 1
            else:
                self.val_idx += 1

            with open(destination_file_name, "wb") as f:
                result = self.download_bucket(self.bucket_name, source_name, f)
                print(result)

            data = np.load(destination_file_name)
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

            os.remove(destination_file_name)
            print(f'removed {destination_file_name} successfully')


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
        download = os.path.abspath(cfg.download_path)

        train_dataset = cls(
            download,
            cfg.T,
            cfg.train_batch_size,
            cfg.micro_batch_size,
            partition=partition,
            dp=dp,
            pp=pp,
            train=True,
            bucket_name=cfg.bucket_name,
            source_blob_name=cfg.source_blob_name,
        )
        val_dataset = cls(
            download,
            cfg.T,
            cfg.val_batch_size,
            cfg.micro_batch_size,
            partition=partition,
            dp=dp,
            pp=pp,
            train=False,
            bucket_name=cfg.bucket_name,
            source_blob_name=cfg.source_blob_name,
        )

        return train_dataset, val_dataset


if __name__ == "__main__":
    test_cfg = dataConfig(
        bucket_name="10bt_gpt4",
        source_blob_name="edufineweb_",
        download_path="./bucket_downloads/downloadedShard",
        T=1024,
        train_batch_size=16,
    )

    start = time.time()
    #jax.random.key(0)
    train, test = Dataset.getDataset(test_cfg, None)
    end = time.time()
    print(f"time taken to load dataset: {(end - start):.2f} seconds")
    print(len(train), len(test))
    breakpoint()
