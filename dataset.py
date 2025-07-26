import os
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np
from typing import Optional, Tuple, List
from utils import dataConfig
import math
from google.cloud import storage
from gcloud.aio.storage import Storage
import shutil
import time
import asyncio
import threading 

class Dataset:
    def __init__(
        self,
        download_path: str,
        process_path : str,
        T: int,
        batch_size: int,
        microbatch: int,
        dp: int,
        pp: int,
        bucket_name: str,
        id: str,
        partition: Optional[NamedSharding] = None,
    ):
        assert (batch_size % microbatch) == 0, "microbatch should divide batch size"
        assert (microbatch % pp) == 0, "pp should divide microbatch size"
        assert len(download_path) > 0, "data should not be empty"

        self.T = T
        self.batch_size = batch_size
        self.dp = dp
        self.microbatch = microbatch


        self.step_idx = 0
        self.shard_idx = 0
        self.partition = partition

        self.bucket_name = bucket_name
        self.download_path = download_path
        self.base_process_path = process_path 
        self.process_path = process_path
        self.id = id
        self.data = self.return_blobs(bucket_name, self.id)

        self.download_thread = threading.Thread(target=self.download_next)
        self.download_thread.start()
        
        self.load_next_shard()

    def return_blobs(self, bucket_name, prefix, delimiter=None):
        res = []
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
        for blob in blobs:
            res.append(blob.name)
        
        return res[1:]
    
    def download_blob_to_stream(self, bucket_name, source_blob_name, file_obj):
        """Downloads a blob to a stream or other file-like object."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        blob = bucket.blob(source_blob_name)
        blob.download_to_file(file_obj)
        print(f"Downloaded blob {source_blob_name} to file-like object.")

        return file_obj

    def download_bucket(self, bucket_name, source_name, f):
        while True:
            try:
                result = self.download_blob_to_stream(bucket_name, source_name, f)
                return result
            except Exception as e:
                print("Failed to download due to exception")
                time.sleep(5)
    
    def download_next(self):
        print("Started downloading")
        source_name = self.data[self.shard_idx % len(self.data)]
        self.shard_idx += 1
        print(f" Downloading {source_name} and {self.shard_idx}")

        self.process_path = f"{self.base_process_path}_{self.id}_{self.shard_idx}"
        with open(self.process_path, "wb") as f:
            result = self.download_bucket(self.bucket_name, source_name, f)
            print(f"Done downloading {result}")

    def load_next_shard(self):
        
        def process_prev():
            print("Processing shard")
            data = np.load(self.process_path)
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

        print("waiting for download to finish..")
        self.download_thread.join()
        print("download done")
        process_prev()
        
        os.remove(self.process_path)

        self.download_thread = threading.Thread(target=self.download_next)
        self.download_thread.start()

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
        download = os.path.abspath(cfg.download_path)

        train_dataset = cls(
            download,
            cfg.process_path,
            cfg.T,
            cfg.train_batch_size,
            cfg.micro_batch_size,
            partition=partition,
            dp=dp,
            pp=pp,
            bucket_name=cfg.bucket_name,
            id=cfg.train_folder_name
        )
        val_dataset = cls(
            download,
            cfg.process_path,
            cfg.T,
            cfg.val_batch_size,
            cfg.micro_batch_size,
            partition=partition,
            dp=dp,
            pp=pp,
            bucket_name=cfg.bucket_name,
            id=cfg.val_folder_name
        )

        return train_dataset, val_dataset


if __name__ == "__main__":
    test_cfg = dataConfig(
        bucket_name="10bt_gpt4",
        download_path="./bucket_downloads/downloadedShard",
        process_path="./bucket_downloads/processShard",
        train_folder_name="train",
        val_folder_name="val",
        T=1024,
        train_batch_size=16,
    )

    
    start = time.time()
    train, test = Dataset.getDataset(test_cfg, None)
    #jax.random.key(0)
    end = time.time()
    print(f"time taken to load dataset: {(end - start):.2f} seconds")
    # for i in range(6104 * 4):
    #     x,y = train()

    print(len(train), len(test))
    breakpoint()

    if os.path.exists(train.process_path):
        os.remove(train.process_path)
    if os.path.exists(test.process_path):
        os.remove(test.process_path)