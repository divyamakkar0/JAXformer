"""
Naive implementation (single process) for uploading locally saved shards to GCP bucket
"""

from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv('BUCKET_NAME')
DATA_CACHE_DIR = os.getenv('DATA_CACHE_DIR')
FILE_NAMES = os.listdir(DATA_CACHE_DIR)
WORKERS = 8

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    print('working on', source_file_name)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


for s in (sorted(os.listdir(DATA_CACHE_DIR))):
    path = str(DATA_CACHE_DIR) + "/" + s
    # print(path)
    upload_blob(BUCKET_NAME, path, s)