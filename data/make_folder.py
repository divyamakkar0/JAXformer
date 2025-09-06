# Before running: 
# 1) pip install -r requirements.txt in data dir
# 2) authenticate using gcloud CLI

from google.cloud import storage_control_v2
from dotenv import load_dotenv
import os

load_dotenv()

def create_folder(bucket_name: str, folder_name: str) -> None:
    storage_control_client = storage_control_v2.StorageControlClient()
    project_path = storage_control_client.common_project_path("_")
    bucket_path = f"{project_path}/buckets/{bucket_name}"

    request = storage_control_v2.CreateFolderRequest(
        parent=bucket_path,
        folder_id=folder_name,
    )
    response = storage_control_client.create_folder(request=request)

    print(f"Created folder: {response.name}")

if __name__ == '__main__':
  bucket_name = os.getenv("GCP_BUCKET", default="")

  for folder_name in ['train', 'test', 'checkpoints']:
    if bucket_name == "":
      print("GCP_BUCKET is not set")
      raise ValueError("GCP_BUCKET is not set")
    else:
      create_folder(bucket_name, folder_name)