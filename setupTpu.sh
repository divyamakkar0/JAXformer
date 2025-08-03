#!/bin/bash


echo "installing dependencies ... "
pip install -U "jax[tpu]"
pip install flax jaxtyping wandb tpu-info einops tiktoken
pip install google-cloud
pip install google-cloud-storage
pip install gcloud
pip install gcloud-aio
pip install gcloud-aio-storage

echo "downloading fineweb 10B shards ... "
bash fineweb.sh

echo "done"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
tpu-info