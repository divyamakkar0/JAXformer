#!/bin/bash


echo "installing dependencies ... "
pip install -U "jax[tpu]"
pip install flax jaxtyping wandb tpu-info einops tiktoken

echo "downloading fineweb 10B shards ... "
bash fineweb.sh

echo "done"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
tpu-info