#!/bin/bash

# script to run to setup dependencies on TPU

pip install -U "jax[tpu]"
pip install flax jaxtyping wandb tpu-info einops tiktoken
pip install google-cloud google-cloud-storage gcloud gcsfs

if [[ -z "~/.config/gcloud/application_default_credentials.json" ]]; then
  echo "gcloud storage credentials found "
else
  echo "gcloud storage credentials not found "
  gcloud auth application-default login
fi

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo -e "\n\ndone run tpu-info to check"
