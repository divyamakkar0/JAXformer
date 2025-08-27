#!/bin/bash

echo "installing dependencies ... "
pip install -U "jax[tpu]"
pip install flax jaxtyping wandb tpu-info einops tiktoken
pip install google-cloud google-cloud-storage gcloud gcsfs

if [[ -z "~/.config/gcloud/application_default_credentials.json" ]]; then
  echo "gcloud storage crendinatials found "
fi

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo -e "\n\ndone run tpu-info to check"
