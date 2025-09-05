## JAXformer

A Zero-to-One Guide on Scaling Modern Transformers with N-Dimensional Parallelism in JAX. The repo for [JAXformer](jaxformer.com) covers data processing, FSDP, pipeline parellism, tensor parellism, weight-sharding, activation-sharding, MoE scaling and much more.

## Structure

The model that is built up throughout the blog is in `model.py`. The main training script is in `main.py`. `utils.py` and `dataset.py` contain the dataclasses and dataset processing implementations. `debug_tpu.sh` launches a TMUX with 8 panes to ssh into 8 nodes at once running the command in the `command` variable. `launcher.sh` ssh's headlessly into each node and executves `run.sh` creating TMUX terminals inside the ssh to allow for runs to continue even if the ssh connection is broken. `setup_tpu.sh` setups all the dependencies on the TPU.

## Results

Results for a 1B model (300M active) trained to 3.28 val loss using 3-D sharding on a cluster of 32 TPU-v4(8 FSDP, 2 Pipeline, 2 Tensor).

### Val-loss
<p align="center">
  <img src="https://raw.githubusercontent.com/divyamakkar0/Jaxformer/main/public/loss-val.png" alt="Validation Loss" width="500"/>
</p>

### Load-loss
<p align="center">
  <img src="https://raw.githubusercontent.com/divyamakkar0/Jaxformer/main/public/loss-load.png" alt="Load Loss" width="500"/>
</p>

### Expert-per-head
<p align="center">
  <img src="https://raw.githubusercontent.com/divyamakkar0/Jaxformer/main/public/experts.png" alt="Experts per Head" width="500"/>
</p>


## Acknowledgements

This guide was written by Aditya Makkar, Divya Makkar, and Chinmay Jindal. The website uses a Distill-style Jekyll theme created by https://github.com/alshedivat/al-folio. The idea of the blog is inspired by https://jax-ml.github.io/scaling-book/ and the work of [Andrej Karpathy](https://x.com/karpathy). The [TRC](https://sites.research.google/trc/about/) was used to provide the compute needed. 
