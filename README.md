## JAXformer

A Zero-to-One Guide on Scaling Modern Transformers with N-Dimensional Parallelism

## description



## structure of repo

## results

Results for a 1B model (300M active) trained to 3.28 val loss using 3-D sharding (8 FSDP, 2 Pipeline, 2 Tensor)

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
