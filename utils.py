import argparse
from dataclasses import dataclass, field
from typing import List, Optional
import jax.numpy as jnp
from jax.numpy import dtype
import json


@dataclass
class modelConfig:
    """model config class"""

    model_dimension: int
    n_heads: int
    T: int
    dhR: int
    dhR_blocks: int
    vocab_size: int
    dropout: float
    blocks: int
    moe: bool = False
    k: int = 0
    n_experts: int = 0
    n_shared: int = 0
    latent_dim: int = 0
    model_dtype: str = "bfloat16"
    grad_checkpoint: bool = False


@dataclass
class dataConfig:
    bucket_name: str
    source_blob_name: str = "edufineweb_"
    download_path: str = "./bucket_downloads/downloadedShard"
    T: int = 6
    train_batch_size: int = 3
    val_batch_size: int = 3
    micro_batch_size: int = 1


@dataclass
class LRConfig:
    """class for keeping track of learning rate args"""

    max_lr: float = 4e-3
    min_lr: float = 0
    end_lr: float = 4e-4
    warmup_steps: int = 1000
    end_steps: int = 6000


@dataclass
class deviceConfig:
    """class for distrbuted config"""

    n_device_axis: List[int]


@dataclass
class config:
    """class for keeping track of model args"""

    model: modelConfig
    data: dataConfig
    lr: LRConfig
    training_steps: int
    name: str
    device_config: deviceConfig
    grad_step: int = 1
    alpha: float = 0.001
    output_dir: str = "./results/"
    checkpoint_steps: int = 10
    checkpoint_manager: str = "./checkpoints/manager/"
    inference_batch: int = 1
    eval_steps: int = 25
    seed: int = 0
    wandb: bool = True
    grad_clip_norm: float = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="model training")
    parser.add_argument("--model_dimension", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dhR", type=int, default=64)
    parser.add_argument("--dhR_blocks", type=int, default=4)
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n_shared", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=64)

    parser.add_argument(
        "--train_dataset", type=str, default="./fineweb-edu-10bt-for-gpt2/train"
    )
    parser.add_argument(
        "--val_dataset", type=str, default="./fineweb-edu-10bt-for-gpt2/test"
    )
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=4)

    parser.add_argument("--max_lr", type=float, default=4e-3)
    parser.add_argument("--min_lr", type=float, default=0)
    parser.add_argument("--end_lr", type=float, default=4e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--end_steps", type=int, default=6000)

    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--name", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/")
    parser.add_argument("--checkpoint_steps", type=int, default=100)
    parser.add_argument(
        "--checkpoint_manager", type=str, default="./checkpoints/manager/"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--grad_checkpoint", action="store_true")

    parser.add_argument("--training_steps", type=int, default=10000)
    parser.add_argument("--grad_step", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--inference_batch", type=int, default=1)
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument(
        "--n_device_axis",
        type=int,
        nargs="*",
        default=[1],
    )

    args = parser.parse_args()

    model_cfg = modelConfig(
        model_dimension=args.model_dimension,
        n_heads=args.n_heads,
        T=args.T,
        dhR=args.dhR,
        dhR_blocks=args.dhR_blocks,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
        blocks=args.blocks,
        n_experts=args.n_experts,
        n_shared=args.n_shared,
        k=args.k,
        moe=args.moe,
        latent_dim=args.latent_dim,
        model_dtype=args.model_dtype,
        grad_checkpoint=args.grad_checkpoint,
    )

    data_cfg = dataConfig(
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        T=args.T,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        micro_batch_size=args.micro_batch_size,
    )

    lr_cfg = LRConfig(
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        end_lr=args.end_lr,
        warmup_steps=args.warmup_steps,
        end_steps=args.end_steps,
    )

    device_cfg = deviceConfig(
        n_device_axis=args.n_device_axis,
    )

    cfg = config(
        model=model_cfg,
        data=data_cfg,
        lr=lr_cfg,
        name=args.name,
        output_dir=args.output_dir,
        device_config=device_cfg,
        checkpoint_steps=args.checkpoint_steps,
        checkpoint_manager=args.checkpoint_manager,
        seed=args.seed,
        training_steps=args.training_steps,
        grad_step=args.grad_step,
        inference_batch=args.inference_batch,
        eval_steps=args.eval_steps,
        alpha=args.alpha,
        wandb=args.wandb,
        grad_clip_norm=args.grad_clip_norm,
    )

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    print(json.dumps(cfg.__dict__, indent=4, default=lambda o: o.__dict__))
