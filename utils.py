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
    vocab_size: int
    n_head: int
    blocks: int
    layers_per_block: int
    T: int
    latent_dim: int
    dhR: int
    dropout_rate: float = 0.1
    model_dtype: str = "bfloat16"
    moe: bool = False
    k: int = 0
    n_experts: int = 0
    n_shared: int = 0
    capacity_factor: float = 1.0


@dataclass
class dataConfig:
    bucket_name: str
    process_path: str = "./bucket_downloads/processShard"
    train_folder_name: str = "train"
    val_folder_name: str = "val"
    T: int = 6
    train_batch_size: int = 3
    val_batch_size: int = 3
    micro_batch_size: int = 1


@dataclass
class LRConfig:
    """class for keeping track of learning rate args"""

    max_lr: float = 6e-4
    min_lr: float = 0
    end_lr: float = 6e-5
    warmup_steps: int = 1000
    end_steps: int = 6000


@dataclass
class deviceConfig:
    """class for distrbuted config"""

    n_device_axis: List[int]


@dataclass
class config:
    """class for keeping track of model args"""

    model_config: modelConfig
    data_config: dataConfig
    lr: LRConfig
    training_steps: int
    name: str
    device_config: deviceConfig
    grad_step: int = 1
    alpha: float = 0.001
    output_dir: str = "./results/"
    checkpoint_steps: int = 10
    inference_batch: int = 1
    eval_steps: int = 25
    seed: int = 0
    wandb: bool = True
    grad_clip_norm: float = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="model training")

    parser.add_argument("--model_dimension", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--layers_per_block", type=int, default=3)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dhR", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--n_shared", type=int, default=2)
    parser.add_argument("--capacity_factor", type=float, default=1.5)

    parser.add_argument(
        "--bucket_name", type=str, default="10bt_gpt2",
    )
    parser.add_argument(
        "--process_path", type=str, default="./bucket_downloads/processShard"
    )
    parser.add_argument("--train_folder_name", type=str, default="train")
    parser.add_argument("--val_folder_name", type=str, default="val")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=4)

    parser.add_argument("--max_lr", type=float, default=4e-3)
    parser.add_argument("--min_lr", type=float, default=0)
    parser.add_argument("--end_lr", type=float, default=4e-5)
    parser.add_argument("--warmup_steps", type=int, default=715)
    parser.add_argument("--end_steps", type=int, default=15000)

    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--name", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/")
    parser.add_argument("--checkpoint_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")

    parser.add_argument("--training_steps", type=int, default=20000)
    parser.add_argument("--grad_step", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--inference_batch", type=int, default=1)
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
        vocab_size=args.vocab_size,
        n_head=args.n_head,
        blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        T=args.T,
        latent_dim=args.latent_dim,
        dhR=args.dhR,
        dropout_rate=args.dropout_rate,
        model_dtype=args.model_dtype,
        moe=args.moe,
        k=args.k,
        n_experts=args.n_experts,
        n_shared=args.n_shared,
        capacity_factor=args.capacity_factor,
    )

    data_cfg = dataConfig(
        bucket_name=args.bucket_name,
        process_path=args.process_path,
        train_folder_name=args.train_folder_name,
        val_folder_name=args.val_folder_name,
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
        model_config=model_cfg,
        data_config=data_cfg,
        lr=lr_cfg,
        name=args.name,
        output_dir=args.output_dir,
        device_config=device_cfg,
        checkpoint_steps=args.checkpoint_steps,
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
