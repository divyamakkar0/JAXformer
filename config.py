import argparse
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class modelConfig:
    """model config class"""
    model_dimension: int
    n_heads: int
    T: int
    dhR: int
    rope_ratio: int
    vocab_size: int
    dropout: float
    blocks: int
    n_experts: int
    k: int
    moe: bool
    latent_dim: int

@dataclass
class dataConfig:
    dataset_path: str = "./tokens.npy"
    val_spilt: float = 0.1
    T: int = 6
    batch_size: int = 3
    shuffle: bool = True

@dataclass
class LRConfig:
    """class for keeping track of learning rate args"""
    max_lr: float = 4e-3
    min_lr: float = 0
    end_lr: float = 4e-4
    warmup_steps: int = 1000
    end_steps: int = 6000

@dataclass
class config:
    """class for keeping track of model args"""
    model: modelConfig
    data: dataConfig
    lr: LRConfig
    training_steps: int
    grad_step: int = 1
    project: str = "jaxformer"
    description: str = "transformer in jax"
    tags: Optional[List[str]] = None
    name: str = None
    output_dir: str = "./results/"
    checkpoint_steps: int = 10
    checkpoint_manager: str = "./checkpoints/manager/"
    seed: int = 0

    def __repr__(self):
        return f"""Configuration:
      Model:
        {self.model}
      Data:
        {self.data}
      Learning Rate:
        {self.lr}
      Training:
        training_steps: {self.training_steps}
        seed: {self.seed}
      Checkpointing:
        checkpoint_steps: {self.checkpoint_steps}
        checkpoint_manager: {self.checkpoint_manager}
      Output:
        output_dir: {self.output_dir}
      Project:
        project: {self.project}
        name: {self.name}
        description: {self.description}
        tags: {self.tags}
    """


def parse_args():

    parser = argparse.ArgumentParser(description="model training")
    parser.add_argument("--model_dimension", type=int, default=24)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--dhR", type=int, default=64)
    parser.add_argument("--rope_ratio", type=int, default=10000)
    parser.add_argument("--vocab_size", type=int, default=100277)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--moe", action='store_true')
    parser.add_argument("--latent_dim", type=int, default=64)

    parser.add_argument("--dataset", type=str, default="./tokens.npy")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_spilt", type=float, default=0.1)

    parser.add_argument("--max_lr", type=float, default=4e-3)
    parser.add_argument("--min_lr", type=float, default=0)
    parser.add_argument("--end_lr", type=float, default=4e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--end_steps", type=int, default=6000)

    parser.add_argument("--project", type=str, default="jaxformer")
    parser.add_argument("--description", type=str, default="transformer in jax")
    parser.add_argument("--tags", nargs='*', type=str, default=None)
    parser.add_argument("--name", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/")
    parser.add_argument("--checkpoint_steps", type=int, default=25)
    parser.add_argument("--checkpoint_manager", type=str, default="./checkpoints/manager/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training_steps", type=int, default=1000)
    parser.add_argument("--grad_step", type=int, default=1)
    args = parser.parse_args()

    model_cfg = modelConfig(
        model_dimension=args.model_dimension,
        n_heads=args.n_heads,
        T=args.T,
        dhR=args.dhR,
        rope_ratio=args.rope_ratio,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
        blocks=args.blocks,
        n_experts=args.n_experts,
        k=args.k,
        moe=args.moe,
        latent_dim=args.latent_dim
    )

    data_cfg = dataConfig(
        dataset_path=args.dataset,
        T=args.T,
        batch_size=args.batch_size,
        val_spilt=args.val_spilt
    )

    lr_cfg = LRConfig(
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        end_lr=args.end_lr,
        warmup_steps=args.warmup_steps,
        end_steps=args.end_steps
    )

    cfg = config(
        model=model_cfg,
        data=data_cfg,
        lr=lr_cfg,
        project=args.project,
        description=args.description,
        tags=args.tags,
        name=args.name,
        output_dir=args.output_dir,
        checkpoint_steps=args.checkpoint_steps,
        checkpoint_manager=args.checkpoint_manager,
        seed=args.seed,
        training_steps=args.training_steps,
        grad_step=args.grad_step
    )

    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    print(cfg)