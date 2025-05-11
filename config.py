import argparse
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class config:
    """class for keeping track of model args"""
    model_dimension: int
    n_heads: int
    T: int 
    vocab_size: int
    dropout: float
    blocks: int
    n_experts: int
    k: int
    moe: bool
    latent_dim: int
    dhR: int
    batch_size: int
    rope_ratio : int
    max_lr: float = 4e-3
    min_lr: float = 0
    end_lr: float = 4e-4
    warmup_steps: int = 1000
    end_steps: int = 6000
    max_lr: float = 4e-3
    min_lr: float = 0
    end_lr: float = 4e-4
    warmup_steps: int = 1000
    end_steps: int = 6000
    project: str = "jaxformer"
    description: str = "transformer in jax"
    tags: Optional[List[str]] = None
    name: str = None
    output_dir: str = "./results/"
    checkpoint_steps: int = 10
    seed: int = 0

def parse_args():
    parser = argparse.ArgumentParser(description="model traning")
    parser.add_argument("-model_dimension", dest="model_dimension", type=int, required=False)
    parser.add_argument("-n_heads", dest="n_heads", type=int, required=False)
    parser.add_argument("-T", dest="T", type=int, required=False)
    parser.add_argument("-vocab_size", dest="vocab_size", type=int, required=False)
    parser.add_argument("-dropout", dest="dropout", type=float, required=False)
    parser.add_argument("-blocks", dest="blocks", type=int, required=False)
    parser.add_argument("-n_experts", dest="n_experts", type=int, required=False)
    parser.add_argument("-k", dest="k", type=int, required=False)
    parser.add_argument("-moe", dest="moe", action='store_true', required=False)
    parser.add_argument("-latent_dim", dest="latent_dim", type=int, required=False)
    parser.add_argument("-dhR", dest="dhR", type=int, required=False)
    parser.add_argument("-batch_size", dest="batch_size", type=int, required=False)
    parser.add_argument("-rope_ratio", dest="rope_ratio", type=int, required=False)
    parser.add_argument("-max_lr", dest="max_lr", type=float, required=False, default=4e-3)
    parser.add_argument("-min_lr", dest="min_lr", type=float, required=False, default=0)
    parser.add_argument("-end_lr", dest="end_lr", type=float, required=False, default=4e-4)
    parser.add_argument("-warmup_steps", dest="warmup_steps", type=int, required=False, default=1000)
    parser.add_argument("-end_steps", dest="end_steps", type=int, required=False, default=6000)
    parser.add_argument("-project", dest="project", type=str, required=False, default="jaxformer")
    parser.add_argument("-description", dest="description", type=str, required=False, default="transformer in jax")
    parser.add_argument("-tags", dest="tags", nargs='*', type=str, required=False, default=None)
    parser.add_argument("-name", dest="name", type=str, required=False, default=None)
    parser.add_argument("-output_dir", dest="output_dir", type=str, required=False, default="./results/")
    parser.add_argument("-checkpoint_steps", dest="checkpoint_steps", type=int, required=False, default=10)
    parser.add_argument("-seed", dest="seed", type=int, required=False, default=0)

    args = parser.parse_args()

    cfg = config(
        model_dimension=args.model_dimension,
        n_heads=args.n_heads,
        T=args.T,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
        blocks=args.blocks,
        n_experts=args.n_experts,
        k=args.k,
        moe=args.moe,
        latent_dim=args.latent_dim,
        dhR=args.dhR,
        batch_size=args.batch_size,
        rope_ratio=args.rope_ratio,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        end_lr=args.end_lr,
        warmup_steps=args.warmup_steps,
        end_steps=args.end_steps,
        project=args.project,
        description=args.description,
        tags=args.tags,
        name=args.name,
        output_dir=args.output_dir,
        checkpoint_steps=args.checkpoint_steps,
        seed=args.seed
    )

    return cfg
