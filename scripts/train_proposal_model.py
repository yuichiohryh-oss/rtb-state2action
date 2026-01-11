from __future__ import annotations

import argparse
from pathlib import Path

from src.proposal_model import TrainConfig, train_proposal_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train proposal model from state_role dataset.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.data.exists():
        parser.error(f"--data not found: {args.data}")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.val_split <= 0 or args.val_split >= 1:
        parser.error("--val-split must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        val_split=args.val_split,
    )
    run_dir = train_proposal_model(config)
    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()
