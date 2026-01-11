from __future__ import annotations

import argparse
from pathlib import Path

from src.hand.train import TrainConfig, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hand CNN classifier.")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    config = TrainConfig(
        labels_jsonl=args.labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
    run_dir = train_model(config)
    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()
