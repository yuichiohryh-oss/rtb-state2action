from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from rtb_state2action.data.pos_ds import PosDataset, load_manifest, summarize_manifest
from rtb_state2action.models.pos_cnn import build_pos_cnn


@dataclass
class TrainConfig:
    train_manifest: Path
    val_manifest: Path
    grid_w: int
    grid_h: int
    img_size: int
    batch_size: int
    epochs: int
    lr: float
    out_dir: Path
    num_workers: int = 0
    seed: int | None = None
    width_mult: float = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train position cell classifier from diff images.")
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--grid-w", required=True, type=int)
    parser.add_argument("--grid-h", required=True, type=int)
    parser.add_argument("--img-size", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width-mult", type=float, default=1.0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.train_manifest.exists():
        parser.error(f"--train-manifest not found: {args.train_manifest}")
    if not args.val_manifest.exists():
        parser.error(f"--val-manifest not found: {args.val_manifest}")
    if args.grid_w <= 0 or args.grid_h <= 0:
        parser.error("--grid-w/--grid-h must be positive")
    if args.img_size <= 0:
        parser.error("--img-size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.width_mult <= 0:
        parser.error("--width-mult must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_topk(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = max(1, k)
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(labels.view(-1, 1))
    return float(correct.any(dim=1).float().sum().item())


def manhattan_distance(
    pred: torch.Tensor, labels: torch.Tensor, grid_w: int
) -> torch.Tensor:
    pred_row = pred // grid_w
    pred_col = pred % grid_w
    label_row = labels // grid_w
    label_col = labels % grid_w
    return (pred_row - label_row).abs() + (pred_col - label_col).abs()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    grid_w: int,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top3_sum = 0.0
    dist_sum = 0.0
    within1_sum = 0.0
    within2_sum = 0.0
    within3_sum = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch = labels.size(0)
            loss_sum += loss.item() * batch
            top1_sum += accuracy_topk(logits, labels, k=1)
            top3_sum += accuracy_topk(logits, labels, k=3)
            preds = logits.argmax(dim=1)
            dist = manhattan_distance(preds, labels, grid_w)
            dist_sum += float(dist.sum().item())
            within1_sum += float((dist <= 1).sum().item())
            within2_sum += float((dist <= 2).sum().item())
            within3_sum += float((dist <= 3).sum().item())
            count += batch
    denom = max(1, count)
    return {
        "loss": loss_sum / denom,
        "top1": top1_sum / denom,
        "top3": top3_sum / denom,
        "mean_manhattan": dist_sum / denom,
        "within1": within1_sum / denom,
        "within2": within2_sum / denom,
        "within3": within3_sum / denom,
        "count": float(count),
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def train(config: TrainConfig) -> None:
    if config.seed is not None:
        set_seed(config.seed)

    train_samples = load_manifest(config.train_manifest)
    val_samples = load_manifest(config.val_manifest)
    if not train_samples:
        raise RuntimeError("Train manifest has no samples")
    if not val_samples:
        raise RuntimeError("Val manifest has no samples")

    train_ds = PosDataset(train_samples, img_size=config.img_size)
    val_ds = PosDataset(val_samples, img_size=config.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config.grid_w * config.grid_h
    model = build_pos_cnn(in_ch=1, num_classes=num_classes, width_mult=config.width_mult).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    config.out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(config)
    config_payload["train_manifest"] = str(config.train_manifest)
    config_payload["val_manifest"] = str(config.val_manifest)
    config_payload["out_dir"] = str(config.out_dir)
    config_payload["device"] = str(device)
    config_payload["train_manifest_summary"] = summarize_manifest(train_samples)
    config_payload["val_manifest_summary"] = summarize_manifest(val_samples)
    save_json(config.out_dir / "config.json", config_payload)

    metrics_path = config.out_dir / "metrics.jsonl"
    best_top1 = -1.0
    best_path = config.out_dir / "model.pt"
    train_sample_count = len(train_samples)
    val_sample_count = len(val_samples)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_eval_loader, criterion, device, config.grid_w)
        val_metrics = evaluate(model, val_loader, criterion, device, config.grid_w)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top3": train_metrics["top3"],
            "train_mean_manhattan": train_metrics["mean_manhattan"],
            "train_within1": train_metrics["within1"],
            "train_within2": train_metrics["within2"],
            "train_within3": train_metrics["within3"],
            "train_count": train_metrics["count"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
            "val_mean_manhattan": val_metrics["mean_manhattan"],
            "val_within1": val_metrics["within1"],
            "val_within2": val_metrics["within2"],
            "val_within3": val_metrics["within3"],
            "val_count": val_metrics["count"],
            "train_samples": train_sample_count,
            "val_samples": val_sample_count,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics, ensure_ascii=True) + "\n")

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "grid_w": config.grid_w,
                    "grid_h": config.grid_h,
                    "img_size": config.img_size,
                    "num_classes": num_classes,
                },
                best_path,
            )

    print(f"Saved best model to: {best_path}")


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out,
        num_workers=args.num_workers,
        seed=args.seed,
        width_mult=args.width_mult,
    )
    train(config)


if __name__ == "__main__":
    main()
