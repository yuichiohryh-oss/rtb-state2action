from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from rtb_state2action.data.pos_ds import load_grayscale_image
from rtb_state2action.models.pos_cnn import build_pos_cnn


class InferSample:
    def __init__(self, record: dict[str, Any], img_path: Path, label: int, index: int):
        self.record = record
        self.img_path = img_path
        self.label = label
        self.index = index


class InferDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[InferSample], img_size: int):
        self.samples = samples
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = load_grayscale_image(str(sample.img_path), self.img_size)
        image = image.astype("float32") / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(sample.label, dtype=torch.long)
        index = torch.tensor(sample.index, dtype=torch.long)
        return tensor, label, index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer position cell classifier on a manifest.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--grid-w", required=True, type=int)
    parser.add_argument("--grid-h", required=True, type=int)
    parser.add_argument("--img-size", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--width-mult", type=float, default=1.0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.model.exists():
        parser.error(f"--model not found: {args.model}")
    if not args.manifest.exists():
        parser.error(f"--manifest not found: {args.manifest}")
    if args.grid_w <= 0 or args.grid_h <= 0:
        parser.error("--grid-w/--grid-h must be positive")
    if args.img_size <= 0:
        parser.error("--img-size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.topk <= 0:
        parser.error("--topk must be positive")
    if args.width_mult <= 0:
        parser.error("--width-mult must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")
    return args


def resolve_image_path(raw: Any, manifest_path: Path) -> Path | None:
    if raw is None:
        return None
    try:
        raw_str = str(raw)
    except (TypeError, ValueError):
        return None
    path = Path(raw_str)
    if path.is_absolute():
        return path if path.exists() else None
    repo_root = Path(__file__).resolve().parent
    candidates = [repo_root / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_manifest(
    path: Path, grid_w: int, grid_h: int
) -> tuple[list[InferSample], int]:
    samples: list[InferSample] = []
    skipped = 0
    num_classes = grid_w * grid_h
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse manifest JSON at line {line_no} in {path}"
                ) from exc
            label = record.get("label")
            if label is None:
                skipped += 1
                continue
            try:
                label = int(label)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if label < 0 or label >= num_classes:
                skipped += 1
                continue
            img_path = resolve_image_path(record.get("img"), path)
            if img_path is None:
                skipped += 1
                continue
            samples.append(InferSample(record=record, img_path=img_path, label=label, index=len(samples)))
    return samples, skipped


def manhattan_distance(
    pred: torch.Tensor, labels: torch.Tensor, grid_w: int
) -> torch.Tensor:
    pred_row = pred // grid_w
    pred_col = pred % grid_w
    label_row = labels // grid_w
    label_col = labels % grid_w
    return (pred_row - label_row).abs() + (pred_col - label_col).abs()


def load_model(
    model_path: Path, device: torch.device, num_classes: int, width_mult: float
) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state") if isinstance(checkpoint, dict) else checkpoint
    if state is None:
        raise RuntimeError(f"Invalid checkpoint format: {model_path}")
    model = build_pos_cnn(in_ch=1, num_classes=num_classes, width_mult=width_mult).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def infer() -> None:
    args = parse_args()
    samples, skipped = read_manifest(args.manifest, args.grid_w, args.grid_h)
    num_classes = args.grid_w * args.grid_h
    if not samples:
        write_jsonl(args.out, [])
        print("n=0")
        print("top1_acc=0.0")
        if args.topk >= 3:
            print("top3_acc=0.0")
        print("mean_manhattan=0.0")
        print("within1=0.0")
        print("within2=0.0")
        print("within3=0.0")
        print(f"skipped={skipped}")
        return

    device = torch.device(args.device)
    model = load_model(args.model, device, num_classes, args.width_mult)
    dataset = InferDataset(samples, img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    k = min(max(1, args.topk), num_classes)
    k3 = min(3, num_classes)
    preds: list[dict[str, Any] | None] = [None] * len(samples)
    top1_correct = 0.0
    top3_correct = 0.0
    dist_sum = 0.0
    within1_sum = 0.0
    within2_sum = 0.0
    within3_sum = 0.0
    count = 0

    with torch.no_grad():
        for images, labels, indices in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            topk_prob, topk_idx = probs.topk(k, dim=1)
            top1 = topk_idx[:, 0]
            batch = labels.size(0)
            count += batch
            top1_correct += float((top1 == labels).sum().item())
            if args.topk >= 3:
                _, top3_idx = logits.topk(k3, dim=1)
                top3_correct += float(top3_idx.eq(labels.view(-1, 1)).any(dim=1).sum().item())
            dist = manhattan_distance(top1, labels, args.grid_w)
            dist_sum += float(dist.sum().item())
            within1_sum += float((dist <= 1).sum().item())
            within2_sum += float((dist <= 2).sum().item())
            within3_sum += float((dist <= 3).sum().item())

            for i in range(batch):
                idx = int(indices[i].item())
                preds[idx] = {
                    "top1": int(top1[i].item()),
                    "topk": [int(v) for v in topk_idx[i].tolist()],
                    "topk_prob": [float(v) for v in topk_prob[i].tolist()],
                }

    out_rows: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample.record)
        pred = preds[sample.index]
        if pred is None:
            raise RuntimeError(f"Missing prediction for sample index {sample.index}")
        row["pred"] = pred
        out_rows.append(row)

    write_jsonl(args.out, out_rows)

    denom = max(1, count)
    print(f"n={count}")
    print(f"top1_acc={top1_correct / denom}")
    if args.topk >= 3:
        print(f"top3_acc={top3_correct / denom}")
    print(f"mean_manhattan={dist_sum / denom}")
    print(f"within1={within1_sum / denom}")
    print(f"within2={within2_sum / denom}")
    print(f"within3={within3_sum / denom}")
    print(f"skipped={skipped}")


def main() -> None:
    infer()


if __name__ == "__main__":
    main()
