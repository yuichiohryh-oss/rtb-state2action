from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.hand.model import build_model


class HandDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


class GaussianNoise:
    def __init__(self, std: float = 0.03):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


@dataclass
class TrainConfig:
    labels_jsonl: Path
    image_size: int = 96
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-3
    seed: int = 42
    num_classes: int = 8
    run_root: Path = Path("runs")


def _load_labels(labels_jsonl: Path, class_names: List[str]) -> List[Tuple[str, int]]:
    label_map: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}
    samples: List[Tuple[str, int]] = []
    with labels_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            record = json.loads(line)
            label_name = record["label"]
            if label_name not in label_map:
                continue
            samples.append((record["path"], label_map[label_name]))
    return samples


def _session_id_from_path(path: str) -> str:
    parts = Path(path).parts
    try:
        idx = parts.index("hand_crops")
        return parts[idx + 1]
    except ValueError:
        return "unknown"
    except IndexError:
        return "unknown"


def _split_by_session(samples: List[Tuple[str, int]], seed: int = 42) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    sessions: Dict[str, List[Tuple[str, int]]] = {}
    for sample in samples:
        session_id = _session_id_from_path(sample[0])
        sessions.setdefault(session_id, []).append(sample)
    session_ids = list(sessions.keys())
    random.Random(seed).shuffle(session_ids)
    split = max(1, int(len(session_ids) * 0.8))
    train_ids = set(session_ids[:split])
    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []
    for session_id, items in sessions.items():
        if session_id in train_ids:
            train_samples.extend(items)
        else:
            val_samples.extend(items)
    return train_samples, val_samples


def _make_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.05, saturation=0.2),
            transforms.RandomAffine(degrees=2, translate=(0.0, 0.08), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            GaussianNoise(std=0.02),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return train_tf, val_tf


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: int = 1) -> float:
    max_k = max(1, topk)
    _, pred = logits.topk(max_k, dim=1)
    correct = pred.eq(labels.view(-1, 1))
    correct_topk = correct.any(dim=1).float().sum().item()
    return correct_topk / labels.size(0)


def _confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for p, t in zip(preds, labels):
        matrix[t][p] += 1
    return matrix


def train_model(config: TrainConfig) -> Path:
    class_names = [f"CARD_{i}" for i in range(1, config.num_classes + 1)]
    samples = _load_labels(config.labels_jsonl, class_names)
    if not samples:
        raise RuntimeError("No labeled samples found")

    print(
        "train_hand_cnn start:",
        f"image_size={config.image_size}",
        f"batch_size={config.batch_size}",
        f"epochs={config.epochs}",
        f"lr={config.lr}",
        "augment=on",
        "split=session_80_20",
        sep=" ",
    )

    train_samples, val_samples = _split_by_session(samples, seed=config.seed)
    train_tf, val_tf = _make_transforms(config.image_size)

    train_ds = HandDataset(train_samples, train_tf)
    val_ds = HandDataset(val_samples, val_tf)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=config.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    acc_sum = 0.0
    top3_sum = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            batch_size = labels.size(0)
            acc_sum += _accuracy(logits, labels, topk=1) * batch_size
            top3_sum += _accuracy(logits, labels, topk=3) * batch_size
            count += batch_size
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = {
        "accuracy": acc_sum / max(1, count),
        "top3_accuracy": top3_sum / max(1, count),
        "confusion_matrix": _confusion_matrix(np.array(all_preds), np.array(all_labels), config.num_classes),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
    }

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = config.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.pt"
    torch.save({"model_state": model.state_dict(), "class_names": class_names}, model_path)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return run_dir
