from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.hand_actions import read_text_lines


@dataclass(frozen=True)
class StateRoleSample:
    in_hand: list[float]
    role: int  # 1..8
    prev_action_onehot: list[float] | None = None
    prev2_action_onehot: list[float] | None = None

    def features(self) -> list[float]:
        if self.prev_action_onehot is None:
            return self.in_hand
        if self.prev2_action_onehot is None:
            return self.in_hand + self.prev_action_onehot
        return self.in_hand + self.prev_action_onehot + self.prev2_action_onehot


def _validate_in_hand_state(values: list[int], source: str, line_no: int) -> list[float]:
    if len(values) != 8:
        raise ValueError(f"in_hand_state must have length 8 at {source}:{line_no}")
    floats: list[float] = []
    for idx, value in enumerate(values, start=1):
        if not isinstance(value, int):
            raise ValueError(f"in_hand_state[{idx}] must be int at {source}:{line_no}")
        if value not in (0, 1):
            raise ValueError(f"in_hand_state[{idx}] must be 0 or 1 at {source}:{line_no}")
        floats.append(float(value))
    return floats


def _validate_role(role: int, source: str, line_no: int) -> int:
    if not isinstance(role, int):
        raise ValueError(f"role must be int at {source}:{line_no}")
    if role < 1 or role > 8:
        raise ValueError(f"role must be 1..8 at {source}:{line_no}")
    return role


def _validate_action_id(action_id: int, name: str, source: str, line_no: int) -> int:
    if not isinstance(action_id, int):
        raise ValueError(f"{name} must be int at {source}:{line_no}")
    if action_id < 0 or action_id > 8:
        raise ValueError(f"{name} must be 0..8 at {source}:{line_no}")
    return action_id


def _action_onehot(action_id: int) -> list[float]:
    values = [0.0 for _ in range(8)]
    if action_id > 0:
        values[action_id - 1] = 1.0
    return values


def _validate_action_onehot(
    values: list[int],
    name: str,
    source: str,
    line_no: int,
) -> list[float]:
    if len(values) != 8:
        raise ValueError(f"{name} must have length 8 at {source}:{line_no}")
    floats: list[float] = []
    for idx, value in enumerate(values, start=1):
        if not isinstance(value, int):
            raise ValueError(f"{name}[{idx}] must be int at {source}:{line_no}")
        if value not in (0, 1):
            raise ValueError(f"{name}[{idx}] must be 0 or 1 at {source}:{line_no}")
        floats.append(float(value))
    return floats


def parse_state_role_lines(lines: Iterable[str], source: str) -> list[StateRoleSample]:
    samples: list[StateRoleSample] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip().lstrip("\ufeff")
        if not stripped:
            continue
        if not stripped.startswith("{"):
            raise ValueError(f"unexpected non-JSON line {idx} in {source}: {stripped[:120]}")
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"failed to parse JSON at {source}:{idx}") from exc
        if "in_hand_state" not in payload or "role" not in payload:
            raise ValueError(f"missing keys at {source}:{idx}")
        in_hand = payload["in_hand_state"]
        role = payload["role"]
        if not isinstance(in_hand, list):
            raise ValueError(f"in_hand_state must be list at {source}:{idx}")
        in_hand_floats = _validate_in_hand_state(in_hand, source, idx)
        role_id = _validate_role(role, source, idx)
        prev_onehot: list[float] | None = None
        prev2_onehot: list[float] | None = None
        if "prev_action_onehot" in payload:
            prev_raw = payload["prev_action_onehot"]
            if not isinstance(prev_raw, list):
                raise ValueError(f"prev_action_onehot must be list at {source}:{idx}")
            prev_onehot = _validate_action_onehot(prev_raw, "prev_action_onehot", source, idx)
        if "prev_action" in payload:
            prev_action = _validate_action_id(payload["prev_action"], "prev_action", source, idx)
            if prev_onehot is None:
                prev_onehot = _action_onehot(prev_action)
            else:
                expected = _action_onehot(prev_action)
                if prev_onehot != expected:
                    raise ValueError(f"prev_action_onehot mismatch at {source}:{idx}")
        if "prev2_action_onehot" in payload:
            prev2_raw = payload["prev2_action_onehot"]
            if not isinstance(prev2_raw, list):
                raise ValueError(f"prev2_action_onehot must be list at {source}:{idx}")
            prev2_onehot = _validate_action_onehot(prev2_raw, "prev2_action_onehot", source, idx)
        if "prev2_action" in payload:
            prev2_action = _validate_action_id(payload["prev2_action"], "prev2_action", source, idx)
            if prev2_onehot is None:
                prev2_onehot = _action_onehot(prev2_action)
            else:
                expected = _action_onehot(prev2_action)
                if prev2_onehot != expected:
                    raise ValueError(f"prev2_action_onehot mismatch at {source}:{idx}")
        if prev2_onehot is not None and prev_onehot is None:
            raise ValueError(f"prev2_action_onehot requires prev_action_onehot at {source}:{idx}")
        samples.append(
            StateRoleSample(
                in_hand=in_hand_floats,
                role=role_id,
                prev_action_onehot=prev_onehot,
                prev2_action_onehot=prev2_onehot,
            )
        )
    return samples


def load_state_role_samples(path: Path) -> list[StateRoleSample]:
    if not path.exists():
        raise FileNotFoundError(f"state_role file not found: {path}")
    lines = read_text_lines(path)
    return parse_state_role_lines(lines, str(path))


class ProposalDataset(Dataset):
    def __init__(self, samples: Sequence[StateRoleSample]):
        self.samples = list(samples)
        self.input_dim = resolve_input_dim(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        features = torch.tensor(sample.features(), dtype=torch.float32)
        label = sample.role - 1
        return features, label


class ProposalModel(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, dropout: float = 0.1, num_classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    seed: int = 42
    val_split: float = 0.2
    run_root: Path = Path("runs")
    hidden_dim: int = 32
    dropout: float = 0.1
    save_best: bool = True
    metric: str = "val_acc"
    early_stopping_patience: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: int = 1) -> float:
    max_k = max(1, topk)
    _, pred = logits.topk(max_k, dim=1)
    correct = pred.eq(labels.view(-1, 1))
    correct_topk = correct.any(dim=1).float().sum().item()
    return correct_topk / labels.size(0)


def _split_samples(
    samples: list[StateRoleSample], val_split: float, seed: int
) -> tuple[list[StateRoleSample], list[StateRoleSample]]:
    if len(samples) < 2:
        raise RuntimeError("Need at least 2 samples for train/val split")
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(len(samples) * val_split))
    if len(samples) - val_size < 1:
        val_size = len(samples) - 1
    val_indices = set(indices[:val_size])
    train_samples: list[StateRoleSample] = []
    val_samples: list[StateRoleSample] = []
    for idx, sample in enumerate(samples):
        if idx in val_indices:
            val_samples.append(sample)
        else:
            train_samples.append(sample)
    return train_samples, val_samples


def resolve_input_dim(samples: Sequence[StateRoleSample]) -> int:
    if not samples:
        return 0
    expected = len(samples[0].features())
    for idx, sample in enumerate(samples, start=1):
        if len(sample.features()) != expected:
            raise ValueError(f"inconsistent feature length at sample {idx}")
    return expected


def split_state_role_samples(
    samples: list[StateRoleSample], val_split: float, seed: int
) -> tuple[list[StateRoleSample], list[StateRoleSample]]:
    return _split_samples(samples, val_split, seed)


def _run_epoch(
    model: ProposalModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    total_loss = 0.0
    total = 0
    acc1_sum = 0.0
    acc2_sum = 0.0
    acc3_sum = 0.0

    if is_train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_train):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            acc1_sum += _accuracy(logits, labels, topk=1) * batch_size
            acc2_sum += _accuracy(logits, labels, topk=2) * batch_size
            acc3_sum += _accuracy(logits, labels, topk=3) * batch_size

    denom = max(1, total)
    return {
        "loss": total_loss / denom,
        "accuracy": acc1_sum / denom,
        "top2_accuracy": acc2_sum / denom,
        "top3_accuracy": acc3_sum / denom,
    }


def _checkpoint_payload(model: ProposalModel, input_dim: int, config: TrainConfig) -> dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "num_classes": 8,
    }


def train_proposal_model(config: TrainConfig) -> Path:
    samples = load_state_role_samples(config.data_path)
    if not samples:
        raise RuntimeError("No samples found in state_role dataset")
    if config.metric not in ("val_acc", "val_top3"):
        raise ValueError("metric must be 'val_acc' or 'val_top3'")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0")

    seed_everything(config.seed)
    input_dim = resolve_input_dim(samples)

    train_samples, val_samples = _split_samples(samples, config.val_split, config.seed)
    train_loader = DataLoader(
        ProposalDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ProposalDataset(val_samples),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProposalModel(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    print(
        "train_proposal_model start:",
        f"batch_size={config.batch_size}",
        f"epochs={config.epochs}",
        f"lr={config.lr}",
        f"val_split={config.val_split}",
        f"seed={config.seed}",
        sep=" ",
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = config.run_root / f"{run_id}_proposal"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("-inf")
    best_epoch = 0
    best_val_acc = 0.0
    best_val_top3 = 0.0
    epochs_since_improve = 0
    metric_key = "accuracy" if config.metric == "val_acc" else "top3_accuracy"

    train_metrics = {}
    val_metrics = {}
    for epoch in range(config.epochs):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device)
        print(
            f"epoch {epoch + 1}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_top2={val_metrics['top2_accuracy']:.4f} "
            f"val_top3={val_metrics['top3_accuracy']:.4f}"
        )

        current_metric = val_metrics[metric_key]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_val_acc = val_metrics["accuracy"]
            best_val_top3 = val_metrics["top3_accuracy"]
            epochs_since_improve = 0
            if config.save_best:
                best_model_path = run_dir / "best_model.pt"
                torch.save(_checkpoint_payload(model, input_dim, config), best_model_path)
            print(
                "best updated:",
                f"epoch={epoch + 1}",
                f"val_acc={val_metrics['accuracy']:.4f}",
                f"val_top3={val_metrics['top3_accuracy']:.4f}",
            )
        else:
            epochs_since_improve += 1
            if config.early_stopping_patience > 0 and epochs_since_improve >= config.early_stopping_patience:
                print(
                    f"early stopping: no improvement for {config.early_stopping_patience} epochs"
                )
                break

    model_path = run_dir / "model.pt"
    torch.save(
        _checkpoint_payload(model, input_dim, config),
        model_path,
    )

    config_path = run_dir / "config.json"
    config_payload = {
        "data_path": str(config.data_path),
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "lr": config.lr,
        "seed": config.seed,
        "val_split": config.val_split,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "save_best": config.save_best,
        "metric": config.metric,
        "early_stopping_patience": config.early_stopping_patience,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.json"
    metrics_payload = {
        "train": train_metrics,
        "val": val_metrics,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "final_train_acc": train_metrics.get("accuracy", 0.0),
        "final_val_acc": val_metrics.get("accuracy", 0.0),
        "final_val_top3": val_metrics.get("top3_accuracy", 0.0),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_top3": best_val_top3,
        "best_metric_name": config.metric,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return run_dir
