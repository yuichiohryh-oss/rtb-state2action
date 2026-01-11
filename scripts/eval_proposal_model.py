from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from src.proposal_model import (
    ProposalModel,
    StateRoleSample,
    load_state_role_samples,
    split_state_role_samples,
)


@dataclass(frozen=True)
class EvalConfig:
    data_path: Path
    model_path: Path | None
    topk: int
    baseline: str
    val_split: float
    seed: int
    split: str
    out_path: Path | None


def _load_model(path: Path) -> ProposalModel:
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    input_dim = int(checkpoint.get("input_dim", 8))
    hidden_dim = int(checkpoint.get("hidden_dim", 32))
    dropout = float(checkpoint.get("dropout", 0.1))
    num_classes = int(checkpoint.get("num_classes", 8))
    model = ProposalModel(
        input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes
    )
    model.load_state_dict(checkpoint["model_state"])
    return model


def _resolve_eval_samples(
    samples: list[StateRoleSample],
    val_split: float,
    seed: int,
    split: str,
    require_split: bool,
) -> tuple[list[StateRoleSample], list[StateRoleSample], list[StateRoleSample]]:
    if split not in {"all", "train", "val"}:
        raise ValueError(f"split must be one of all/train/val, got: {split}")
    if split == "all" and not require_split:
        return samples, samples, samples
    train_samples, val_samples = split_state_role_samples(samples, val_split, seed)
    if split == "all":
        return samples, train_samples, val_samples
    if split == "train":
        return train_samples, train_samples, val_samples
    return val_samples, train_samples, val_samples


def _compute_confusion_matrix(
    true_labels: list[int], pred_labels: list[int], num_classes: int
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_idx, pred_idx in zip(true_labels, pred_labels):
        matrix[true_idx][pred_idx] += 1
    return matrix


def _class_distribution(labels: list[int], num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    for label in labels:
        counts[label] += 1
    return counts


def _compute_f1_scores(confusion: list[list[int]]) -> dict[str, float]:
    num_classes = len(confusion)
    supports = [sum(row) for row in confusion]
    total_support = sum(supports)
    f1_scores: list[float] = []
    weighted_f1_sum = 0.0

    for idx in range(num_classes):
        tp = confusion[idx][idx]
        fp = sum(confusion[row][idx] for row in range(num_classes)) - tp
        fn = sum(confusion[idx]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_scores.append(f1)
        weighted_f1_sum += f1 * supports[idx]

    macro_f1 = sum(f1_scores) / num_classes if num_classes > 0 else 0.0
    weighted_f1 = weighted_f1_sum / total_support if total_support > 0 else 0.0
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1}


def _accuracy_topk(
    true_labels: list[int], pred_topk: list[list[int]], k: int
) -> float:
    if not true_labels:
        return 0.0
    correct = 0
    for true_idx, preds in zip(true_labels, pred_topk):
        if true_idx in preds[:k]:
            correct += 1
    return correct / len(true_labels)


def _predict_model(
    model: ProposalModel,
    samples: list[StateRoleSample],
    device: torch.device,
    topk: int,
) -> tuple[list[int], list[list[int]]]:
    model.eval()
    pred_top1: list[int] = []
    pred_topk: list[list[int]] = []
    if not samples:
        return pred_top1, pred_topk
    expected_dim = model.net[0].in_features
    batch_size = 128
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            features_list = [sample.features() for sample in batch]
            if any(len(row) != expected_dim for row in features_list):
                raise RuntimeError(f"input length mismatch for model input_dim {expected_dim}")
            features = torch.tensor(features_list, dtype=torch.float32).to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            k = min(topk, probs.size(1))
            _, indices = torch.topk(probs, k=k, dim=1)
            indices_list = indices.cpu().tolist()
            for row in indices_list:
                pred_top1.append(row[0])
                pred_topk.append(row)
    return pred_top1, pred_topk


def _most_frequent_class(samples: list[StateRoleSample], num_classes: int) -> int:
    counts = [0 for _ in range(num_classes)]
    for sample in samples:
        counts[sample.role - 1] += 1
    if sum(counts) == 0:
        raise ValueError("cannot compute most frequent class from empty samples")
    return max(range(num_classes), key=lambda idx: (counts[idx], -idx))


def _predict_mostfreq(
    samples: list[StateRoleSample], most_freq: int, num_classes: int, topk: int
) -> tuple[list[int], list[list[int]]]:
    pred_top1 = [most_freq for _ in samples]
    if topk <= 1:
        pred_topk = [[most_freq] for _ in samples]
        return pred_top1, pred_topk
    fallback = [idx for idx in range(num_classes) if idx != most_freq]
    topk_list = [most_freq] + fallback[: max(0, min(topk, num_classes) - 1)]
    pred_topk = [topk_list for _ in samples]
    return pred_top1, pred_topk


def _predict_random(
    samples: list[StateRoleSample], num_classes: int, topk: int, seed: int
) -> tuple[list[int], list[list[int]]]:
    rng = random.Random(seed)
    pred_top1: list[int] = []
    pred_topk: list[list[int]] = []
    k = min(topk, num_classes)
    for _ in samples:
        choices = list(range(num_classes))
        rng.shuffle(choices)
        selected = choices[:k]
        pred_top1.append(selected[0])
        pred_topk.append(selected)
    return pred_top1, pred_topk


def evaluate_samples(
    samples: list[StateRoleSample],
    pred_top1: list[int],
    pred_topk: list[list[int]],
    topk: int,
    num_classes: int,
) -> dict[str, object]:
    true_labels = [sample.role - 1 for sample in samples]
    accuracy_top1 = _accuracy_topk(true_labels, pred_topk, k=1)
    accuracy_topk = _accuracy_topk(true_labels, pred_topk, k=topk)
    confusion = _compute_confusion_matrix(true_labels, pred_top1, num_classes)
    f1_scores = _compute_f1_scores(confusion)
    return {
        "accuracy_top1": accuracy_top1,
        "accuracy_topk": accuracy_topk,
        "macro_f1": f1_scores["macro_f1"],
        "weighted_f1": f1_scores["weighted_f1"],
        "confusion_matrix": confusion,
        "class_distribution": {
            "true": _class_distribution(true_labels, num_classes),
            "pred": _class_distribution(pred_top1, num_classes),
        },
    }


def _resolve_out_path(out_path: Path | None) -> Path | None:
    if out_path is None:
        return None
    if out_path.suffix.lower() == ".json":
        return out_path
    return out_path / "metrics.json"


def _print_summary(
    metrics: dict[str, object],
    total: int,
    split: str,
    baseline: str,
    topk: int,
    model_path: Path | None,
    extra: str | None = None,
) -> None:
    print("eval_proposal_model summary:")
    print(f"split={split} total={total} baseline={baseline}")
    if model_path is not None:
        print(f"model={model_path}")
    if extra:
        print(extra)
    print(f"accuracy_top1={metrics['accuracy_top1']:.4f}")
    print(f"accuracy_top{topk}={metrics['accuracy_topk']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"weighted_f1={metrics['weighted_f1']:.4f}")
    print(f"confusion_matrix={metrics['confusion_matrix']}")
    print(f"class_distribution_true={metrics['class_distribution']['true']}")
    print(f"class_distribution_pred={metrics['class_distribution']['pred']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate proposal model on state_role dataset.")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--baseline", choices=["none", "mostfreq", "random"], default="none")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--out", type=Path)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.data.exists():
        parser.error(f"--data not found: {args.data}")
    if args.baseline == "none":
        if args.model is None:
            parser.error("--model is required when --baseline is none")
        if not args.model.exists():
            parser.error(f"--model not found: {args.model}")
    if args.topk <= 0:
        parser.error("--topk must be positive")
    if args.val_split <= 0 or args.val_split >= 1:
        parser.error("--val-split must be between 0 and 1")
    return args


def eval_proposal_model(config: EvalConfig) -> dict[str, object]:
    samples = load_state_role_samples(config.data_path)
    if not samples:
        raise RuntimeError("No samples found in state_role dataset")

    require_split = config.baseline == "mostfreq" or config.split != "all"
    eval_samples, train_samples, _ = _resolve_eval_samples(
        samples, config.val_split, config.seed, config.split, require_split
    )
    if not eval_samples:
        raise RuntimeError(f"No samples available for split: {config.split}")

    num_classes = 8
    extra = None

    if config.baseline == "mostfreq":
        most_freq = _most_frequent_class(train_samples, num_classes)
        pred_top1, pred_topk = _predict_mostfreq(
            eval_samples, most_freq, num_classes, config.topk
        )
        extra = f"most_frequent_class={most_freq + 1}"
    elif config.baseline == "random":
        pred_top1, pred_topk = _predict_random(
            eval_samples, num_classes, config.topk, config.seed
        )
        extra = f"random_seed={config.seed}"
    else:
        if config.model_path is None:
            raise RuntimeError("model_path is required for baseline=none")
        try:
            model = _load_model(config.model_path)
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            raise RuntimeError(str(exc)) from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        pred_top1, pred_topk = _predict_model(model, eval_samples, device, config.topk)

    metrics = evaluate_samples(
        eval_samples,
        pred_top1,
        pred_topk,
        config.topk,
        num_classes,
    )

    _print_summary(
        metrics,
        total=len(eval_samples),
        split=config.split,
        baseline=config.baseline,
        topk=config.topk,
        model_path=config.model_path,
        extra=extra,
    )

    out_path = _resolve_out_path(config.out_path)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": config.split,
            "baseline": config.baseline,
            "topk": config.topk,
            "seed": config.seed,
            "val_split": config.val_split,
            "total": len(eval_samples),
            "metrics": metrics,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {out_path}")

    return metrics


def main() -> None:
    args = parse_args()
    config = EvalConfig(
        data_path=args.data,
        model_path=args.model,
        topk=args.topk,
        baseline=args.baseline,
        val_split=args.val_split,
        seed=args.seed,
        split=args.split,
        out_path=args.out,
    )
    eval_proposal_model(config)


if __name__ == "__main__":
    main()
