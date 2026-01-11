from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.proposal_model import (
    ProposalModel,
    StateRoleSample,
    load_state_role_samples,
)


def _parse_hand_arg(hand_arg: str) -> list[float]:
    items = [item.strip() for item in hand_arg.split(",") if item.strip() != ""]
    if len(items) != 8:
        raise ValueError("hand must have 8 comma-separated values")
    values: list[float] = []
    for idx, item in enumerate(items, start=1):
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"hand[{idx}] must be int") from exc
        if value not in (0, 1):
            raise ValueError(f"hand[{idx}] must be 0 or 1")
        values.append(float(value))
    return values


def _load_model(path: Path) -> ProposalModel:
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    hidden_dim = int(checkpoint.get("hidden_dim", 32))
    dropout = float(checkpoint.get("dropout", 0.1))
    num_classes = int(checkpoint.get("num_classes", 8))
    model = ProposalModel(hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    return model


def _topk_predictions(
    model: ProposalModel, hand: list[float], device: torch.device, k: int
) -> list[tuple[int, float]]:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(hand, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        k = min(k, probs.size(1))
        values, indices = torch.topk(probs, k=k, dim=1)
    results: list[tuple[int, float]] = []
    for score, idx in zip(values[0].tolist(), indices[0].tolist()):
        results.append((idx + 1, float(score)))
    return results


def _print_predictions(prefix: str, preds: list[tuple[int, float]]) -> None:
    formatted = ", ".join([f"role={role} prob={prob:.3f}" for role, prob in preds])
    print(f"{prefix}{formatted}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer proposal model on a hand state.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=3)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hand", type=str)
    group.add_argument("--from-state-role", type=Path)
    parser.add_argument("--n", type=int, default=20)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.model.exists():
        parser.error(f"--model not found: {args.model}")
    if args.topk <= 0:
        parser.error("--topk must be positive")
    if args.hand is None and args.from_state_role is None:
        parser.error("--hand or --from-state-role is required")
    if args.from_state_role is not None and not args.from_state_role.exists():
        parser.error(f"--from-state-role not found: {args.from_state_role}")
    if args.n <= 0:
        parser.error("--n must be positive")
    return args


def _infer_single(model: ProposalModel, hand: list[float], device: torch.device, topk: int) -> None:
    preds = _topk_predictions(model, hand, device, topk)
    _print_predictions("topk: ", preds)


def _infer_dataset(
    model: ProposalModel, samples: list[StateRoleSample], device: torch.device, topk: int
) -> None:
    for idx, sample in enumerate(samples, start=1):
        preds = _topk_predictions(model, sample.in_hand, device, topk)
        prefix = f"idx={idx} true_role={sample.role} topk: "
        _print_predictions(prefix, preds)


def main() -> None:
    args = parse_args()
    try:
        model = _load_model(args.model)
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.hand is not None:
        try:
            hand = _parse_hand_arg(args.hand)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        _infer_single(model, hand, device, args.topk)
        return

    try:
        samples = load_state_role_samples(args.from_state_role)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    _infer_dataset(model, samples[: args.n], device, args.topk)


if __name__ == "__main__":
    main()
