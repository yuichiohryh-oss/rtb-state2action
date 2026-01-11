from __future__ import annotations

import argparse
from pathlib import Path

from src.hand.labeler import LabelerConfig, label_crops


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label hand crop images.")
    parser.add_argument("--crops-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/hand_labels.jsonl"))
    parser.add_argument(
        "--show-help",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help="Show key help overlay (default: true).",
    )
    parser.add_argument("--no-help", dest="show_help", action="store_false", help="Hide key help overlay.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    config = LabelerConfig(
        crops_jsonl=args.crops_jsonl,
        output_jsonl=args.output_jsonl,
        show_help=args.show_help,
    )
    label_crops(config)


if __name__ == "__main__":
    main()
