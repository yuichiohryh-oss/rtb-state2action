from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.hand_actions import extract_actions, parse_hand_frames, read_text_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract play actions from infer_hand JSONL output."
    )
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True, type=Path)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--cooldown-ms", type=int, default=1000)
    parser.add_argument("--pre-hold", type=int, default=2)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.window <= 0:
        parser.error("--window must be positive")
    if args.confirm_frames <= 0:
        parser.error("--confirm-frames must be positive")
    if args.cooldown_ms < 0:
        parser.error("--cooldown-ms must be non-negative")
    if args.pre_hold < 0:
        parser.error("--pre-hold must be non-negative")
    return args


def main() -> None:
    args = parse_args()

    input_path = None if args.input_path == "-" else Path(args.input_path)
    lines = read_text_lines(input_path, stdin=sys.stdin.buffer)
    frames = parse_hand_frames(lines)
    actions = extract_actions(
        frames,
        window=args.window,
        confirm_frames=args.confirm_frames,
        cooldown_ms=args.cooldown_ms,
        pre_hold=args.pre_hold,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
