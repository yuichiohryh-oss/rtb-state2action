from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.hand_actions import parse_hand_frames, read_text_lines
from src.state_role_dataset import build_state_role_records, parse_actions_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build state-role JSONL dataset from hand frames and action events."
    )
    parser.add_argument("--hand", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--state-offset-ms", type=int, default=1000)
    parser.add_argument("--max-gap-ms", type=int, default=1500)
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--include-prev-action", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.state_offset_ms < 0:
        parser.error("--state-offset-ms must be non-negative")
    if args.max_gap_ms < 0:
        parser.error("--max-gap-ms must be non-negative")
    return args


def _load_hand_frames(hand_arg: str) -> list:
    path = None if hand_arg == "-" else Path(hand_arg)
    try:
        lines = read_text_lines(path, stdin=sys.stdin.buffer)
    except FileNotFoundError as exc:
        raise SystemExit(f"hand log not found: {hand_arg}") from exc
    except ValueError as exc:
        raise SystemExit(f"failed to decode hand log {hand_arg}: {exc}") from exc
    try:
        return parse_hand_frames(lines)
    except ValueError as exc:
        source = "stdin" if path is None else str(path)
        raise SystemExit(f"failed to parse hand log {source}: {exc}") from exc


def _load_actions(actions_arg: str) -> list[dict]:
    path = Path(actions_arg)
    if not path.exists():
        raise SystemExit(f"actions file not found: {actions_arg}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"failed to read actions file: {actions_arg}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"failed to decode actions as utf-8: {actions_arg}") from exc
    try:
        return parse_actions_lines(text.splitlines(), str(path))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    args = parse_args()
    frames = _load_hand_frames(args.hand)
    actions = _load_actions(args.actions)
    records, stats = build_state_role_records(
        frames,
        actions,
        state_offset_ms=args.state_offset_ms,
        max_gap_ms=args.max_gap_ms,
        include_debug=args.include_debug,
        include_prev_action=args.include_prev_action,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(
        "state_role dataset: "
        f"generated={stats.generated} skipped={stats.skipped_no_frame} "
        f"total_actions={stats.total_actions}"
    )


if __name__ == "__main__":
    main()
