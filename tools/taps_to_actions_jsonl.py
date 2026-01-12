from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert taps.csv into actions_tap.jsonl entries."
    )
    parser.add_argument("--taps", required=True, type=Path, help="taps.csv path.")
    parser.add_argument(
        "--meta",
        required=False,
        type=Path,
        help="meta.json path (optional; kept for backward compatibility).",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if not args.taps.exists():
        raise FileNotFoundError(f"taps.csv not found: {args.taps}")
    if args.meta is not None and not args.meta.exists():
        raise FileNotFoundError(f"meta.json not found: {args.meta}")

    actions: list[dict] = []
    with args.taps.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event = (row.get("event") or "").strip().lower()
            if event not in {"down", "tap"}:
                continue
            t_ms = int(float(row["t_ms"]))
            x_raw = row.get("x_raw")
            y_raw = row.get("y_raw")
            if x_raw is None or y_raw is None:
                x_raw = row.get("x")
                y_raw = row.get("y")
            tap = {
                "x": int(float(row["x"])),
                "y": int(float(row["y"])),
                "x_raw": int(float(x_raw)) if x_raw is not None else 0,
                "y_raw": int(float(y_raw)) if y_raw is not None else 0,
            }
            actions.append({"t_ms": t_ms, "event": "tap", "tap": tap})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
