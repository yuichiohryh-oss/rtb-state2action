from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a manifest JSONL for position classification training."
    )
    parser.add_argument("--input", required=True, type=Path, help="actions_tap_pos.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="manifest.jsonl output path")
    parser.add_argument("--min-conf", type=float, default=0.7)
    parser.add_argument("--grid-w", type=int, default=18)
    parser.add_argument("--grid-h", type=int, default=11)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"--input not found: {args.input}")
    if args.min_conf < 0:
        parser.error("--min-conf must be non-negative")
    if args.grid_w <= 0 or args.grid_h <= 0:
        parser.error("--grid-w/--grid-h must be positive")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def resolve_diff_path(raw_path: str, jsonl_path: Path) -> Path | None:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [repo_root / candidate, jsonl_path.parent / candidate]
    for path in candidates:
        if path.exists():
            return path
    return None


def extract_diff_path(action: dict[str, Any], jsonl_path: Path) -> Path | None:
    paths = action.get("paths")
    raw = None
    if isinstance(paths, dict):
        raw = paths.get("diff")
    if raw is None:
        raw = action.get("diff")
    if raw is None:
        return None
    try:
        return resolve_diff_path(str(raw), jsonl_path)
    except (TypeError, ValueError):
        return None


def normalize_path(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    args = parse_args()
    actions = read_jsonl(args.input)
    out_rows: list[dict[str, Any]] = []
    skipped = 0

    for action in actions:
        pos = action.get("pos")
        if not isinstance(pos, dict):
            skipped += 1
            continue
        cell_id = pos.get("cell_id")
        if cell_id is None:
            skipped += 1
            continue
        try:
            cell_id = int(cell_id)
        except (TypeError, ValueError):
            skipped += 1
            continue

        conf = pos.get("conf", action.get("pos_conf", 1.0))
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < args.min_conf:
            skipped += 1
            continue

        diff_path = extract_diff_path(action, args.input)
        if diff_path is None:
            skipped += 1
            continue

        grid_w = pos.get("grid_w", args.grid_w)
        grid_h = pos.get("grid_h", args.grid_h)
        try:
            grid_w = int(grid_w)
            grid_h = int(grid_h)
        except (TypeError, ValueError):
            grid_w = args.grid_w
            grid_h = args.grid_h

        out_rows.append(
            {
                "img": normalize_path(diff_path),
                "label": cell_id,
                "conf": conf,
                "grid_w": grid_w,
                "grid_h": grid_h,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"input_rows={len(actions)}")
    print(f"written={len(out_rows)}")
    print(f"skipped={skipped}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
