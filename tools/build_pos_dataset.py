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
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="labeler debug_dir for resolving diff images",
    )
    parser.add_argument(
        "--prefer-diff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prefer filenames containing 'diff' when multiple matches exist",
    )
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


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    total_lines = 0
    bad_json = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            total_lines += 1
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                bad_json += 1
    return rows, total_lines, bad_json


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


def normalize_t_ms(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return None


def select_match(matches: list[Path], prefer_diff: bool) -> Path | None:
    if not matches:
        return None
    if not prefer_diff:
        return sorted(matches, key=lambda path: path.name.lower())[0]
    return sorted(
        matches,
        key=lambda path: (
            0 if "diff" in path.name.lower() else 1,
            path.name.lower(),
        ),
    )[0]


def resolve_diff_from_debug(
    debug_dir: Path, t_ms: str, prefer_diff: bool
) -> Path | None:
    patterns = [
        f"*t{t_ms}*diff*.png",
        f"*t{t_ms}*diff*.jpg",
        f"*t{t_ms}*diff*.*",
        f"*t{t_ms}*.png",
        f"*t{t_ms}*.jpg",
    ]
    for pattern in patterns:
        matches = list(debug_dir.glob(pattern))
        if matches:
            return select_match(matches, prefer_diff)
    return None


def pick_diff_path(
    action: dict[str, Any],
    jsonl_path: Path,
    debug_dir: Path | None,
    prefer_diff: bool,
) -> Path | None:
    diff_path = extract_diff_path(action, jsonl_path)
    if diff_path is not None:
        return diff_path
    if debug_dir is None:
        return None
    t_ms = normalize_t_ms(action.get("t_ms"))
    if t_ms is None:
        return None
    return resolve_diff_from_debug(debug_dir, t_ms, prefer_diff)


def main() -> None:
    args = parse_args()
    actions, total_lines, bad_json = read_jsonl(args.input)
    out_rows: list[dict[str, Any]] = []
    skip_counts = {
        "no_label": 0,
        "no_img": 0,
        "low_conf": 0,
        "bad_label": 0,
        "bad_json": bad_json,
    }

    for action in actions:
        pos = action.get("pos")
        if not isinstance(pos, dict):
            skip_counts["no_label"] += 1
            continue
        cell_id = pos.get("cell_id")
        if cell_id is None:
            skip_counts["no_label"] += 1
            continue
        try:
            cell_id = int(cell_id)
        except (TypeError, ValueError):
            skip_counts["bad_label"] += 1
            continue

        conf = pos.get("conf", action.get("pos_conf", 1.0))
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < args.min_conf:
            skip_counts["low_conf"] += 1
            continue

        diff_path = pick_diff_path(action, args.input, args.debug_dir, args.prefer_diff)
        if diff_path is None:
            skip_counts["no_img"] += 1
            continue

        grid_w = pos.get("grid_w", args.grid_w)
        grid_h = pos.get("grid_h", args.grid_h)
        try:
            grid_w = int(grid_w)
            grid_h = int(grid_h)
        except (TypeError, ValueError):
            grid_w = args.grid_w
            grid_h = args.grid_h

        out_row = {
            "img": normalize_path(diff_path),
            "label": cell_id,
            "conf": conf,
            "grid_w": grid_w,
            "grid_h": grid_h,
        }
        t_ms = normalize_t_ms(action.get("t_ms"))
        if t_ms is not None:
            out_row["t_ms"] = int(t_ms)
        out_rows.append(out_row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    total_skipped = sum(skip_counts.values())
    print(f"input_rows={total_lines}")
    print(f"written={len(out_rows)}")
    print(f"skipped={total_skipped}")
    print("skip_breakdown=" + json.dumps(skip_counts, ensure_ascii=True))
    print(f"written>0={len(out_rows) > 0}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
