from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


def parse_roi(text: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be formatted as x,y,w,h")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError("ROI values must be integers") from exc
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("ROI values must be non-negative with positive width/height")
    return x, y, w, h


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label play positions from frame diffs around action timestamps."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--roi", type=parse_roi, default=None, help="x,y,w,h in pixels")
    parser.add_argument("--roi-pick-tms", type=int, default=None)
    parser.add_argument("--dt-ms", type=int, default=300)
    parser.add_argument("--grid-w", type=int, default=18)
    parser.add_argument("--grid-h", type=int, default=11)
    parser.add_argument("--thr", type=int, default=25)
    parser.add_argument("--after-stability", type=int, default=1)
    parser.add_argument("--after-step-ms", type=int, default=100)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--max-area", type=int, default=0)
    parser.add_argument(
        "--self-side-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limit candidates to the self-side (below the river).",
    )
    parser.add_argument(
        "--self-side-ratio",
        type=float,
        default=0.52,
        help="Self-side cutoff as a ratio of ROI height (cy >= H * ratio).",
    )
    parser.add_argument(
        "--component-score",
        choices=("area", "sum", "meanxarea"),
        default="sum",
        help="Score used to pick a component.",
    )
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument(
        "--debug-topn",
        type=int,
        default=0,
        help="Draw top-N candidate component boxes/scores in debug output.",
    )
    parser.add_argument(
        "--tap-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When action.tap is available, downweight candidates far from the tap.",
    )
    parser.add_argument(
        "--tap-sigma-frac",
        type=float,
        default=0.20,
        help="Sigma as a fraction of ROI width when --tap-sigma-px is 0.",
    )
    parser.add_argument(
        "--tap-sigma-px",
        type=int,
        default=0,
        help="Sigma in pixels (overrides --tap-sigma-frac when > 0).",
    )
    parser.add_argument(
        "--tap-prior-max-factor",
        type=float,
        default=1.0,
        help="Clamp upper bound for tap prior weight.",
    )
    parser.add_argument(
        "--tap-prior-min-factor",
        type=float,
        default=0.05,
        help="Clamp lower bound for tap prior weight.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dt_ms < 0:
        parser.error("--dt-ms must be non-negative")
    if args.grid_w <= 0 or args.grid_h <= 0:
        parser.error("--grid-w/--grid-h must be positive")
    if args.thr < 0:
        parser.error("--thr must be non-negative")
    if args.after_stability <= 0:
        parser.error("--after-stability must be positive")
    if args.after_step_ms < 0:
        parser.error("--after-step-ms must be non-negative")
    if args.min_area < 0 or args.max_area < 0:
        parser.error("--min-area/--max-area must be non-negative")
    if not (0.0 <= args.self_side_ratio <= 1.0):
        parser.error("--self-side-ratio must be between 0 and 1")
    if args.debug_topn < 0:
        parser.error("--debug-topn must be non-negative")
    if args.tap_sigma_frac < 0:
        parser.error("--tap-sigma-frac must be non-negative")
    if args.tap_sigma_px < 0:
        parser.error("--tap-sigma-px must be non-negative")
    if args.tap_prior_min_factor < 0 or args.tap_prior_max_factor < 0:
        parser.error("--tap-prior-min-factor/--tap-prior-max-factor must be non-negative")
    if args.tap_prior_min_factor > args.tap_prior_max_factor:
        parser.error("--tap-prior-min-factor must be <= --tap-prior-max-factor")
    return args


def read_actions(path: Path) -> list[dict]:
    actions: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            actions.append(json.loads(stripped))
    return actions


def get_frame_at(cap: cv2.VideoCapture, t_ms: int) -> np.ndarray:
    safe_t = max(0, int(t_ms))
    ok = cap.set(cv2.CAP_PROP_POS_MSEC, float(safe_t))
    if not ok:
        raise RuntimeError(f"Failed to seek to t_ms={safe_t}")
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame at t_ms={safe_t}")
    return frame


def crop_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def diff_mask(before: np.ndarray, after: np.ndarray, thr: int) -> tuple[np.ndarray, np.ndarray]:
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    before_blur = cv2.GaussianBlur(before_gray, (5, 5), 0)
    after_blur = cv2.GaussianBlur(after_gray, (5, 5), 0)
    diff = cv2.absdiff(after_blur, before_blur)
    _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    return diff, mask


def refine_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    return cleaned


def compute_component_scores(
    mask: np.ndarray,
    diff: np.ndarray,
    min_area: int,
    max_area: int,
    score_mode: str,
) -> list[dict]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    sums = np.bincount(
        labels.ravel(),
        weights=diff.astype(np.float32).ravel(),
        minlength=num_labels,
    )
    components: list[dict] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        if score_mode == "area":
            score = float(area)
        else:
            score = float(sums[label])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        components.append(
            {
                "label": label,
                "area": area,
                "score": score,
                "bbox": (x, y, w, h),
                "centroid": (float(cx), float(cy)),
            }
        )
    return components


def filter_self_side(
    components: list[dict], roi_h: int, self_side_ratio: float
) -> list[dict]:
    cutoff = roi_h * self_side_ratio
    return [comp for comp in components if comp["centroid"][1] >= cutoff]


def pick_component(components: list[dict]) -> dict | None:
    if not components:
        return None
    return max(components, key=lambda comp: (comp["score"], comp["area"]))


def grid_cell(
    cx: float, cy: float, roi_w: int, roi_h: int, grid_w: int, grid_h: int
) -> Tuple[int, int]:
    cell_w = roi_w / grid_w
    cell_h = roi_h / grid_h
    col = int(cx / cell_w) if cell_w > 0 else 0
    row = int(cy / cell_h) if cell_h > 0 else 0
    col = max(0, min(grid_w - 1, col))
    row = max(0, min(grid_h - 1, row))
    return row, col


def ensure_roi_in_frame(roi: Tuple[int, int, int, int], frame: np.ndarray) -> None:
    x, y, w, h = roi
    frame_h, frame_w = frame.shape[:2]
    if x + w > frame_w or y + h > frame_h:
        raise ValueError("ROI exceeds frame bounds")


def select_roi(cap: cv2.VideoCapture, t_ms: int) -> Tuple[int, int, int, int]:
    frame = get_frame_at(cap, t_ms)
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        raise ValueError("ROI selection canceled or invalid")
    ensure_roi_in_frame((x, y, w, h), frame)
    return x, y, w, h


def iterate_after_times(base_t_ms: int, after_stability: int, step_ms: int) -> Iterable[int]:
    for idx in range(after_stability):
        yield base_t_ms + idx * step_ms


def draw_debug_image(
    debug_path: Path,
    after_roi: np.ndarray,
    mask: np.ndarray,
    centroid: Tuple[float, float] | None,
    tap_xy: Tuple[float, float] | None,
    picked_bbox: Tuple[int, int, int, int] | None,
    candidates: list[dict] | None,
    debug_topn: int,
    grid_w: int,
    grid_h: int,
) -> None:
    canvas = after_roi.copy()
    roi_h, roi_w = canvas.shape[:2]
    for col in range(1, grid_w):
        x = int(col * roi_w / grid_w)
        cv2.line(canvas, (x, 0), (x, roi_h), (0, 255, 0), 1)
    for row in range(1, grid_h):
        y = int(row * roi_h / grid_h)
        cv2.line(canvas, (0, y), (roi_w, y), (0, 255, 0), 1)
    if centroid is not None:
        cx, cy = centroid
        cv2.circle(canvas, (int(cx), int(cy)), 6, (0, 0, 255), -1)
    if tap_xy is not None:
        tx, ty = tap_xy
        tx_i, ty_i = int(round(tx)), int(round(ty))
        cv2.line(canvas, (tx_i - 6, ty_i), (tx_i + 6, ty_i), (255, 0, 255), 1)
        cv2.line(canvas, (tx_i, ty_i - 6), (tx_i, ty_i + 6), (255, 0, 255), 1)
    if picked_bbox is not None:
        x, y, w, h = picked_bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 140, 255), 2)

    if candidates and debug_topn > 0:
        for comp in sorted(candidates, key=lambda item: item["score"], reverse=True)[
            :debug_topn
        ]:
            x, y, w, h = comp["bbox"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 200, 0), 1)
            label = f"{comp['score']:.1f}"
            cv2.putText(
                canvas,
                label,
                (x, max(10, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )

    thumb_w = max(50, roi_w // 4)
    thumb_h = max(50, roi_h // 4)
    mask_thumb = cv2.resize(mask, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
    mask_thumb_bgr = cv2.cvtColor(mask_thumb, cv2.COLOR_GRAY2BGR)
    canvas[0:thumb_h, 0:thumb_w] = mask_thumb_bgr

    cv2.imwrite(str(debug_path), canvas)


def extract_tap_xy(action: dict) -> Tuple[float, float] | None:
    tap = action.get("tap")
    if not isinstance(tap, dict):
        return None
    if "x" not in tap or "y" not in tap:
        return None
    try:
        return float(tap["x"]), float(tap["y"])
    except (TypeError, ValueError):
        return None


def tap_xy_in_roi(
    tap_xy: Tuple[float, float], roi: Tuple[int, int, int, int]
) -> Tuple[float, float] | None:
    tap_x, tap_y = tap_xy
    roi_x, roi_y, roi_w, roi_h = roi
    rel_x = tap_x - roi_x
    rel_y = tap_y - roi_y
    if rel_x < 0 or rel_y < 0 or rel_x >= roi_w or rel_y >= roi_h:
        return None
    return float(rel_x), float(rel_y)


def tap_sigma_px(args: argparse.Namespace, roi_w: int, roi_h: int) -> float:
    if args.tap_sigma_px > 0:
        sigma = float(args.tap_sigma_px)
    else:
        sigma = float(roi_w) * float(args.tap_sigma_frac)
    return max(10.0, sigma)


def apply_tap_prior(
    candidates: list[dict],
    tap_xy: Tuple[float, float],
    sigma_px: float,
    min_factor: float,
    max_factor: float,
) -> None:
    if sigma_px <= 0:
        return
    denom = 2.0 * sigma_px * sigma_px
    for comp in candidates:
        score_raw = float(comp["score"])
        comp["score_raw"] = score_raw
        centroid = comp.get("centroid")
        if centroid is None:
            x, y, w, h = comp["bbox"]
            cx = x + w / 2.0
            cy = y + h / 2.0
        else:
            cx, cy = centroid
        dx = float(cx) - tap_xy[0]
        dy = float(cy) - tap_xy[1]
        dist2 = dx * dx + dy * dy
        w_raw = float(np.exp(-dist2 / denom))
        w = max(min_factor, min(max_factor, w_raw))
        comp["score"] = score_raw * w


def main() -> None:
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.actions.exists():
        raise FileNotFoundError(f"Actions not found: {args.actions}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    try:
        actions = read_actions(args.actions)

        if args.roi is None:
            pick_t_ms = args.roi_pick_tms
            if pick_t_ms is None:
                pick_t_ms = int(actions[0]["t_ms"]) if actions else 0
            roi = select_roi(cap, pick_t_ms)
        else:
            roi = args.roi
            frame_for_roi = get_frame_at(cap, 0)
            ensure_roi_in_frame(roi, frame_for_roi)

        if args.debug_dir is not None:
            args.debug_dir.mkdir(parents=True, exist_ok=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for idx, action in enumerate(actions):
                if "t_ms" not in action:
                    raise ValueError("Action missing t_ms field")
                t_ms = int(action["t_ms"])
                before_frame = get_frame_at(cap, t_ms - args.dt_ms)
                before_roi = crop_roi(before_frame, roi)

                mask = None
                diff_accum = None
                after_roi = None
                for after_t_ms in iterate_after_times(
                    t_ms + args.dt_ms, args.after_stability, args.after_step_ms
                ):
                    after_frame = get_frame_at(cap, after_t_ms)
                    after_roi = crop_roi(after_frame, roi)
                    current_diff, current_mask = diff_mask(before_roi, after_roi, args.thr)
                    if mask is None:
                        mask = current_mask
                        diff_accum = current_diff
                    else:
                        mask = cv2.bitwise_and(mask, current_mask)
                        diff_accum = np.minimum(diff_accum, current_diff)

                if mask is None or after_roi is None:
                    raise RuntimeError("Failed to compute diff mask")

                mask = refine_mask(mask)
                components = compute_component_scores(
                    mask, diff_accum, args.min_area, args.max_area, args.component_score
                )
                candidates = components
                if args.self_side_only:
                    filtered = filter_self_side(components, mask.shape[0], args.self_side_ratio)
                    if filtered:
                        candidates = filtered
                tap_xy = None
                tap_roi_xy = None
                sigma_px = None
                best_before = None
                best_after = None
                if args.tap_prior:
                    tap_xy = extract_tap_xy(action)
                    if tap_xy is not None:
                        tap_roi_xy = tap_xy_in_roi(tap_xy, roi)
                if tap_roi_xy is not None and candidates:
                    sigma_px = tap_sigma_px(args, roi[2], roi[3])
                    best_before = max(comp["score"] for comp in candidates)
                    apply_tap_prior(
                        candidates,
                        tap_roi_xy,
                        sigma_px,
                        args.tap_prior_min_factor,
                        args.tap_prior_max_factor,
                    )
                    best_after = max(comp["score"] for comp in candidates)
                picked = pick_component(candidates)
                centroid = None if picked is None else picked["centroid"]

                if centroid is None:
                    pos = {
                        "cell_id": None,
                        "grid_w": args.grid_w,
                        "grid_h": args.grid_h,
                        "cx_roi": None,
                        "cy_roi": None,
                    }
                else:
                    cx, cy = centroid
                    row, col = grid_cell(
                        cx, cy, roi[2], roi[3], args.grid_w, args.grid_h
                    )
                    pos = {
                        "cell_id": int(row * args.grid_w + col),
                        "grid_w": args.grid_w,
                        "grid_h": args.grid_h,
                        "cx_roi": float(cx),
                        "cy_roi": float(cy),
                    }

                action_out = dict(action)
                action_out["pos"] = pos
                handle.write(json.dumps(action_out, ensure_ascii=True) + "\n")

                if args.debug_dir is not None:
                    if tap_roi_xy is not None:
                        best_before_str = (
                            f"{best_before:.3f}" if best_before is not None else "n/a"
                        )
                        best_after_str = (
                            f"{best_after:.3f}" if best_after is not None else "n/a"
                        )
                        print(
                            "tap_prior"
                            f" idx={idx}"
                            f" t_ms={t_ms}"
                            f" tap=({tap_roi_xy[0]:.1f},{tap_roi_xy[1]:.1f})"
                            f" sigma_px={sigma_px:.2f}"
                            f" best_before={best_before_str}"
                            f" best_after={best_after_str}"
                        )
                    name = f"{idx:04d}_t{t_ms}.jpg"
                    debug_path = args.debug_dir / name
                    draw_debug_image(
                        debug_path,
                        after_roi,
                        mask,
                        centroid,
                        tap_roi_xy,
                        None if picked is None else picked["bbox"],
                        candidates,
                        args.debug_topn,
                        args.grid_w,
                        args.grid_h,
                    )
    finally:
        cap.release()


if __name__ == "__main__":
    main()
