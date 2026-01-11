from __future__ import annotations

import argparse
from pathlib import Path

from src.capture.roi import RoiParams
from src.hand.crop_collector import CollectorConfig, collect_hand_crops


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect hand crops from a window capture.")
    parser.add_argument("--window-title")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--hand-y-ratio", type=float, default=None)
    parser.add_argument("--hand-height-ratio", type=float, default=None)
    parser.add_argument("--y-ratio", type=float, default=0.72, help="Deprecated: use --hand-y-ratio")
    parser.add_argument("--height-ratio", type=float, default=0.26, help="Deprecated: use --hand-height-ratio")
    parser.add_argument("--x-margin-ratio", type=float, default=0.02)
    parser.add_argument("--preview", action="store_true", help="Show ROI preview window without saving images (q to quit).")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video is None and args.window_title is None:
        parser.error("either --window-title or --video is required")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    return args


def main() -> None:
    args = parse_args()

    y_ratio = args.hand_y_ratio if args.hand_y_ratio is not None else args.y_ratio
    height_ratio = args.hand_height_ratio if args.hand_height_ratio is not None else args.height_ratio
    roi_params = RoiParams(
        y_ratio=y_ratio,
        height_ratio=height_ratio,
        x_margin_ratio=args.x_margin_ratio,
    )
    config = CollectorConfig(
        window_title=args.window_title,
        video_path=args.video,
        video_fps=args.video_fps,
        interval_ms=args.interval_ms,
        max_frames=args.max_frames,
        roi_params=roi_params,
        preview=args.preview,
    )
    collect_hand_crops(config)


if __name__ == "__main__":
    main()
