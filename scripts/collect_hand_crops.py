from __future__ import annotations

import argparse

from src.capture.roi import RoiParams
from src.hand.crop_collector import CollectorConfig, collect_hand_crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hand crops from a window capture.")
    parser.add_argument("--window-title", required=True)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--hand-y-ratio", type=float, default=None)
    parser.add_argument("--hand-height-ratio", type=float, default=None)
    parser.add_argument("--y-ratio", type=float, default=0.72, help="Deprecated: use --hand-y-ratio")
    parser.add_argument("--height-ratio", type=float, default=0.26, help="Deprecated: use --hand-height-ratio")
    parser.add_argument("--x-margin-ratio", type=float, default=0.02)
    parser.add_argument("--preview", action="store_true", help="Show ROI preview window without saving images (q to quit).")
    args = parser.parse_args()

    y_ratio = args.hand_y_ratio if args.hand_y_ratio is not None else args.y_ratio
    height_ratio = args.hand_height_ratio if args.hand_height_ratio is not None else args.height_ratio
    roi_params = RoiParams(
        y_ratio=y_ratio,
        height_ratio=height_ratio,
        x_margin_ratio=args.x_margin_ratio,
    )
    config = CollectorConfig(
        window_title=args.window_title,
        interval_ms=args.interval_ms,
        roi_params=roi_params,
        preview=args.preview,
    )
    collect_hand_crops(config)


if __name__ == "__main__":
    main()
