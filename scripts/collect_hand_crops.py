from __future__ import annotations

import argparse

from src.capture.roi import RoiParams
from src.hand.crop_collector import CollectorConfig, collect_hand_crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hand crops from a window capture.")
    parser.add_argument("--window-title", required=True)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--y-ratio", type=float, default=0.72)
    parser.add_argument("--height-ratio", type=float, default=0.26)
    parser.add_argument("--x-margin-ratio", type=float, default=0.02)
    args = parser.parse_args()

    roi_params = RoiParams(
        y_ratio=args.y_ratio,
        height_ratio=args.height_ratio,
        x_margin_ratio=args.x_margin_ratio,
    )
    config = CollectorConfig(
        window_title=args.window_title,
        interval_ms=args.interval_ms,
        roi_params=roi_params,
    )
    collect_hand_crops(config)


if __name__ == "__main__":
    main()
