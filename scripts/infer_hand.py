from __future__ import annotations

import argparse
from pathlib import Path

from src.capture.roi import RoiParams
from src.hand.infer import InferConfig, infer_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hand inference on a window capture.")
    parser.add_argument("--window-title", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--smoothing", choices=["majority", "max"], default="majority")
    parser.add_argument("--state-out", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--y-ratio", type=float, default=0.72)
    parser.add_argument("--height-ratio", type=float, default=0.26)
    parser.add_argument("--x-margin-ratio", type=float, default=0.02)
    args = parser.parse_args()

    roi_params = RoiParams(
        y_ratio=args.y_ratio,
        height_ratio=args.height_ratio,
        x_margin_ratio=args.x_margin_ratio,
    )
    config = InferConfig(
        window_title=args.window_title,
        model_path=args.model,
        interval_ms=args.interval_ms,
        history=args.history,
        smoothing=args.smoothing,
        state_out=args.state_out,
        image_size=args.image_size,
        roi_params=roi_params,
    )
    infer_loop(config)


if __name__ == "__main__":
    main()
