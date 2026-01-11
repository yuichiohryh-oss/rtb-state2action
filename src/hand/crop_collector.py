from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2

from src.capture.roi import RoiParams, compute_hand_roi, split_roi_into_slots
from src.capture.window_capture import WindowCapture


@dataclass
class CollectorConfig:
    window_title: str
    interval_ms: int = 250
    out_root: Path = Path("data/hand_crops")
    meta_path: Path = Path("data/hand_crops.jsonl")
    roi_params: RoiParams = RoiParams()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_hand_crops(config: CollectorConfig) -> None:
    capture = WindowCapture(config.window_title, prefer_dxcam=True)
    session_id = time.strftime("%Y%m%d_%H%M%S")
    session_dir = config.out_root / session_id
    _ensure_dir(session_dir)
    _ensure_dir(config.meta_path.parent)
    last_ms = 0

    with config.meta_path.open("a", encoding="utf-8") as meta_fp:
        try:
            while True:
                now_ms = int(time.time() * 1000)
                if now_ms - last_ms < config.interval_ms:
                    time.sleep(0.01)
                    continue
                last_ms = now_ms

                frame = capture.capture()
                window_h, window_w = frame.shape[:2]
                roi = compute_hand_roi(window_w, window_h, config.roi_params)
                slots = split_roi_into_slots(roi, slots=4)
                saved_paths: List[Path] = []

                for slot_idx, (x1, y1, x2, y2) in enumerate(slots):
                    crop = frame[y1:y2, x1:x2]
                    filename = f"t_{now_ms}_slot{slot_idx}.png"
                    out_path = session_dir / filename
                    cv2.imwrite(str(out_path), crop)
                    saved_paths.append(out_path)

                    record: Dict[str, object] = {
                        "session_id": session_id,
                        "t_ms": now_ms,
                        "slot": slot_idx,
                        "path": str(out_path).replace("\\", "/"),
                        "window_w": window_w,
                        "window_h": window_h,
                        "roi_params": {"y_ratio": config.roi_params.y_ratio},
                    }
                    meta_fp.write(json.dumps(record) + "\n")
                meta_fp.flush()
        except KeyboardInterrupt:
            return
