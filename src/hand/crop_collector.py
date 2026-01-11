from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2

from src.capture.frame_source import FrameSource, VideoFrameSource, WindowFrameSource
from src.capture.roi import RoiParams, compute_hand_roi
from src.hand.frame_slots import iter_hand_slots


@dataclass
class CollectorConfig:
    window_title: str | None
    video_path: Path | None = None
    video_fps: float = 4.0
    interval_ms: int = 250
    max_frames: int | None = None
    out_root: Path = Path("data/hand_crops")
    meta_path: Path = Path("data/hand_crops.jsonl")
    roi_params: RoiParams = RoiParams()
    preview: bool = False


def _log_start(config: CollectorConfig, window_w: int, window_h: int, session_dir: Path) -> None:
    roi = compute_hand_roi(window_w, window_h, config.roi_params)
    source_label = config.window_title or (str(config.video_path) if config.video_path else "unknown")
    print(
        "collect_hand_crops start:",
        f"source={source_label}",
        f"window_rect=({window_w},{window_h})",
        f"HAND_ROI=y_ratio={config.roi_params.y_ratio:.3f},height_ratio={config.roi_params.height_ratio:.3f},x_margin_ratio={config.roi_params.x_margin_ratio:.3f}",
        f"interval_ms={config.interval_ms}",
        f"video_fps={config.video_fps}",
        f"output_dir={session_dir}",
        f"roi_px={roi}",
        sep=" ",
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _preview_loop(frame_source: FrameSource, config: CollectorConfig, session_dir: Path) -> None:
    first = True
    for frame_data in frame_source.frames():
        window_h, window_w = frame_data.frame.shape[:2]
        if first:
            _log_start(config, window_w, window_h, session_dir)
            first = False
        roi = compute_hand_roi(window_w, window_h, config.roi_params)
        x1, y1, x2, y2 = roi
        display = frame_data.frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("hand_roi_preview", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


def _build_frame_source(config: CollectorConfig, for_preview: bool) -> FrameSource:
    if config.video_path is not None:
        return VideoFrameSource(config.video_path, video_fps=config.video_fps)
    if not config.window_title:
        raise ValueError("window_title is required when video_path is not set")
    interval_ms = 0 if for_preview else config.interval_ms
    return WindowFrameSource(config.window_title, interval_ms=interval_ms, prefer_dxcam=True)


def collect_hand_crops(config: CollectorConfig) -> None:
    session_id = time.strftime("%Y%m%d_%H%M%S")
    session_dir = config.out_root / session_id
    _ensure_dir(session_dir)
    _ensure_dir(config.meta_path.parent)

    if config.preview:
        preview_source = _build_frame_source(config, for_preview=True)
        _preview_loop(preview_source, config, session_dir)
        return

    with config.meta_path.open("a", encoding="utf-8") as meta_fp:
        try:
            frame_source = _build_frame_source(config, for_preview=False)
            started = False
            for frame_slots in iter_hand_slots(frame_source, config.roi_params, config.max_frames):
                window_h, window_w = frame_slots.frame.shape[:2]
                if not started:
                    _log_start(config, window_w, window_h, session_dir)
                    started = True
                saved_paths: List[Path] = []

                for slot_idx, (x1, y1, x2, y2) in enumerate(frame_slots.slots):
                    crop = frame_slots.frame[y1:y2, x1:x2]
                    filename = f"t_{frame_slots.t_ms}_slot{slot_idx}.png"
                    out_path = session_dir / filename
                    cv2.imwrite(str(out_path), crop)
                    saved_paths.append(out_path)

                    record: Dict[str, object] = {
                        "session_id": session_id,
                        "t_ms": frame_slots.t_ms,
                        "slot": slot_idx,
                        "path": str(out_path).replace("\\", "/"),
                        "window_w": window_w,
                        "window_h": window_h,
                        "roi_params": {
                            "y_ratio": config.roi_params.y_ratio,
                            "height_ratio": config.roi_params.height_ratio,
                            "x_margin_ratio": config.roi_params.x_margin_ratio,
                        },
                    }
                    meta_fp.write(json.dumps(record) + "\n")
                meta_fp.flush()
        except KeyboardInterrupt:
            return
