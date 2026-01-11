from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np

from src.capture.frame_source import FrameSource
from src.capture.roi import RoiParams, compute_hand_roi, split_roi_into_slots


@dataclass
class FrameSlots:
    t_ms: int
    slots: List[Tuple[int, int, int, int]]
    frame: np.ndarray


def iter_hand_slots(
    frame_source: FrameSource,
    roi_params: RoiParams,
    max_frames: int | None,
) -> Iterator[FrameSlots]:
    count = 0
    for frame_data in frame_source.frames():
        if max_frames is not None and count >= max_frames:
            break
        window_h, window_w = frame_data.frame.shape[:2]
        roi = compute_hand_roi(window_w, window_h, roi_params)
        slots = split_roi_into_slots(roi, slots=4)
        yield FrameSlots(t_ms=frame_data.t_ms, slots=slots, frame=frame_data.frame)
        count += 1
