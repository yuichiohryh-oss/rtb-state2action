from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RoiParams:
    y_ratio: float = 0.72
    height_ratio: float = 0.26
    x_margin_ratio: float = 0.02
    x_offset_ratio: float = 0.0


def compute_hand_roi(window_w: int, window_h: int, params: RoiParams) -> Tuple[int, int, int, int]:
    x_margin = int(window_w * params.x_margin_ratio)
    y_start = int(window_h * params.y_ratio)
    roi_h = int(window_h * params.height_ratio)
    x1 = max(0, x_margin)
    y1 = max(0, y_start)
    x2 = min(window_w, window_w - x_margin)
    y2 = min(window_h, y_start + roi_h)
    dx = int(window_w * params.x_offset_ratio)
    x1 += dx
    x2 += dx
    x1 = max(0, x1)
    x2 = min(window_w, x2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI computed after applying offsets; check ROI params")
    return x1, y1, x2, y2


def split_roi_into_slots(roi: Tuple[int, int, int, int], slots: int = 4) -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi
    width = x2 - x1
    slot_w = width // slots
    boxes = []
    for idx in range(slots):
        sx1 = x1 + idx * slot_w
        sx2 = x1 + (idx + 1) * slot_w if idx < slots - 1 else x2
        boxes.append((sx1, y1, sx2, y2))
    return boxes
