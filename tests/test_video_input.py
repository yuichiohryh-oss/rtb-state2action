from __future__ import annotations

import numpy as np

import scripts.collect_hand_crops as collect_hand_crops
import scripts.infer_hand as infer_hand
from src.capture.frame_source import FrameData
from src.capture.roi import RoiParams
from src.hand import frame_slots
from src.hand.frame_slots import iter_hand_slots


def test_collect_hand_crops_parser_accepts_video_args() -> None:
    args = collect_hand_crops.parse_args(["--video", "sample.mp4"])
    assert args.video is not None


def test_infer_hand_parser_accepts_video_args() -> None:
    args = infer_hand.parse_args(["--video", "sample.mp4", "--model", "model.pt"])
    assert args.video is not None


def test_iter_hand_slots_uses_roi_and_respects_max_frames(monkeypatch) -> None:
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
    t_ms_values = [0, 100, 200, 300, 400]

    class FakeVideoFrameSource:
        def frames(self):
            for idx, frame in enumerate(frames):
                yield FrameData(frame=frame, t_ms=t_ms_values[idx], frame_index=idx)

    call_count = {"split": 0}

    def spy_split(roi, slots):
        call_count["split"] += 1
        return [(0, 0, 1, 1)] * slots

    monkeypatch.setattr(frame_slots, "split_roi_into_slots", spy_split)

    results = list(iter_hand_slots(FakeVideoFrameSource(), RoiParams(), max_frames=3))
    assert call_count["split"] == 3
    assert len(results) == 3
    observed = [result.t_ms for result in results]
    assert all(b >= a for a, b in zip(observed, observed[1:]))
