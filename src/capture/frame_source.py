from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from src.capture.window_capture import WindowCapture


@dataclass
class FrameData:
    frame: np.ndarray
    t_ms: int
    frame_index: int


class FrameSource(Protocol):
    def frames(self) -> Iterator[FrameData]:
        ...


class WindowFrameSource:
    def __init__(self, window_title: str, interval_ms: int, prefer_dxcam: bool = True) -> None:
        self._capture = WindowCapture(window_title, prefer_dxcam=prefer_dxcam)
        self._interval_ms = max(0, interval_ms)

    def frames(self) -> Iterator[FrameData]:
        last_ms = 0
        frame_index = 0
        while True:
            now_ms = int(time.time() * 1000)
            if self._interval_ms > 0 and now_ms - last_ms < self._interval_ms:
                time.sleep(0.01)
                continue
            last_ms = now_ms
            frame = self._capture.capture()
            yield FrameData(frame=frame, t_ms=now_ms, frame_index=frame_index)
            frame_index += 1


class VideoFrameSource:
    def __init__(self, video_path: Path, video_fps: float) -> None:
        if video_fps <= 0:
            raise ValueError("video_fps must be positive")
        self._video_path = video_path
        self._video_fps = video_fps
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        source_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._source_fps = source_fps if source_fps > 0 else None

    def frames(self) -> Iterator[FrameData]:
        frame_index = 0
        next_target_ms = 0.0
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if frame_index == 0:
                    raise RuntimeError(f"Failed to read frames from video file: {self._video_path}")
                break
            pos_msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
            if pos_msec <= 0:
                if self._source_fps is not None:
                    t_ms = int((frame_index / self._source_fps) * 1000)
                else:
                    t_ms = int((frame_index / self._video_fps) * 1000)
            else:
                t_ms = int(pos_msec)
            frame_index += 1
            if t_ms < next_target_ms:
                continue
            yield FrameData(frame=frame, t_ms=t_ms, frame_index=frame_index - 1)
            next_target_ms = t_ms + (1000.0 / self._video_fps)
