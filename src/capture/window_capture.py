from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import dxcam  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dxcam = None

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mss = None

try:
    import win32gui  # type: ignore
except Exception as exc:  # pragma: no cover - platform dependency
    raise RuntimeError("pywin32 is required on Windows for window capture") from exc


@dataclass
class WindowInfo:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


def _find_window_handle(title: str) -> int:
    matches = []

    def _enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if title.lower() in window_title.lower():
                matches.append(hwnd)

    win32gui.EnumWindows(_enum_handler, None)
    if not matches:
        raise RuntimeError(f"Window not found for title: {title}")
    return matches[0]


def get_client_rect(title: str) -> WindowInfo:
    hwnd = _find_window_handle(title)
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
    client_right, client_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    return WindowInfo(
        left=client_left,
        top=client_top,
        right=client_right,
        bottom=client_bottom,
    )


class WindowCapture:
    def __init__(self, title: str, prefer_dxcam: bool = True):
        self.title = title
        self.prefer_dxcam = prefer_dxcam and dxcam is not None
        self._dxcam = None
        self._mss = None
        if self.prefer_dxcam:
            self._dxcam = dxcam.create(output_idx=0)
        elif mss is not None:
            self._mss = mss.mss()
        else:
            raise RuntimeError("Neither dxcam nor mss is available")

    def capture(self) -> np.ndarray:
        rect = get_client_rect(self.title)
        region = (rect.left, rect.top, rect.right, rect.bottom)
        if self._dxcam is not None:
            frame = self._dxcam.grab(region=region)
            if frame is None:
                time.sleep(0.01)
                frame = self._dxcam.grab(region=region)
            if frame is None:
                raise RuntimeError("dxcam failed to capture frame")
            return frame[:, :, :3].copy()
        if self._mss is None:
            raise RuntimeError("mss is not initialized")
        monitor = {
            "left": rect.left,
            "top": rect.top,
            "width": rect.width,
            "height": rect.height,
        }
        grab = self._mss.grab(monitor)
        frame = np.array(grab, dtype=np.uint8)
        return frame[:, :, :3].copy()

    def get_window_size(self) -> Tuple[int, int]:
        rect = get_client_rect(self.title)
        return rect.width, rect.height
