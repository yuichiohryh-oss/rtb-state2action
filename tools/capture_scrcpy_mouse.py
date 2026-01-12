from __future__ import annotations

import argparse
import csv
import json
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClickSample:
    t_ms: int
    x_client: float
    y_client: float
    client_w: float
    client_h: float
    button: str
    event: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record scrcpy video while capturing mouse click positions."
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--record-seconds", type=int, default=180, help="Recording duration in seconds."
    )
    parser.add_argument(
        "--scrcpy",
        type=str,
        default="scrcpy",
        help="Path to scrcpy executable (default: scrcpy).",
    )
    parser.add_argument("--max-fps", type=int, default=30, help="Max recording FPS.")
    parser.add_argument(
        "--max-size",
        type=int,
        default=720,
        help="Max scrcpy video size (long side).",
    )
    parser.add_argument(
        "--no-playback",
        dest="no_playback",
        action="store_true",
        default=True,
        help="Disable scrcpy playback window (default: true).",
    )
    parser.add_argument(
        "--playback",
        dest="no_playback",
        action="store_false",
        help="Enable scrcpy playback window.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.record_seconds <= 0:
        parser.error("--record-seconds must be positive")
    if args.max_fps <= 0:
        parser.error("--max-fps must be positive")
    if args.max_size <= 0:
        parser.error("--max-size must be positive")
    return args


def resolve_executable(path_value: str) -> str | None:
    found = shutil.which(path_value)
    if found:
        return found
    candidate = Path(path_value)
    if candidate.is_file():
        return str(candidate)
    return None


def load_win32():
    try:
        import win32con
        import win32gui
        import win32process
    except Exception as exc:  # pragma: no cover - platform import guard
        raise RuntimeError("pywin32 is required on Windows to run this tool.") from exc
    return win32con, win32gui, win32process


def load_pynput():
    try:
        from pynput import mouse
    except Exception as exc:  # pragma: no cover - platform import guard
        raise RuntimeError("pynput is required to capture mouse clicks.") from exc
    return mouse


def load_cv2():
    try:
        import cv2
    except Exception:
        return None
    return cv2


def find_window_for_pid(pid: int, timeout_s: float = 8.0) -> int | None:
    win32con, win32gui, win32process = load_win32()
    end_time = time.time() + timeout_s
    while time.time() < end_time:
        matches: list[int] = []

        def enum_handler(hwnd: int, _args: list[int]) -> None:
            _, window_pid = win32process.GetWindowThreadProcessId(hwnd)
            if window_pid != pid:
                return
            if not win32gui.IsWindow(hwnd):
                return
            matches.append(hwnd)

        win32gui.EnumWindows(enum_handler, [])
        if matches:
            return matches[0]
        time.sleep(0.2)
    return None


def get_client_metrics(hwnd: int) -> tuple[int, int, int, int]:
    _, win32gui, _ = load_win32()
    left_top = win32gui.ClientToScreen(hwnd, (0, 0))
    client_rect = win32gui.GetClientRect(hwnd)
    client_w = client_rect[2] - client_rect[0]
    client_h = client_rect[3] - client_rect[1]
    return left_top[0], left_top[1], client_w, client_h


def scale_to_frame(
    x_client: float, y_client: float, client_w: float, client_h: float, frame_w: int, frame_h: int
) -> tuple[int, int]:
    x_frame = int(round(x_client / client_w * frame_w))
    y_frame = int(round(y_client / client_h * frame_h))
    x_frame = max(0, min(frame_w - 1, x_frame))
    y_frame = max(0, min(frame_h - 1, y_frame))
    return x_frame, y_frame


def stop_scrcpy(proc: subprocess.Popen, hwnd: int | None, verbose: bool) -> None:
    if proc.poll() is not None:
        return

    win32con, win32gui, _ = load_win32()

    if hwnd:
        if verbose:
            print("Stopping scrcpy via WM_CLOSE...")
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass

    if verbose:
        print("Stopping scrcpy via CTRL_C_EVENT...")
    try:
        proc.send_signal(signal.CTRL_C_EVENT)
        proc.wait(timeout=5)
        return
    except Exception:
        pass

    if verbose:
        print("Stopping scrcpy via terminate/kill...")
    try:
        proc.terminate()
        proc.wait(timeout=3)
        return
    except Exception:
        pass
    proc.kill()


def read_video_meta(video_path: Path, verbose: bool) -> tuple[int | None, int | None, float | None]:
    cv2 = load_cv2()
    if cv2 is None:
        if verbose:
            print("WARNING: opencv-python not available; frame size/fps unavailable.")
        return None, None, None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if verbose:
            print(f"WARNING: Failed to open video: {video_path}")
        return None, None, None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0:
            return None, None, fps
        return width, height, fps
    finally:
        cap.release()


def main() -> None:
    args = parse_args()

    if sys.platform != "win32":
        print("ERROR: capture_scrcpy_mouse.py is supported on Windows only.")
        sys.exit(2)

    scrcpy_path = resolve_executable(args.scrcpy)
    if scrcpy_path is None:
        print("ERROR: scrcpy executable not found:", args.scrcpy)
        sys.exit(2)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video.mp4"
    taps_path = out_dir / "taps.csv"
    meta_path = out_dir / "meta.json"

    scrcpy_cmd = [
        scrcpy_path,
        "--record",
        str(video_path),
        "--max-fps",
        str(args.max_fps),
        "--max-size",
        str(args.max_size),
        "--no-audio",
    ]
    if args.no_playback:
        scrcpy_cmd.append("--no-playback")

    stdout_target = None if args.verbose else subprocess.DEVNULL
    stderr_target = None if args.verbose else subprocess.DEVNULL
    scrcpy_proc = subprocess.Popen(
        scrcpy_cmd,
        stdout=stdout_target,
        stderr=stderr_target,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    record_start_perf = time.perf_counter()

    hwnd = find_window_for_pid(scrcpy_proc.pid)
    if hwnd is None:
        stop_scrcpy(scrcpy_proc, None, args.verbose)
        print("ERROR: Failed to find scrcpy window. Use --playback to enable it.")
        sys.exit(2)

    mouse = load_pynput()
    taps: list[ClickSample] = []

    def on_click(x: int, y: int, button: object, pressed: bool) -> None:
        if not pressed:
            return
        if button != mouse.Button.left:
            return
        try:
            client_left, client_top, client_w, client_h = get_client_metrics(hwnd)
        except Exception:
            return
        if client_w <= 0 or client_h <= 0:
            return
        x_client = x - client_left
        y_client = y - client_top
        if x_client < 0 or y_client < 0 or x_client > client_w or y_client > client_h:
            return
        t_ms = int((time.perf_counter() - record_start_perf) * 1000)
        if t_ms < 0:
            t_ms = 0
        taps.append(
            ClickSample(
                t_ms=t_ms,
                x_client=float(x_client),
                y_client=float(y_client),
                client_w=float(client_w),
                client_h=float(client_h),
                button="left",
                event="down",
            )
        )

    listener = mouse.Listener(on_click=on_click)
    listener.start()

    try:
        end_time = record_start_perf + args.record_seconds
        while time.perf_counter() < end_time:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        listener.join(timeout=2)
        stop_scrcpy(scrcpy_proc, hwnd, args.verbose)

    frame_w, frame_h, fps = read_video_meta(video_path, args.verbose)

    with taps_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_ms", "x", "y", "button", "event"])
        for tap in taps:
            if frame_w is not None and frame_h is not None:
                x_frame, y_frame = scale_to_frame(
                    tap.x_client, tap.y_client, tap.client_w, tap.client_h, frame_w, frame_h
                )
            else:
                x_frame, y_frame = int(round(tap.x_client)), int(round(tap.y_client))
            writer.writerow([tap.t_ms, x_frame, y_frame, tap.button, tap.event])

    meta = {
        "scrcpy_path": scrcpy_path,
        "scrcpy_args": scrcpy_cmd,
        "record_start_perf": record_start_perf,
        "record_seconds": args.record_seconds,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "fps": fps,
        "audio": False,
        "click_source": "pynput",
        "coords": "frame" if frame_w is not None and frame_h is not None else "client",
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
