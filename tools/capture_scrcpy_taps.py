from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


TYPE_MAP = {
    "EV_ABS": "EV_ABS",
    "EV_KEY": "EV_KEY",
    "EV_SYN": "EV_SYN",
    "0003": "EV_ABS",
    "0001": "EV_KEY",
    "0000": "EV_SYN",
}

CODE_X = {"ABS_MT_POSITION_X", "ABS_X", "0035", "0000"}
CODE_Y = {"ABS_MT_POSITION_Y", "ABS_Y", "0036", "0001"}
CODE_TRACKING_ID = {"ABS_MT_TRACKING_ID", "0039"}
CODE_BTN_TOUCH = {"BTN_TOUCH", "014a"}
CODE_SYN_REPORT = {"SYN_REPORT", "0000"}


@dataclass
class TapEvent:
    t_s: float
    event: str
    x_raw: int
    y_raw: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record scrcpy video while capturing tap coordinates from adb getevent."
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--record-seconds", type=int, default=180, help="Recording duration in seconds."
    )
    parser.add_argument("--serial", type=str, default=None, help="adb device serial.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.record_seconds <= 0:
        parser.error("--record-seconds must be positive")
    return args


def run_command(args: list[str], timeout: int = 10) -> str:
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(args)}): {result.stderr.strip()}"
        )
    return result.stdout


def parse_wm_size(output: str) -> tuple[int, int]:
    matches = re.findall(r"(\d+)x(\d+)", output)
    if not matches:
        raise RuntimeError("Failed to parse device size from adb output.")
    w, h = matches[-1]
    return int(w), int(h)


def get_device_size(adb_args: list[str]) -> tuple[int, int]:
    output = run_command(adb_args + ["shell", "wm", "size"])
    return parse_wm_size(output)


def parse_raw_max(output: str) -> tuple[int | None, int | None]:
    raw_max_x = None
    raw_max_y = None
    for line in output.splitlines():
        if "ABS_MT_POSITION_X" in line or "ABS_X" in line:
            match = re.search(r"max\s+(\d+)", line)
            if match:
                raw_max_x = int(match.group(1))
        if "ABS_MT_POSITION_Y" in line or "ABS_Y" in line:
            match = re.search(r"max\s+(\d+)", line)
            if match:
                raw_max_y = int(match.group(1))
    return raw_max_x, raw_max_y


def get_raw_max(adb_args: list[str]) -> tuple[int | None, int | None, str | None]:
    try:
        output = run_command(adb_args + ["shell", "getevent", "-lp"], timeout=15)
    except Exception:
        return None, None, None
    raw_max_x, raw_max_y = parse_raw_max(output)
    if raw_max_x is None and raw_max_y is None:
        return None, None, None
    return raw_max_x, raw_max_y, "adb getevent -lp"


def open_video_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    finally:
        cap.release()


def parse_getevent_line(line: str) -> tuple[float, str, str, str] | None:
    match = re.match(r"^\[\s*(\d+\.\d+)\]\s+(.+)$", line.strip())
    if not match:
        return None
    t_s = float(match.group(1))
    rest = match.group(2)
    if ":" in rest:
        rest = rest.split(":", 1)[1].strip()
    tokens = rest.split()
    if len(tokens) < 3:
        return None
    type_code, code, value = tokens[-3], tokens[-2], tokens[-1]
    return t_s, type_code, code, value


def parse_hex_value(text: str) -> int:
    value = int(text, 16)
    if value >= 2**31:
        value -= 2**32
    return value


class GeteventTapParser:
    def __init__(self) -> None:
        self.x_raw: int | None = None
        self.y_raw: int | None = None
        self.pending_event: str | None = None

    def feed_line(self, line: str) -> list[TapEvent]:
        parsed = parse_getevent_line(line)
        if parsed is None:
            return []
        t_s, type_code, code, value_text = parsed
        event_type = TYPE_MAP.get(type_code, type_code)
        code_norm = code
        try:
            value = parse_hex_value(value_text)
        except ValueError:
            return []

        if event_type == "EV_ABS":
            if code_norm in CODE_X:
                self.x_raw = value
            elif code_norm in CODE_Y:
                self.y_raw = value
            elif code_norm in CODE_TRACKING_ID:
                self.pending_event = "up" if value == -1 else "down"
        elif event_type == "EV_KEY" and code_norm in CODE_BTN_TOUCH:
            self.pending_event = "down" if value == 1 else "up"

        if event_type == "EV_SYN" and code_norm in CODE_SYN_REPORT:
            if self.pending_event and self.x_raw is not None and self.y_raw is not None:
                event = TapEvent(
                    t_s=t_s,
                    event=self.pending_event,
                    x_raw=self.x_raw,
                    y_raw=self.y_raw,
                )
                self.pending_event = None
                return [event]
            self.pending_event = None
        return []


def scale_raw(
    raw: int, raw_max: int | None, device_max: int
) -> tuple[int, str]:
    if raw_max is None or raw_max <= 0:
        return int(max(0, min(device_max - 1, raw))), "raw_as_device"
    scaled = int(round(raw / raw_max * (device_max - 1)))
    return int(max(0, min(device_max - 1, scaled))), "scaled_from_raw_max"


def stop_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    args = parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video.mp4"
    taps_path = out_dir / "taps.csv"
    meta_path = out_dir / "meta.json"

    adb_args = ["adb"]
    if args.serial:
        adb_args += ["-s", args.serial]

    device_w, device_h = get_device_size(adb_args)
    raw_max_x, raw_max_y, raw_max_source = get_raw_max(adb_args)

    getevent_cmd = adb_args + ["shell", "getevent", "-lt"]
    getevent_proc = subprocess.Popen(
        getevent_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    scrcpy_cmd = [
        "scrcpy",
        "--record",
        str(video_path),
        "--no-playback",
    ]
    if args.serial:
        scrcpy_cmd += ["--serial", args.serial]

    scrcpy_proc = subprocess.Popen(
        scrcpy_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    capture_start_wall = time.monotonic()
    parser = GeteventTapParser()
    tap_rows: list[dict] = []
    base_event_ts: float | None = None
    base_event_wall: float | None = None
    stop_flag = threading.Event()

    def reader() -> None:
        nonlocal base_event_ts, base_event_wall
        if getevent_proc.stdout is None:
            return
        for line in getevent_proc.stdout:
            if stop_flag.is_set():
                break
            events = parser.feed_line(line)
            for event in events:
                wall_now = time.monotonic()
                if base_event_ts is None:
                    base_event_ts = event.t_s
                    base_event_wall = wall_now
                if base_event_wall is None or base_event_ts is None:
                    continue
                event_wall = base_event_wall + (event.t_s - base_event_ts)
                t_ms = int(round((event_wall - capture_start_wall) * 1000))
                x, x_scale = scale_raw(event.x_raw, raw_max_x, device_w)
                y, y_scale = scale_raw(event.y_raw, raw_max_y, device_h)
                tap_rows.append(
                    {
                        "t_ms": t_ms,
                        "event": event.event,
                        "x": x,
                        "y": y,
                        "x_raw": event.x_raw,
                        "y_raw": event.y_raw,
                        "x_scale": x_scale,
                        "y_scale": y_scale,
                    }
                )

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    try:
        end_time = capture_start_wall + args.record_seconds
        while time.monotonic() < end_time:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        stop_process(scrcpy_proc, "scrcpy")
        stop_process(getevent_proc, "adb getevent")
        reader_thread.join(timeout=2)

    frame_w, frame_h = open_video_size(video_path)

    with taps_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t_ms",
                "event",
                "x",
                "y",
                "x_raw",
                "y_raw",
                "device_w",
                "device_h",
                "frame_w",
                "frame_h",
            ]
        )
        for row in tap_rows:
            writer.writerow(
                [
                    row["t_ms"],
                    row["event"],
                    row["x"],
                    row["y"],
                    row["x_raw"],
                    row["y_raw"],
                    device_w,
                    device_h,
                    frame_w,
                    frame_h,
                ]
            )

    meta = {
        "serial": args.serial,
        "device_w": device_w,
        "device_h": device_h,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "raw_max_x": raw_max_x,
        "raw_max_y": raw_max_y,
        "raw_scale_source": raw_max_source or "raw_as_device",
        "raw_scale_note": "x/y are scaled to device coordinates.",
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
