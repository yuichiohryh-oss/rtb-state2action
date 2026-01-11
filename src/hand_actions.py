from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, List, Optional, Sequence, Set


@dataclass(frozen=True)
class HandFrame:
    t_ms: int
    in_hand: dict[str, int]


def _detect_encoding(raw: bytes) -> str:
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "utf-16-le"


def read_text_lines(path: Path | None, stdin: Optional[BinaryIO] = None) -> list[str]:
    if path is None:
        if stdin is None:
            raise ValueError("stdin is required when path is None")
        raw = stdin.read()
    else:
        raw = Path(path).read_bytes()
    encoding = _detect_encoding(raw)
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(f"failed to decode input as {encoding}") from exc
    return text.splitlines()


def _parse_card_id(name: str) -> int:
    if name.startswith("CARD_"):
        suffix = name[5:]
        if suffix.isdigit():
            return int(suffix)
    raise ValueError(f"invalid card key: {name}")


def parse_hand_frames(lines: Iterable[str]) -> list[HandFrame]:
    frames: list[HandFrame] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip().lstrip("\ufeff")
        if not stripped:
            continue
        if not stripped.startswith("{"):
            if stripped.startswith("infer_hand start:"):
                continue
            raise ValueError(f"unexpected non-JSON line {idx}: {stripped[:120]}")
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"failed to parse JSON at line {idx}") from exc
        if "t_ms" not in payload or "in_hand" not in payload:
            raise ValueError(f"missing keys at line {idx}: {payload}")
        t_ms = payload["t_ms"]
        in_hand = payload["in_hand"]
        if not isinstance(t_ms, int):
            raise ValueError(f"t_ms must be int at line {idx}")
        if not isinstance(in_hand, dict):
            raise ValueError(f"in_hand must be object at line {idx}")
        frames.append(HandFrame(t_ms=t_ms, in_hand=in_hand))
    return frames


def _frame_set(in_hand: dict[str, int]) -> Set[int]:
    present = {(_parse_card_id(name)) for name, value in in_hand.items() if value == 1}
    return present


def compute_stable_hands(
    frames: Sequence[HandFrame], window: int
) -> tuple[list[Optional[Set[int]]], list[Optional[Set[int]]]]:
    if window <= 0:
        raise ValueError("window must be positive")
    history: list[Set[int]] = []
    stable: list[Optional[Set[int]]] = []
    raw: list[Optional[Set[int]]] = []
    current: Optional[Set[int]] = None

    for frame in frames:
        frame_set = _frame_set(frame.in_hand)
        raw_set: Optional[Set[int]] = None
        if len(frame_set) == 4:
            raw_set = frame_set
            history.append(frame_set)
            history = history[-window:]
        raw.append(raw_set)

        counts: dict[frozenset[int], int] = {}
        last_seen: dict[frozenset[int], int] = {}
        for idx, hand_set in enumerate(history):
            key = frozenset(hand_set)
            counts[key] = counts.get(key, 0) + 1
            last_seen[key] = idx
        majority: Optional[Set[int]] = None
        if counts:
            best_key = max(
                counts.keys(),
                key=lambda k: (counts[k], last_seen.get(k, -1)),
            )
            if counts[best_key] >= 2:
                majority = set(best_key)
        if majority is not None:
            current = majority
        stable.append(current if current is None else set(current))

    return stable, raw


def extract_actions(
    frames: Sequence[HandFrame],
    window: int = 3,
    confirm_frames: int = 2,
    cooldown_ms: int = 1000,
    pre_hold: int = 2,
) -> list[dict]:
    if confirm_frames <= 0:
        raise ValueError("confirm_frames must be positive")
    if cooldown_ms < 0:
        raise ValueError("cooldown_ms must be non-negative")
    if pre_hold < 0:
        raise ValueError("pre_hold must be non-negative")

    stable, raw = compute_stable_hands(frames, window)
    events: list[dict] = []
    last_event_time: dict[int, int] = {}

    for idx in range(1, len(frames)):
        prev = stable[idx - 1]
        curr = stable[idx]
        if not prev or not curr or prev == curr:
            continue
        removed = prev - curr
        added = curr - prev
        if len(removed) != 1 or len(added) != 1:
            continue

        if idx + confirm_frames - 1 >= len(frames):
            continue
        confirm = all(stable[j] == curr for j in range(idx, idx + confirm_frames))
        if not confirm:
            continue

        card_id = next(iter(removed))
        t_ms = frames[idx].t_ms
        last_time = last_event_time.get(card_id)
        if last_time is not None and t_ms - last_time < cooldown_ms:
            continue

        pre_count = 0
        if pre_hold > 0:
            for j in range(max(0, idx - pre_hold), idx):
                if stable[j] and card_id in stable[j]:
                    pre_count += 1
        invalid_count = 0
        for j in range(max(0, idx - 2), min(len(frames), idx + 3)):
            if raw[j] is None:
                invalid_count += 1

        confidence = 0.6
        if confirm:
            confidence += 0.2
        if pre_hold > 0 and pre_count >= pre_hold:
            confidence += 0.1
        if invalid_count >= 2:
            confidence -= 0.2
        confidence = max(0.0, min(1.0, confidence))

        notes = (
            f"confirm={confirm_frames} pre_hold={pre_count}/{pre_hold} "
            f"invalid_window={invalid_count}"
        )

        events.append(
            {
                "t_ms": t_ms,
                "card_id": card_id,
                "event": "play",
                "confidence": round(confidence, 3),
                "hand_before": sorted(prev),
                "hand_after": sorted(curr),
                "notes": notes,
            }
        )
        last_event_time[card_id] = t_ms

    return events
