from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.hand_actions import HandFrame


@dataclass(frozen=True)
class BuildStats:
    total_actions: int
    generated: int
    skipped_no_frame: int


def parse_actions_lines(lines: Iterable[str], source: str) -> list[dict]:
    actions: list[dict] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip().lstrip("\ufeff")
        if not stripped:
            continue
        if not stripped.startswith("{"):
            raise ValueError(f"unexpected non-JSON line {idx} in {source}: {stripped[:120]}")
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"failed to parse JSON at {source}:{idx}") from exc
        actions.append(payload)
    return actions


def _in_hand_list(in_hand: dict[str, int]) -> list[int]:
    values: list[int] = []
    for idx in range(1, 9):
        key = f"CARD_{idx}"
        value = in_hand.get(key, 0)
        if not isinstance(value, int):
            raise ValueError(f"in_hand[{key}] must be int")
        if value not in (0, 1):
            raise ValueError(f"in_hand[{key}] must be 0 or 1")
        values.append(value)
    return values


def build_state_role_records(
    frames: Sequence[HandFrame],
    actions: Sequence[dict],
    state_offset_ms: int = 1000,
    max_gap_ms: int = 1500,
    include_debug: bool = False,
    include_prev_action: bool = False,
    history: int = 1,
) -> tuple[list[dict], BuildStats]:
    if state_offset_ms < 0:
        raise ValueError("state_offset_ms must be non-negative")
    if max_gap_ms < 0:
        raise ValueError("max_gap_ms must be non-negative")
    if history < 0 or history > 2:
        raise ValueError("history must be 0..2")

    frame_times = [frame.t_ms for frame in frames]
    records: list[dict] = []
    skipped = 0
    prev_action_id = 0
    prev2_action_id = 0
    include_prev = include_prev_action or history >= 1
    include_prev2 = history >= 2

    for action in actions:
        t_ms_event = action.get("t_ms")
        card_id = action.get("card_id")
        if not isinstance(t_ms_event, int):
            raise ValueError("action t_ms must be int")
        if not isinstance(card_id, int):
            raise ValueError("action card_id must be int")

        t_ms_state = t_ms_event - state_offset_ms
        idx = bisect_right(frame_times, t_ms_state) - 1
        if idx < 0:
            skipped += 1
            prev2_action_id = prev_action_id
            prev_action_id = card_id
            continue
        gap = t_ms_state - frame_times[idx]
        if gap > max_gap_ms:
            skipped += 1
            prev2_action_id = prev_action_id
            prev_action_id = card_id
            continue

        frame = frames[idx]
        record = {
            "t_ms_event": t_ms_event,
            "t_ms_state": t_ms_state,
            "in_hand_state": _in_hand_list(frame.in_hand),
            "role": card_id,
            "state_source": "nearest_frame",
        }
        if include_prev:
            if prev_action_id < 0 or prev_action_id > 8:
                raise ValueError(f"prev_action must be 0..8, got {prev_action_id}")
            prev_onehot = [0 for _ in range(8)]
            if prev_action_id > 0:
                prev_onehot[prev_action_id - 1] = 1
            record["prev_action"] = prev_action_id
            record["prev_action_onehot"] = prev_onehot
        if include_prev2:
            if prev2_action_id < 0 or prev2_action_id > 8:
                raise ValueError(f"prev2_action must be 0..8, got {prev2_action_id}")
            prev2_onehot = [0 for _ in range(8)]
            if prev2_action_id > 0:
                prev2_onehot[prev2_action_id - 1] = 1
            record["prev2_action"] = prev2_action_id
            record["prev2_action_onehot"] = prev2_onehot
        if include_debug:
            if "confidence" in action:
                record["confidence"] = action["confidence"]
            if "hand_before" in action:
                record["hand_before"] = action["hand_before"]
            if "hand_after" in action:
                record["hand_after"] = action["hand_after"]
        records.append(record)
        prev2_action_id = prev_action_id
        prev_action_id = card_id

    stats = BuildStats(
        total_actions=len(actions),
        generated=len(records),
        skipped_no_frame=skipped,
    )
    return records, stats
