from __future__ import annotations

import json
from pathlib import Path

from src.hand_actions import HandFrame, parse_hand_frames, read_text_lines
from src.state_role_dataset import build_state_role_records


def make_in_hand(cards: set[int]) -> dict[str, int]:
    return {f"CARD_{idx}": 1 if idx in cards else 0 for idx in range(1, 9)}


def make_frame(t_ms: int, cards: set[int]) -> HandFrame:
    return HandFrame(t_ms=t_ms, in_hand=make_in_hand(cards))


def test_parse_hand_frames_ignores_header_line() -> None:
    lines = [
        "infer_hand start: smoothing=majority",
        json.dumps({"t_ms": 0, "in_hand": make_in_hand({1, 2, 3, 4})}),
    ]
    frames = parse_hand_frames(lines)
    assert len(frames) == 1
    assert frames[0].t_ms == 0


def test_build_state_role_uses_offset_and_nearest_frame() -> None:
    frames = [
        make_frame(0, {1, 2, 3, 4}),
        make_frame(250, {1, 2, 3, 4}),
        make_frame(500, {2, 3, 4, 5}),
    ]
    actions = [{"t_ms": 1250, "card_id": 8, "event": "play"}]
    records, stats = build_state_role_records(
        frames,
        actions,
        state_offset_ms=1000,
        max_gap_ms=1500,
    )
    assert stats.generated == 1
    record = records[0]
    assert record["t_ms_state"] == 250
    assert record["in_hand_state"] == [1, 1, 1, 1, 0, 0, 0, 0]
    assert record["role"] == 8


def test_build_state_role_skips_when_gap_exceeds_max() -> None:
    frames = [make_frame(0, {1, 2, 3, 4})]
    actions = [{"t_ms": 2000, "card_id": 1, "event": "play"}]
    records, stats = build_state_role_records(
        frames,
        actions,
        state_offset_ms=1000,
        max_gap_ms=500,
    )
    assert records == []
    assert stats.skipped_no_frame == 1


def test_utf16_hand_log_is_readable(tmp_path: Path) -> None:
    sample = [
        "infer_hand start: smoothing=majority",
        json.dumps({"t_ms": 0, "in_hand": make_in_hand({1, 2, 3, 4})}),
    ]
    payload = "\n".join(sample).encode("utf-16")
    path = tmp_path / "infer.txt"
    path.write_bytes(payload)

    lines = read_text_lines(path)
    frames = parse_hand_frames(lines)
    assert len(frames) == 1
    assert frames[0].in_hand["CARD_1"] == 1
