from __future__ import annotations

import json
from pathlib import Path

from src.hand_actions import (
    HandFrame,
    compute_stable_hands,
    extract_actions,
    parse_hand_frames,
    read_text_lines,
)


def make_in_hand(cards: set[int]) -> dict[str, int]:
    return {f"CARD_{idx}": 1 if idx in cards else 0 for idx in range(1, 9)}


def make_frame(t_ms: int, cards: set[int]) -> HandFrame:
    return HandFrame(t_ms=t_ms, in_hand=make_in_hand(cards))


def test_stable_hand_survives_invalid_frame() -> None:
    hand = {1, 2, 3, 4}
    frames = [
        make_frame(0, hand),
        make_frame(100, hand),
        make_frame(200, {1, 2, 3}),
        make_frame(300, hand),
    ]
    stable, _ = compute_stable_hands(frames, window=3)
    assert stable[1] == hand
    assert stable[2] == hand
    assert stable[3] == hand


def test_extracts_single_play_on_swap_with_confirm() -> None:
    hand_a = {1, 2, 3, 4}
    hand_b = {2, 3, 4, 5}
    frames = [
        make_frame(0, hand_a),
        make_frame(250, hand_a),
        make_frame(500, hand_a),
        make_frame(750, hand_b),
        make_frame(1000, hand_b),
        make_frame(1250, hand_b),
    ]
    actions = extract_actions(frames, window=3, confirm_frames=2)
    assert len(actions) == 1
    action = actions[0]
    assert action["card_id"] == 1
    assert action["t_ms"] == 1000
    assert action["hand_before"] == sorted(hand_a)
    assert action["hand_after"] == sorted(hand_b)


def test_no_action_without_confirm_frames() -> None:
    hand_a = {1, 2, 3, 4}
    hand_b = {2, 3, 4, 5}
    frames = [
        make_frame(0, hand_a),
        make_frame(100, hand_a),
        make_frame(200, hand_b),
        make_frame(300, hand_b),
    ]
    actions = extract_actions(frames, window=3, confirm_frames=2)
    assert actions == []


def test_cooldown_blocks_duplicate_card_events() -> None:
    hand_a = {1, 2, 3, 4}
    hand_b = {2, 3, 4, 5}
    hand_c = {1, 2, 3, 5}
    hand_d = {2, 3, 5, 6}
    frames = [
        make_frame(0, hand_a),
        make_frame(100, hand_a),
        make_frame(200, hand_a),
        make_frame(300, hand_b),
        make_frame(400, hand_b),
        make_frame(500, hand_b),
        make_frame(600, hand_c),
        make_frame(700, hand_c),
        make_frame(800, hand_c),
        make_frame(900, hand_d),
        make_frame(1000, hand_d),
        make_frame(1100, hand_d),
    ]
    actions = extract_actions(frames, window=3, confirm_frames=2, cooldown_ms=1000)
    card_ids = [action["card_id"] for action in actions]
    assert card_ids == [1, 4]


def test_utf16_input_is_parsed(tmp_path: Path) -> None:
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
    assert frames[0].t_ms == 0
    assert frames[0].in_hand["CARD_1"] == 1
