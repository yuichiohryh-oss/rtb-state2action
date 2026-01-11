from __future__ import annotations

from src.hand_actions import HandFrame
from src.state_role_dataset import build_state_role_records


def _make_in_hand(cards: set[int]) -> dict[str, int]:
    return {f"CARD_{idx}": 1 if idx in cards else 0 for idx in range(1, 9)}


def _make_frame(t_ms: int, cards: set[int]) -> HandFrame:
    return HandFrame(t_ms=t_ms, in_hand=_make_in_hand(cards))


def test_build_state_role_prev_action_fields() -> None:
    frames = [
        _make_frame(0, {1, 2, 3, 4}),
        _make_frame(1000, {2, 3, 4, 5}),
        _make_frame(2000, {3, 4, 5, 6}),
    ]
    actions = [
        {"t_ms": 1000, "card_id": 1, "event": "play"},
        {"t_ms": 2000, "card_id": 2, "event": "play"},
        {"t_ms": 3000, "card_id": 3, "event": "play"},
    ]

    records, stats = build_state_role_records(
        frames,
        actions,
        state_offset_ms=1000,
        max_gap_ms=1500,
        include_prev_action=True,
    )

    assert stats.generated == 3
    assert records[0]["prev_action"] == 0
    assert records[0]["prev_action_onehot"] == [0, 0, 0, 0, 0, 0, 0, 0]
    assert records[1]["prev_action"] == 1
    assert records[1]["prev_action_onehot"] == [1, 0, 0, 0, 0, 0, 0, 0]
    assert records[2]["prev_action"] == 2
    assert records[2]["prev_action_onehot"] == [0, 1, 0, 0, 0, 0, 0, 0]
