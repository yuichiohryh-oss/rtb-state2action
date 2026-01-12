import numpy as np

from src.hand.infer import _build_slot_ids, _resolve_card_class_indices, _select_unique_cards, build_in_hand


def test_in_hand_shape_and_values():
    class_names = [f"CARD_{i}" for i in range(1, 9)]
    slot_preds = [0, 3, 3, 7]
    in_hand = build_in_hand(slot_preds, class_names)
    assert len(in_hand) == 8
    assert set(in_hand.keys()) == set(class_names)
    for value in in_hand.values():
        assert value in (0, 1)
    assert in_hand["CARD_1"] == 1
    assert in_hand["CARD_4"] == 1
    assert in_hand["CARD_8"] == 1


def test_select_unique_cards_prefers_unique_assignment():
    class_names = [f"CARD_{i}" for i in range(1, 9)]
    card_indices = _resolve_card_class_indices(class_names)
    assert card_indices == list(range(8))
    slot_probs = []
    for slot_idx in range(4):
        probs = np.full(8, 0.01, dtype=np.float32)
        probs[slot_idx] = 0.9
        slot_probs.append(probs)
    enforced = _select_unique_cards(slot_probs, card_indices)
    assert enforced == [0, 1, 2, 3]


def test_build_slot_ids_maps_unknown_to_zero():
    class_names = [f"CARD_{i}" for i in range(1, 9)] + ["UNKNOWN"]
    slot_preds = [0, 7, 8, -1]
    slots = _build_slot_ids(slot_preds, class_names)
    assert slots == [1, 8, 0, 0]
