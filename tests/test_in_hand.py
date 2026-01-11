from src.hand.infer import build_in_hand


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
