from src.capture.roi import RoiParams, compute_hand_roi, split_roi_into_slots


def test_split_roi_bounds():
    window_w, window_h = 1280, 720
    params = RoiParams(y_ratio=0.7, height_ratio=0.25, x_margin_ratio=0.02)
    roi = compute_hand_roi(window_w, window_h, params)
    slots = split_roi_into_slots(roi, slots=4)
    assert len(slots) == 4
    for x1, y1, x2, y2 in slots:
        assert 0 <= x1 < x2 <= window_w
        assert 0 <= y1 < y2 <= window_h
