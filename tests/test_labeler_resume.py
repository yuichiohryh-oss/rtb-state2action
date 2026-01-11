import json

from src.hand.labeler import _load_existing_labels, _load_paths


def test_labeler_skips_labeled_paths(tmp_path):
    crops_jsonl = tmp_path / "crops.jsonl"
    output_jsonl = tmp_path / "labels.jsonl"
    paths = [
        "data/hand_crops/session1/t_1_slot0.png",
        "data/hand_crops/session1/t_1_slot1.png",
        "data/hand_crops/session1/t_1_slot2.png",
    ]
    with crops_jsonl.open("w", encoding="utf-8") as fp:
        for path in paths:
            fp.write(json.dumps({"path": path}) + "\n")
    output_jsonl.write_text(json.dumps({"path": paths[1], "label": "CARD_2"}) + "\n", encoding="utf-8")

    all_paths = _load_paths(crops_jsonl)
    labeled = _load_existing_labels(output_jsonl)
    remaining = [path for path in all_paths if path not in labeled]

    assert remaining == [paths[0], paths[2]]


def test_labeler_last_label_wins(tmp_path):
    output_jsonl = tmp_path / "labels.jsonl"
    output_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"path": "a.png", "label": "CARD_1"}),
                json.dumps({"path": "a.png", "label": "CARD_3"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    labels = _load_existing_labels(output_jsonl)

    assert labels["a.png"] == "CARD_3"
