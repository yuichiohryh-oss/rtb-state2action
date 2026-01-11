from __future__ import annotations

import json
from pathlib import Path

from src.proposal_model import ProposalDataset, load_state_role_samples


def _write_state_role_jsonl_v3(path: Path) -> None:
    records = [
        {
            "in_hand_state": [1, 0, 0, 0, 0, 0, 0, 0],
            "prev_action_onehot": [0, 0, 0, 0, 0, 0, 0, 0],
            "prev2_action_onehot": [0, 0, 0, 0, 0, 0, 0, 0],
            "role": 1,
        },
        {
            "in_hand_state": [0, 1, 0, 0, 0, 0, 0, 0],
            "prev_action_onehot": [1, 0, 0, 0, 0, 0, 0, 0],
            "prev2_action_onehot": [0, 0, 0, 0, 0, 0, 0, 0],
            "role": 2,
        },
        {
            "in_hand_state": [0, 0, 1, 0, 0, 0, 0, 0],
            "prev_action_onehot": [0, 1, 0, 0, 0, 0, 0, 0],
            "prev2_action_onehot": [1, 0, 0, 0, 0, 0, 0, 0],
            "role": 3,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_v3_dataset_input_dim(tmp_path: Path) -> None:
    data_path = tmp_path / "state_role_v3.jsonl"
    _write_state_role_jsonl_v3(data_path)
    samples = load_state_role_samples(data_path)
    dataset = ProposalDataset(samples)
    features, label = dataset[0]
    assert dataset.input_dim == 24
    assert features.shape == (24,)
    assert label == 0
