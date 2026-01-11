from __future__ import annotations

import json
from pathlib import Path

from src.proposal_model import ProposalDataset, TrainConfig, load_state_role_samples, train_proposal_model


def _write_state_role_jsonl_v2(path: Path) -> None:
    records = [
        {
            "in_hand_state": [1, 0, 0, 0, 0, 0, 0, 0],
            "prev_action_onehot": [0, 0, 0, 0, 0, 0, 0, 0],
            "role": 1,
        },
        {
            "in_hand_state": [0, 1, 0, 0, 0, 0, 0, 0],
            "prev_action_onehot": [1, 0, 0, 0, 0, 0, 0, 0],
            "role": 2,
        },
        {
            "in_hand_state": [0, 0, 1, 0, 0, 0, 0, 0],
            "prev_action_onehot": [0, 1, 0, 0, 0, 0, 0, 0],
            "role": 3,
        },
        {
            "in_hand_state": [0, 0, 0, 1, 0, 0, 0, 0],
            "prev_action_onehot": [0, 0, 1, 0, 0, 0, 0, 0],
            "role": 4,
        },
        {
            "in_hand_state": [0, 0, 0, 0, 1, 0, 0, 0],
            "prev_action_onehot": [0, 0, 0, 1, 0, 0, 0, 0],
            "role": 5,
        },
        {
            "in_hand_state": [0, 0, 0, 0, 0, 1, 0, 0],
            "prev_action_onehot": [0, 0, 0, 0, 1, 0, 0, 0],
            "role": 6,
        },
        {
            "in_hand_state": [0, 0, 0, 0, 0, 0, 1, 0],
            "prev_action_onehot": [0, 0, 0, 0, 0, 1, 0, 0],
            "role": 7,
        },
        {
            "in_hand_state": [0, 0, 0, 0, 0, 0, 0, 1],
            "prev_action_onehot": [0, 0, 0, 0, 0, 0, 1, 0],
            "role": 8,
        },
        {
            "in_hand_state": [1, 1, 0, 0, 0, 0, 0, 0],
            "prev_action_onehot": [0, 0, 0, 0, 0, 0, 0, 1],
            "role": 1,
        },
        {
            "in_hand_state": [0, 1, 1, 0, 0, 0, 0, 0],
            "prev_action_onehot": [0, 0, 0, 0, 0, 0, 0, 0],
            "role": 2,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_v2_dataset_input_dim(tmp_path: Path) -> None:
    data_path = tmp_path / "state_role_v2.jsonl"
    _write_state_role_jsonl_v2(data_path)
    samples = load_state_role_samples(data_path)
    dataset = ProposalDataset(samples)
    features, label = dataset[0]
    assert dataset.input_dim == 16
    assert features.shape == (16,)
    assert label == 0


def test_train_one_epoch_v2(tmp_path: Path) -> None:
    data_path = tmp_path / "state_role_v2.jsonl"
    _write_state_role_jsonl_v2(data_path)
    run_root = tmp_path / "runs"
    config = TrainConfig(
        data_path=data_path,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        seed=123,
        val_split=0.2,
        run_root=run_root,
    )
    run_dir = train_proposal_model(config)
    assert (run_dir / "model.pt").exists()
