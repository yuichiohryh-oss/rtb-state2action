from __future__ import annotations

import json
from pathlib import Path

import torch

from src.proposal_model import (
    ProposalDataset,
    StateRoleSample,
    TrainConfig,
    load_state_role_samples,
    train_proposal_model,
)


def _write_state_role_jsonl(path: Path) -> None:
    records = [
        {"in_hand_state": [1, 0, 0, 0, 0, 0, 0, 0], "role": 1},
        {"in_hand_state": [0, 1, 0, 0, 0, 0, 0, 0], "role": 2},
        {"in_hand_state": [0, 0, 1, 0, 0, 0, 0, 0], "role": 3},
        {"in_hand_state": [0, 0, 0, 1, 0, 0, 0, 0], "role": 4},
        {"in_hand_state": [0, 0, 0, 0, 1, 0, 0, 0], "role": 5},
        {"in_hand_state": [0, 0, 0, 0, 0, 1, 0, 0], "role": 6},
        {"in_hand_state": [0, 0, 0, 0, 0, 0, 1, 0], "role": 7},
        {"in_hand_state": [0, 0, 0, 0, 0, 0, 0, 1], "role": 8},
        {"in_hand_state": [1, 1, 0, 0, 0, 0, 0, 0], "role": 1},
        {"in_hand_state": [0, 1, 1, 0, 0, 0, 0, 0], "role": 2},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_load_state_role_samples(tmp_path: Path) -> None:
    path = tmp_path / "state_role.jsonl"
    _write_state_role_jsonl(path)
    samples = load_state_role_samples(path)
    assert len(samples) == 10
    assert samples[0].role == 1
    assert samples[0].in_hand == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_proposal_dataset_shape() -> None:
    samples = [StateRoleSample(in_hand=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], role=1)]
    dataset = ProposalDataset(samples)
    features, label = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (8,)
    assert label == 0


def test_train_one_epoch(tmp_path: Path) -> None:
    data_path = tmp_path / "state_role.jsonl"
    _write_state_role_jsonl(data_path)
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
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.json").exists()
