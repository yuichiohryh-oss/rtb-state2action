from __future__ import annotations

import json
from pathlib import Path

from src.proposal_model import TrainConfig, train_proposal_model


def _write_state_role_jsonl_imbalanced(path: Path) -> None:
    records = [
        {"in_hand_state": [1, 0, 0, 0, 0, 0, 0, 0], "role": 1},
        {"in_hand_state": [1, 1, 0, 0, 0, 0, 0, 0], "role": 1},
        {"in_hand_state": [0, 1, 0, 0, 0, 0, 0, 0], "role": 2},
        {"in_hand_state": [0, 1, 1, 0, 0, 0, 0, 0], "role": 2},
        {"in_hand_state": [0, 0, 0, 1, 0, 0, 0, 0], "role": 4},
        {"in_hand_state": [0, 0, 0, 1, 1, 0, 0, 0], "role": 4},
        {"in_hand_state": [0, 0, 0, 1, 0, 1, 0, 0], "role": 4},
        {"in_hand_state": [0, 0, 0, 1, 0, 0, 1, 0], "role": 4},
        {"in_hand_state": [0, 0, 0, 1, 0, 0, 0, 1], "role": 4},
        {"in_hand_state": [1, 0, 0, 1, 0, 0, 0, 0], "role": 4},
        {"in_hand_state": [0, 1, 0, 1, 0, 0, 0, 0], "role": 4},
        {"in_hand_state": [0, 0, 1, 1, 0, 0, 0, 0], "role": 4},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _load_metrics(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def test_train_class_weight_balanced(tmp_path: Path) -> None:
    data_path = tmp_path / "state_role.jsonl"
    _write_state_role_jsonl_imbalanced(data_path)
    run_root = tmp_path / "runs"
    config = TrainConfig(
        data_path=data_path,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        seed=7,
        val_split=0.2,
        run_root=run_root,
        class_weight="balanced",
    )
    run_dir = train_proposal_model(config)
    metrics = _load_metrics(run_dir)
    assert metrics["class_weight_mode"] == "balanced"
    assert metrics["class_weight_clip"] == 5.0
    weights = metrics["class_weights"]
    assert isinstance(weights, list)
    assert len(weights) == 8
    assert max(weights) <= 5.0
    train_dist = metrics["train_class_distribution"]
    assert isinstance(train_dist, list)
    assert len(train_dist) == 8
    assert sum(train_dist) == metrics["num_train"]


def test_train_class_weight_none(tmp_path: Path) -> None:
    data_path = tmp_path / "state_role.jsonl"
    _write_state_role_jsonl_imbalanced(data_path)
    run_root = tmp_path / "runs"
    config = TrainConfig(
        data_path=data_path,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        seed=7,
        val_split=0.2,
        run_root=run_root,
        class_weight="none",
    )
    run_dir = train_proposal_model(config)
    metrics = _load_metrics(run_dir)
    assert metrics["class_weight_mode"] == "none"
    assert "class_weights" not in metrics
    train_dist = metrics["train_class_distribution"]
    assert isinstance(train_dist, list)
    assert len(train_dist) == 8
    assert sum(train_dist) == metrics["num_train"]
