from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_proposal_model import EvalConfig, eval_proposal_model
from src.proposal_model import TrainConfig, train_proposal_model


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


def test_eval_proposal_model_metrics(tmp_path: Path) -> None:
    data_path = tmp_path / "state_role.jsonl"
    _write_state_role_jsonl(data_path)
    run_root = tmp_path / "runs"
    train_config = TrainConfig(
        data_path=data_path,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        seed=123,
        val_split=0.2,
        run_root=run_root,
    )
    run_dir = train_proposal_model(train_config)
    model_path = run_dir / "model.pt"

    eval_config = EvalConfig(
        data_path=data_path,
        model_path=model_path,
        topk=3,
        baseline="none",
        val_split=0.2,
        seed=123,
        split="val",
        out_path=None,
    )
    metrics = eval_proposal_model(eval_config)

    confusion = metrics["confusion_matrix"]
    assert len(confusion) == 8
    assert all(len(row) == 8 for row in confusion)
    assert 0.0 <= metrics["accuracy_top1"] <= 1.0
    assert 0.0 <= metrics["accuracy_topk"] <= 1.0
