from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_pos_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "train_pos_model.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert "--train-manifest" in stdout
    assert "--grid-w" in stdout
    assert "--grid-h" in stdout
    assert "--img-size" in stdout
