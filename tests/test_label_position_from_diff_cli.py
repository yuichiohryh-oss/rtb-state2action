from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_label_position_from_diff_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "label_position_from_diff.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
