from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_capture_scrcpy_taps_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "capture_scrcpy_taps.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--adb" in result.stdout
    assert "--scrcpy" in result.stdout
