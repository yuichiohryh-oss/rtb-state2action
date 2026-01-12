import subprocess
import sys


def test_capture_scrcpy_mouse_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/capture_scrcpy_mouse.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "--scrcpy" in combined
    assert "--record-seconds" in combined
    assert "--out" in combined
