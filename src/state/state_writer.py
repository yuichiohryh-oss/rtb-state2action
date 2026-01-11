from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def append_state(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(state) + "\n")
