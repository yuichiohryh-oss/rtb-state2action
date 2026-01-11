from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


@dataclass
class LabelerConfig:
    crops_jsonl: Path
    output_jsonl: Path = Path("data/hand_labels.jsonl")
    class_names: Tuple[str, ...] = (
        "CARD_1",
        "CARD_2",
        "CARD_3",
        "CARD_4",
        "CARD_5",
        "CARD_6",
        "CARD_7",
        "CARD_8",
    )


def _load_paths(crops_jsonl: Path) -> List[str]:
    paths = []
    with crops_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            record = json.loads(line)
            paths.append(record["path"])
    return paths


def label_crops(config: LabelerConfig) -> None:
    paths = _load_paths(config.crops_jsonl)
    labels: Dict[str, str] = {}
    idx = 0

    while idx < len(paths):
        img_path = paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue
        display = img.copy()
        cv2.putText(
            display,
            f"{idx + 1}/{len(paths)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("hand_labeler", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("n"):
            idx += 1
            continue
        if key == ord("b"):
            idx = max(0, idx - 1)
            labels.pop(img_path, None)
            continue
        if ord("1") <= key <= ord("8"):
            class_idx = key - ord("1")
            if class_idx < len(config.class_names):
                labels[img_path] = config.class_names[class_idx]
            idx += 1
            continue

    cv2.destroyAllWindows()
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open("w", encoding="utf-8") as fp:
        for path, label in labels.items():
            fp.write(json.dumps({"path": path, "label": label}) + "\n")
