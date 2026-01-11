from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from src.constants.cards_26hog import CARD_LABELS

@dataclass
class LabelerConfig:
    crops_jsonl: Path
    output_jsonl: Path = Path("data/hand_labels.jsonl")
    show_help: bool = True
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


def _build_help_lines() -> Tuple[str, ...]:
    lines = [
        "1-8 : assign card",
        "n   : skip",
        "b   : back",
        "q   : quit",
        "",
    ]
    for idx in sorted(CARD_LABELS):
        lines.append(f"{idx} {CARD_LABELS[idx]}")
    return tuple(lines)


_HELP_LINES = _build_help_lines()


def _draw_help_overlay(image) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 1
    outline_thickness = 3
    x = 10
    y = 55
    (_, text_height), baseline = cv2.getTextSize("Ag", font, font_scale, text_thickness)
    line_height = text_height + baseline + 6

    for line in _HELP_LINES:
        if line:
            cv2.putText(
                image,
                line,
                (x, y),
                font,
                font_scale,
                (0, 0, 0),
                outline_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                line,
                (x, y),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
        y += line_height


def _load_paths(crops_jsonl: Path) -> List[str]:
    paths = []
    with crops_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            record = json.loads(line)
            paths.append(record["path"])
    return paths


def _load_existing_labels(output_jsonl: Path) -> Dict[str, str]:
    if not output_jsonl.exists():
        return {}
    labels: Dict[str, str] = {}
    with output_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            record = json.loads(line)
            labels[record["path"]] = record["label"]
    return labels


def label_crops(config: LabelerConfig) -> None:
    paths = _load_paths(config.crops_jsonl)
    labels = _load_existing_labels(config.output_jsonl)
    target_paths = [path for path in paths if path not in labels]
    if not target_paths:
        print("No unlabeled crops found. Nothing to do.")
        return
    idx = 0

    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open("a", encoding="utf-8") as fp:
        while idx < len(target_paths):
            img_path = target_paths[idx]
            img = cv2.imread(img_path)
            if img is None:
                idx += 1
                continue
            display = img.copy()
            cv2.putText(
                display,
                f"{idx + 1}/{len(target_paths)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if config.show_help:
                _draw_help_overlay(display)
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
                    fp.write(json.dumps({"path": img_path, "label": labels[img_path]}) + "\n")
                    fp.flush()
                idx += 1
                continue

    cv2.destroyAllWindows()
