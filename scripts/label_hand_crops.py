from __future__ import annotations

import argparse
from pathlib import Path

from src.hand.labeler import LabelerConfig, label_crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Label hand crop images.")
    parser.add_argument("--crops-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/hand_labels.jsonl"))
    args = parser.parse_args()

    config = LabelerConfig(crops_jsonl=args.crops_jsonl, output_jsonl=args.output_jsonl)
    label_crops(config)


if __name__ == "__main__":
    main()
