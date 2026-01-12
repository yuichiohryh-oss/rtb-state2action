from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


@dataclass(frozen=True)
class PosSample:
    img_path: str
    label: int
    conf: float
    grid_w: int
    grid_h: int


def load_manifest(path: Path) -> list[PosSample]:
    samples: list[PosSample] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse manifest JSON at line {line_no} in {path}"
                ) from exc
            try:
                samples.append(
                    PosSample(
                        img_path=str(record["img"]),
                        label=int(record["label"]),
                        conf=float(record.get("conf", 1.0)),
                        grid_w=int(record.get("grid_w", 0)),
                        grid_h=int(record.get("grid_h", 0)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return samples


def _read_image_cv2(path: Path) -> np.ndarray | None:
    if cv2 is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _read_image_pil(path: Path) -> np.ndarray | None:
    if Image is None:
        return None
    with Image.open(path) as img:
        img = img.convert("L")
        return np.array(img)


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    if Image is None:
        raise RuntimeError("PIL is required for resizing when OpenCV is unavailable")
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((size, size), resample=Image.BILINEAR)
    return np.array(pil_img)


def load_grayscale_image(path: str, img_size: int) -> np.ndarray:
    img_path = Path(path)
    image = _read_image_cv2(img_path)
    if image is None:
        image = _read_image_pil(img_path)
    if image is None:
        raise RuntimeError(f"Failed to load image: {img_path}")
    if img_size > 0:
        image = _resize_image(image, img_size)
    return image


class PosDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: Sequence[PosSample], img_size: int):
        self.samples = list(samples)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = load_grayscale_image(sample.img_path, self.img_size)
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(sample.label, dtype=torch.long)
        return tensor, label


def summarize_manifest(samples: Sequence[PosSample]) -> dict[str, Any]:
    if not samples:
        return {"count": 0}
    grid_w = samples[0].grid_w
    grid_h = samples[0].grid_h
    return {
        "count": len(samples),
        "grid_w": grid_w,
        "grid_h": grid_h,
    }
