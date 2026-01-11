from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.capture.frame_source import FrameSource, VideoFrameSource, WindowFrameSource
from src.capture.roi import RoiParams, compute_hand_roi
from src.constants.cards_26hog import CARD_ORDER
from src.hand.model import build_model
from src.hand.frame_slots import iter_hand_slots
from src.state.state_writer import append_state


@dataclass
class InferConfig:
    window_title: str | None
    model_path: Path
    interval_ms: int = 250
    video_path: Path | None = None
    video_fps: float = 4.0
    max_frames: int | None = None
    history: int = 3
    smoothing: str = "majority"
    roi_params: RoiParams = RoiParams()
    state_out: Path | None = None
    image_size: int = 96


def _load_model(model_path: Path, device: torch.device):
    payload = torch.load(model_path, map_location=device)
    class_names = payload.get("class_names", [f"CARD_{i}" for i in range(1, 9)])
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, class_names


def _preprocess(image: np.ndarray, image_size: int) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    pil = Image.fromarray(image[:, :, ::-1])
    return tf(pil)


def _stable_prediction(history: List[Tuple[int, float]], smoothing: str) -> int:
    if not history:
        return -1
    if smoothing == "max":
        return max(history, key=lambda x: x[1])[0]
    counts: Dict[int, int] = {}
    for idx, _ in history:
        counts[idx] = counts.get(idx, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]


def build_in_hand(slot_preds: List[int], class_names: List[str]) -> Dict[str, int]:
    in_hand = {name: 0 for name in class_names}
    for idx in slot_preds:
        if 0 <= idx < len(class_names):
            in_hand[class_names[idx]] = 1
    return in_hand


def _card_sort_key(name: str) -> Tuple[int, int | str]:
    if name.startswith("CARD_"):
        suffix = name[5:]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, name)


def _order_in_hand(in_hand: Dict[str, int], class_names: List[str]) -> Dict[str, int]:
    ordered: Dict[str, int] = {}
    if set(class_names) == set(CARD_ORDER):
        ordered_names = [name for name in CARD_ORDER if name in class_names]
    else:
        ordered_names = sorted(class_names, key=_card_sort_key)
    for name in ordered_names:
        ordered[name] = in_hand.get(name, 0)
    return ordered


def _log_start(config: InferConfig, window_w: int, window_h: int) -> None:
    roi = compute_hand_roi(window_w, window_h, config.roi_params)
    print(
        "infer_hand start:",
        f"window_rect=({window_w},{window_h})",
        "HAND_ROI="
        f"y_ratio={config.roi_params.y_ratio:.3f},"
        f"height_ratio={config.roi_params.height_ratio:.3f},"
        f"x_margin_ratio={config.roi_params.x_margin_ratio:.3f},"
        f"x_offset_ratio={config.roi_params.x_offset_ratio:.3f}",
        f"roi_px={roi}",
        f"image_size={config.image_size}",
        f"smoothing={config.smoothing}",
        sep=" ",
    )


def infer_loop(config: InferConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = _load_model(config.model_path, device)
    frame_source = _build_frame_source(config)

    history: List[List[Tuple[int, float]]] = [[] for _ in range(4)]
    started = False

    for frame_slots in iter_hand_slots(frame_source, config.roi_params, config.max_frames):
        if not started:
            window_h, window_w = frame_slots.frame.shape[:2]
            _log_start(config, window_w, window_h)
            started = True
        slot_preds: List[int] = []

        for slot_idx, (x1, y1, x2, y2) in enumerate(frame_slots.slots):
            crop = frame_slots.frame[y1:y2, x1:x2]
            tensor = _preprocess(crop, config.image_size).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            history[slot_idx].append((pred_idx, float(probs[pred_idx])))
            history[slot_idx] = history[slot_idx][-config.history :]
            stable = _stable_prediction(history[slot_idx], config.smoothing)
            slot_preds.append(stable)

        in_hand = build_in_hand(slot_preds, class_names)
        ordered_in_hand = _order_in_hand(in_hand, class_names)
        state = {"t_ms": frame_slots.t_ms, "in_hand": ordered_in_hand}
        if config.state_out is not None:
            append_state(config.state_out, state)
        print(json.dumps(state))


def _build_frame_source(config: InferConfig) -> FrameSource:
    if config.video_path is not None:
        return VideoFrameSource(config.video_path, video_fps=config.video_fps)
    if not config.window_title:
        raise ValueError("window_title is required when video_path is not set")
    return WindowFrameSource(config.window_title, interval_ms=config.interval_ms, prefer_dxcam=True)
