from __future__ import annotations

import math

import torch
from torch import nn


def _round_channels(channels: int, width_mult: float) -> int:
    return max(8, int(math.ceil(channels * width_mult / 8.0)) * 8)


class PosCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, width_mult: float = 1.0) -> None:
        super().__init__()
        ch1 = _round_channels(16, width_mult)
        ch2 = _round_channels(32, width_mult)
        ch3 = _round_channels(64, width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, ch1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(ch2, ch3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(ch3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_pos_cnn(in_ch: int, num_classes: int, width_mult: float = 1.0) -> PosCNN:
    return PosCNN(in_ch=in_ch, num_classes=num_classes, width_mult=width_mult)
