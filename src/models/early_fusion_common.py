from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, pool: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


@dataclass(frozen=True)
class BackboneConfig:
    final_channels: int = 128


class AudioFeatureBackbone(nn.Module):
    def __init__(self, config: BackboneConfig = BackboneConfig()) -> None:
        super().__init__()
        self.output_channels = config.final_channels
        self.network = nn.Sequential(
            ConvBnRelu(1, 32, pool=True),
            ConvBnRelu(32, 64, pool=True),
            ConvBnRelu(64, config.final_channels, pool=False),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.network(audio)


class VideoFeatureBackbone(nn.Module):
    def __init__(self, config: BackboneConfig = BackboneConfig()) -> None:
        super().__init__()
        self.output_channels = config.final_channels
        self.network = nn.Sequential(
            ConvBnRelu(3, 32, pool=True),
            ConvBnRelu(32, 64, pool=True),
            ConvBnRelu(64, config.final_channels, pool=False),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video tensor with shape (B, F, C, H, W), got {tuple(video.shape)}.")
        batch_size, num_frames, _, height, width = video.shape
        feature_maps = self.network(video.reshape(batch_size * num_frames, 3, height, width))
        return feature_maps.reshape(
            batch_size,
            num_frames,
            self.output_channels,
            feature_maps.shape[-2],
            feature_maps.shape[-1],
        )
