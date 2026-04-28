from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.common.ravdess import LABEL_ORDER
from src.models.early_fusion_common import AudioFeatureBackbone, ConvBnRelu, VideoFeatureBackbone


@dataclass(frozen=True)
class EarlyFusion3CNNConfig:
    dropout: float = 0.3
    num_classes: int = len(LABEL_ORDER)
    gating_enabled: bool = False
    gating_hidden_dim: int = 64
    gating_scale_floor: float = 0.5


class ModalityFusionGate(nn.Module):
    """2-modality gating for audio and video feature maps."""

    def __init__(self, channels: int, *, hidden_dim: int, scale_floor: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"Expected positive gating hidden_dim, got {hidden_dim}.")
        if scale_floor < 0.0:
            raise ValueError(f"Expected non-negative gating_scale_floor, got {scale_floor}.")
        self.scale_floor = scale_floor
        self.hidden = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.audio_head = nn.Linear(hidden_dim, channels)
        self.video_head = nn.Linear(hidden_dim, channels)

    def compute_scales(self, audio_map: torch.Tensor, video_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_vector = F.adaptive_avg_pool2d(audio_map, output_size=(1, 1)).flatten(1)
        video_vector = F.adaptive_avg_pool2d(video_map, output_size=(1, 1)).flatten(1)
        fused_context = self.hidden(torch.cat([audio_vector, video_vector], dim=1))
        audio_scale = (self.scale_floor + torch.sigmoid(self.audio_head(fused_context))).unsqueeze(-1).unsqueeze(-1)
        video_scale = (self.scale_floor + torch.sigmoid(self.video_head(fused_context))).unsqueeze(-1).unsqueeze(-1)
        return audio_scale, video_scale

    def forward(self, audio_map: torch.Tensor, video_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_scale, video_scale = self.compute_scales(audio_map, video_map)
        return audio_map * audio_scale, video_map * video_scale


class EarlyFusion3CNN(nn.Module):
    def __init__(self, config: EarlyFusion3CNNConfig = EarlyFusion3CNNConfig()) -> None:
        super().__init__()
        self.config = config
        self.audio_backbone = AudioFeatureBackbone()
        self.video_backbone = VideoFeatureBackbone()

        audio_ch = self.audio_backbone.output_channels
        video_ch = self.video_backbone.output_channels

        fusion_in_channels = audio_ch + video_ch
        self.fusion_gate = (
            ModalityFusionGate(
                audio_ch,
                hidden_dim=config.gating_hidden_dim,
                scale_floor=config.gating_scale_floor,
            )
            if config.gating_enabled
            else None
        )

        self.fusion_backbone = nn.Sequential(
            ConvBnRelu(fusion_in_channels, 256, pool=False),
            ConvBnRelu(256, 128, pool=False),
        )
        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(128, config.num_classes)

    def forward(
        self,
        audio_input: torch.Tensor,
        video_input: torch.Tensor,
    ) -> torch.Tensor:
        audio_map = F.adaptive_avg_pool2d(self.audio_backbone(audio_input), output_size=(8, 8))
        video_frame_maps = self.video_backbone(video_input)
        video_map = F.adaptive_avg_pool2d(video_frame_maps.mean(dim=1), output_size=(8, 8))

        if self.fusion_gate is not None:
            audio_map, video_map = self.fusion_gate(audio_map, video_map)
        fused_map = self.fusion_backbone(torch.cat([audio_map, video_map], dim=1))

        fused_vector = F.adaptive_avg_pool2d(fused_map, output_size=(1, 1)).flatten(1)
        return self.classifier(self.dropout(fused_vector))
