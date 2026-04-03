from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.common.ravdess import LABEL_ORDER
from src.models.early_fusion_common import AudioFeatureBackbone, VideoFeatureBackbone


@dataclass(frozen=True)
class EarlyFusionBilinearConfig:
    projection_dim: int = 256
    fusion_hidden_dim: int = 128
    rank: int = 4
    dropout: float = 0.3
    num_classes: int = len(LABEL_ORDER)


class LowRankBilinearFusion(nn.Module):
    def __init__(self, config: EarlyFusionBilinearConfig) -> None:
        super().__init__()
        self.audio_projections = nn.ModuleList(
            [nn.Linear(config.projection_dim, config.fusion_hidden_dim) for _ in range(config.rank)]
        )
        self.video_projections = nn.ModuleList(
            [nn.Linear(config.projection_dim, config.fusion_hidden_dim) for _ in range(config.rank)]
        )
        self.layer_norm = nn.LayerNorm(config.fusion_hidden_dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.head = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.num_classes),
        )

    def forward(self, audio_vector: torch.Tensor, video_vector: torch.Tensor) -> torch.Tensor:
        components = []
        for audio_projection, video_projection in zip(self.audio_projections, self.video_projections, strict=True):
            audio_hidden = F.gelu(audio_projection(audio_vector))
            video_hidden = F.gelu(video_projection(video_vector))
            components.append(audio_hidden * video_hidden)
        fused = torch.stack(components, dim=0).mean(dim=0)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        return self.head(fused)


class EarlyFusionBilinear(nn.Module):
    def __init__(self, config: EarlyFusionBilinearConfig = EarlyFusionBilinearConfig()) -> None:
        super().__init__()
        self.audio_backbone = AudioFeatureBackbone()
        self.video_backbone = VideoFeatureBackbone()
        self.audio_projection = nn.Linear(self.audio_backbone.output_channels, config.projection_dim)
        self.video_projection = nn.Linear(self.video_backbone.output_channels, config.projection_dim)
        self.fusion = LowRankBilinearFusion(config)

    def forward(self, audio_input: torch.Tensor, video_input: torch.Tensor) -> torch.Tensor:
        audio_vector = F.adaptive_avg_pool2d(self.audio_backbone(audio_input), output_size=(1, 1)).flatten(1)
        video_maps = self.video_backbone(video_input).mean(dim=1)
        video_vector = F.adaptive_avg_pool2d(video_maps, output_size=(1, 1)).flatten(1)
        audio_vector = self.audio_projection(audio_vector)
        video_vector = self.video_projection(video_vector)
        return self.fusion(audio_vector, video_vector)
