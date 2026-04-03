from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.common.ravdess import LABEL_ORDER
from src.models.early_fusion_common import AudioFeatureBackbone, VideoFeatureBackbone


@dataclass(frozen=True)
class EarlyFusionXAttnConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 1
    ffn_dim: int = 256
    dropout: float = 0.3
    num_classes: int = len(LABEL_ORDER)


class CrossAttentionBlock(nn.Module):
    def __init__(self, *, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        attention_output, _ = self.attention(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            need_weights=False,
        )
        hidden = self.norm1(query_tokens + attention_output)
        return self.norm2(hidden + self.feed_forward(hidden))


class BidirectionalCrossAttentionLayer(nn.Module):
    def __init__(self, config: EarlyFusionXAttnConfig) -> None:
        super().__init__()
        self.audio_to_video = CrossAttentionBlock(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )
        self.video_to_audio = CrossAttentionBlock(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )

    def forward(self, audio_tokens: torch.Tensor, video_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        updated_audio = self.audio_to_video(audio_tokens, video_tokens)
        updated_video = self.video_to_audio(video_tokens, updated_audio)
        return updated_audio, updated_video


class EarlyFusionXAttn(nn.Module):
    def __init__(self, config: EarlyFusionXAttnConfig = EarlyFusionXAttnConfig()) -> None:
        super().__init__()
        self.audio_backbone = AudioFeatureBackbone()
        self.video_backbone = VideoFeatureBackbone()
        self.layers = nn.ModuleList([BidirectionalCrossAttentionLayer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, audio_input: torch.Tensor, video_input: torch.Tensor) -> torch.Tensor:
        audio_tokens = self._encode_audio_tokens(audio_input)
        video_tokens = self._encode_video_tokens(video_input)

        for layer in self.layers:
            audio_tokens, video_tokens = layer(audio_tokens, video_tokens)

        pooled = torch.cat([audio_tokens.mean(dim=1), video_tokens.mean(dim=1)], dim=1)
        return self.classifier(self.dropout(pooled))

    def _encode_audio_tokens(self, audio_input: torch.Tensor) -> torch.Tensor:
        audio_map = self.audio_backbone(audio_input)
        pooled = F.adaptive_avg_pool2d(audio_map, output_size=(4, 4))
        batch_size, channels, height, width = pooled.shape
        return pooled.view(batch_size, channels, height * width).transpose(1, 2).contiguous()

    def _encode_video_tokens(self, video_input: torch.Tensor) -> torch.Tensor:
        frame_maps = self.video_backbone(video_input)
        batch_size, num_frames, channels, _, _ = frame_maps.shape
        pooled = F.adaptive_avg_pool2d(
            frame_maps.reshape(batch_size * num_frames, channels, frame_maps.shape[-2], frame_maps.shape[-1]),
            output_size=(2, 2),
        )
        pooled = pooled.reshape(batch_size, num_frames, channels, 2, 2)
        tokens = pooled.flatten(start_dim=3).permute(0, 1, 3, 2).contiguous()
        return tokens.reshape(batch_size, num_frames * 4, channels)
