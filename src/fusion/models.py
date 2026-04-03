from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EmbeddingMLPConfig:
    input_dim: int = 2144
    hidden_dim_1: int = 512
    hidden_dim_2: int = 128
    dropout: float = 0.2
    num_classes: int = 8


class EmbeddingMLP(nn.Module):
    def __init__(self, config: EmbeddingMLPConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim_1, config.hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim_2, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
