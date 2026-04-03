from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ReferenceResNet50Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 3),
            self._make_layer(128, 4, stride=2),
            self._make_layer(256, 6, stride=2),
            self._make_layer(512, 3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        pooled = self.pool(features)
        return torch.flatten(pooled, 1)


class ReferenceExpressionModel(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        backbone_model = ReferenceResNet50Backbone()
        self.backbone = backbone_model.backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_expression = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        pooled = torch.flatten(self.pool(features), 1)
        logits = self.head_expression(pooled)
        return pooled, logits


def load_reference_backbone(checkpoint_path: Path, *, device: torch.device) -> ReferenceResNet50Backbone:
    model = ReferenceResNet50Backbone()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.backbone.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_reference_expression_model(checkpoint_path: Path, *, device: torch.device) -> ReferenceExpressionModel:
    model = ReferenceExpressionModel(num_classes=8)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
