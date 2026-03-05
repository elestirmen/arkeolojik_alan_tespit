"""Shared attention layers used by training and inference wrappers."""

from __future__ import annotations

import torch


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(in_channels // reduction, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, reduced, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(reduced, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(torch.nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, in_channels: int, reduction: int = 4):
        super().__init__()
        self.input_attention = CBAM(in_channels, reduction=reduction)
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_attention(x)
        return self.base_model(x)
