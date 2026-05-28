"""WideResNet architecture used across IAM experiments."""
from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn


class BasicUnit(nn.Module):
    """Residual unit retaining the input channels via two 3x3 convolutions."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            OrderedDict(
                [
                    ("bn1", nn.BatchNorm2d(channels)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("conv1", nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)),
                    ("bn2", nn.BatchNorm2d(channels)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("dropout", nn.Dropout(dropout, inplace=False)),
                    ("conv2", nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)),
                ]
            )
        )

    def forward(self, x):
        return x + self.block(x)


class DownsampleUnit(nn.Module):
    """Residual unit that changes spatial resolution and channel width."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.norm_act = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(in_channels)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.block = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)),
                    ("bn1", nn.BatchNorm2d(out_channels)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("dropout", nn.Dropout(dropout, inplace=False)),
                    ("conv2", nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)),
                ]
            )
        )
        self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)


class Block(nn.Module):
    """Stack of an initial downsample followed by several residual units."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout),
            *[BasicUnit(out_channels, dropout) for _ in range(depth)],
        )

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    """WideResNet backbone used for CIFAR-style experiments."""

    def __init__(
        self,
        *,
        depth: int,
        width_factor: int,
        dropout: float,
        in_channels: int,
        labels: int,
    ) -> None:
        super().__init__()

        self.filters = [16, 16 * width_factor, 32 * width_factor, 64 * width_factor]
        self.block_depth = (depth - 4) // (3 * 2)

        self.network = nn.Sequential(
            OrderedDict(
                [
                    ("conv_in", nn.Conv2d(in_channels, self.filters[0], 3, stride=1, padding=1, bias=False)),
                    ("block1", Block(self.filters[0], self.filters[1], 1, self.block_depth, dropout)),
                    ("block2", Block(self.filters[1], self.filters[2], 2, self.block_depth, dropout)),
                    ("block3", Block(self.filters[2], self.filters[3], 2, self.block_depth, dropout)),
                    ("bn", nn.BatchNorm2d(self.filters[3])),
                    ("relu", nn.ReLU(inplace=True)),
                    ("pool", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten", nn.Flatten()),
                    ("head", nn.Linear(self.filters[3], labels)),
                ]
            )
        )

        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        return self.network(x)


__all__ = ["WideResNet"]
