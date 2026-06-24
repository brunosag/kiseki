from collections.abc import Iterable
from typing import Literal

import torch
from captum.attr._utils.custom_modules import Addition_Module
from torch import nn
from torch.nn import functional as F

from .dataset_types import DatasetName

ActivationName = Literal["fc1_relu", "final_hidden"]


class CNN2C2DMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 10)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d | nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.final_hidden(x))

    def final_hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.relu3(self.fc1(x))

    def named_activations(
        self,
        x: torch.Tensor,
        names: Iterable[ActivationName] = ("fc1_relu",),
    ) -> dict[ActivationName, torch.Tensor]:
        requested = set(names)
        activations: dict[ActivationName, torch.Tensor] = {}

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        if "fc1_relu" in requested:
            activations["fc1_relu"] = x
        if "final_hidden" in requested:
            activations["final_hidden"] = x

        return activations


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = cifar_batch_norm(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = cifar_batch_norm(out_channels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.add = Addition_Module()
        self.stride = stride
        self.channel_padding = out_channels - in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(self.add(out, residual))

    def shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1 and self.channel_padding == 0:
            return x

        out = x[:, :, :: self.stride, :: self.stride]
        if self.channel_padding == 0:
            return out

        pad_before = self.channel_padding // 2
        pad_after = self.channel_padding - pad_before
        return F.pad(out, (0, 0, 0, 0, pad_before, pad_after))


class CIFARResNet20(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = cifar_batch_norm(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(16, block_count=3, stride=1)
        self.layer2 = self._make_layer(32, block_count=3, stride=2)
        self.layer3 = self._make_layer(64, block_count=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        self.reset_parameters()

    def _make_layer(self, out_channels: int, *, block_count: int, stride: int) -> nn.Sequential:
        strides = [stride, *([1] * (block_count - 1))]
        blocks = []
        for block_stride in strides:
            blocks.append(CIFARBasicBlock(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels * CIFARBasicBlock.expansion
        return nn.Sequential(*blocks)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.final_hidden(x))

    def final_hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        return torch.flatten(x, 1)

    def named_activations(
        self,
        x: torch.Tensor,
        names: Iterable[ActivationName] = ("final_hidden",),
    ) -> dict[ActivationName, torch.Tensor]:
        requested = set(names)
        activations: dict[ActivationName, torch.Tensor] = {}
        hidden = self.final_hidden(x)
        if "final_hidden" in requested:
            activations["final_hidden"] = hidden
        return activations


def cifar_batch_norm(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, track_running_stats=False)


def build_model(dataset: DatasetName) -> nn.Module:
    if dataset == "mnist":
        return CNN2C2DMNIST()
    if dataset == "cifar10":
        return CIFARResNet20()
    raise ValueError(f"Unsupported dataset: {dataset}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
