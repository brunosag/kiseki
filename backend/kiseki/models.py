from collections.abc import Iterable
from typing import Literal

import torch
from torch import nn

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


class CIFAR10CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.final_hidden(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)

    def final_hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)
        x = self.relu5(self.conv5(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        return self.relu6(self.fc1(x))

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


def build_model(dataset: DatasetName) -> nn.Module:
    if dataset == "mnist":
        return CNN2C2DMNIST()
    if dataset == "cifar10":
        return CIFAR10CNN()
    raise ValueError(f"Unsupported dataset: {dataset}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
