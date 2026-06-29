from pathlib import Path

import numpy as np
import torch
import pytest
from PIL import Image
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    WeightedRandomSampler,
)
from torchvision import transforms

from kiseki.data import (
    deterministic_batch_stream,
    load_cifar10,
    load_cifar10_test,
    load_fashion_mnist,
    load_fashion_mnist_test,
    load_mnist,
    load_mnist_test,
)


def test_mnist_loader_shape_and_regular_shuffle() -> None:
    data_dir = Path("data")
    if not (data_dir / "MNIST" / "raw" / "train-images-idx3-ubyte").exists():
        pytest.skip("MNIST data is not available locally")

    train_loader, val_loader = load_mnist(data_dir, batch_size=16, seed=123, download=False)
    inputs, targets = next(iter(train_loader))

    assert inputs.shape == (16, 1, 28, 28)
    assert targets.shape == (16,)
    assert isinstance(train_loader.sampler, RandomSampler)
    assert not isinstance(train_loader.sampler, WeightedRandomSampler)
    assert len(train_loader.dataset) == 50000
    assert len(val_loader.dataset) == 10000


def test_mnist_test_loader_is_fixed_batch_and_not_shuffled(tmp_path, monkeypatch) -> None:
    class FakeMNIST:
        def __init__(self, root, train: bool, download: bool) -> None:
            assert root == tmp_path
            assert train is False
            assert download is False
            self.data = torch.arange(20 * 28 * 28, dtype=torch.uint8).reshape(20, 28, 28)
            self.targets = torch.arange(20) % 10

    monkeypatch.setattr("kiseki.data.datasets.MNIST", FakeMNIST)

    first_loader = load_mnist_test(tmp_path, download=False)
    second_loader = load_mnist_test(tmp_path, download=False)
    first_inputs, first_targets = next(iter(first_loader))
    second_inputs, second_targets = next(iter(second_loader))

    assert first_loader.batch_size == 512
    assert isinstance(first_loader.sampler, SequentialSampler)
    assert first_inputs.shape == (20, 1, 28, 28)
    assert torch.equal(first_targets, torch.arange(20) % 10)
    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_targets, second_targets)


def test_fashion_mnist_loader_shape_and_regular_shuffle(tmp_path, monkeypatch) -> None:
    class FakeFashionMNIST:
        def __init__(self, root, train: bool, download: bool) -> None:
            assert root == tmp_path
            assert train is True
            assert download is False
            self.data = torch.zeros((60000, 28, 28), dtype=torch.uint8)
            self.targets = torch.arange(60000) % 10

    monkeypatch.setattr("kiseki.data.datasets.FashionMNIST", FakeFashionMNIST)

    train_loader, val_loader = load_fashion_mnist(
        tmp_path,
        batch_size=16,
        seed=123,
        download=False,
    )
    inputs, targets = next(iter(train_loader))

    assert inputs.shape == (16, 1, 28, 28)
    assert targets.shape == (16,)
    assert isinstance(train_loader.sampler, RandomSampler)
    assert not isinstance(train_loader.sampler, WeightedRandomSampler)
    assert len(train_loader.dataset) == 50000
    assert len(val_loader.dataset) == 10000


def test_fashion_mnist_test_loader_is_fixed_batch_and_not_shuffled(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeFashionMNIST:
        def __init__(self, root, train: bool, download: bool) -> None:
            assert root == tmp_path
            assert train is False
            assert download is False
            self.data = torch.zeros((20, 28, 28), dtype=torch.uint8)
            self.targets = torch.arange(20) % 10

    monkeypatch.setattr("kiseki.data.datasets.FashionMNIST", FakeFashionMNIST)

    first_loader = load_fashion_mnist_test(tmp_path, download=False)
    second_loader = load_fashion_mnist_test(tmp_path, download=False)
    first_inputs, first_targets = next(iter(first_loader))
    second_inputs, second_targets = next(iter(second_loader))

    assert first_loader.batch_size == 512
    assert isinstance(first_loader.sampler, SequentialSampler)
    assert first_inputs.shape == (20, 1, 28, 28)
    assert torch.equal(first_targets, torch.arange(20) % 10)
    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_targets, second_targets)


def test_cifar10_loader_split_shape_and_transforms(tmp_path, monkeypatch) -> None:
    class FakeCIFAR10:
        def __init__(self, root, train: bool, download: bool, transform=None) -> None:
            assert root == tmp_path
            assert train is True
            assert download is False
            self.transform = transform

        def __len__(self) -> int:
            return 50000

        def __getitem__(self, index: int):
            image = Image.fromarray(
                np.full((32, 32, 3), index % 256, dtype=np.uint8),
                mode="RGB",
            )
            if self.transform is not None:
                image = self.transform(image)
            return image, index % 10

    monkeypatch.setattr("kiseki.data.datasets.CIFAR10", FakeCIFAR10)

    train_loader, val_loader = load_cifar10(tmp_path, batch_size=16, seed=123, download=False)
    inputs, targets = next(iter(train_loader))

    assert inputs.shape == (16, 3, 32, 32)
    assert targets.shape == (16,)
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert len(train_loader.dataset) == 45000
    assert len(val_loader.dataset) == 5000

    train_transform = train_loader.dataset.dataset.transform
    val_transform = val_loader.dataset.dataset.transform
    assert [type(transform) for transform in train_transform.transforms] == [
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert [type(transform) for transform in val_transform.transforms] == [
        transforms.ToTensor,
        transforms.Normalize,
    ]


def test_cifar10_test_loader_is_fixed_batch_and_not_shuffled(tmp_path, monkeypatch) -> None:
    class FakeCIFAR10:
        def __init__(self, root, train: bool, download: bool, transform=None) -> None:
            assert root == tmp_path
            assert train is False
            assert download is False
            self.transform = transform

        def __len__(self) -> int:
            return 20

        def __getitem__(self, index: int):
            image = Image.fromarray(
                np.full((32, 32, 3), index % 256, dtype=np.uint8),
                mode="RGB",
            )
            if self.transform is not None:
                image = self.transform(image)
            return image, index % 10

    monkeypatch.setattr("kiseki.data.datasets.CIFAR10", FakeCIFAR10)

    loader = load_cifar10_test(tmp_path, download=False)
    inputs, targets = next(iter(loader))

    assert loader.batch_size == 512
    assert isinstance(loader.sampler, SequentialSampler)
    assert inputs.shape == (20, 3, 32, 32)
    assert torch.equal(targets, torch.arange(20) % 10)
    assert [type(transform) for transform in loader.dataset.transform.transforms] == [
        transforms.ToTensor,
        transforms.Normalize,
    ]


def test_deterministic_batch_stream_restores_exact_next_batch() -> None:
    dataset = TensorDataset(torch.arange(12).float().reshape(12, 1), torch.arange(12))
    train_loader = DataLoader(dataset, batch_size=3)
    val_loader = DataLoader(dataset, batch_size=3)

    uninterrupted = deterministic_batch_stream(train_loader, val_loader, batch_size=3, seed=9)
    restored = deterministic_batch_stream(train_loader, val_loader, batch_size=3, seed=9)

    next(uninterrupted)
    next(uninterrupted)
    restored.load_state_dict(uninterrupted.state_dict())

    expected_inputs, expected_targets = next(uninterrupted)
    actual_inputs, actual_targets = next(restored)

    assert torch.equal(actual_inputs, expected_inputs)
    assert torch.equal(actual_targets, expected_targets)
    assert restored.state_dict()["next_batch_offset"] == uninterrupted.state_dict()[
        "next_batch_offset"
    ]
