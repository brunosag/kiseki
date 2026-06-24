from pathlib import Path

import torch
import pytest
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    WeightedRandomSampler,
)

from kiseki.data import deterministic_batch_stream, load_mnist, load_mnist_test


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
