from pathlib import Path

import torch
import pytest
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, WeightedRandomSampler

from kiseki.data import deterministic_batch_stream, load_mnist


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
