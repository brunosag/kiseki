from pathlib import Path

import pytest
from torch.utils.data import RandomSampler, WeightedRandomSampler

from kiseki.data import load_mnist


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
