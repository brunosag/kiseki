"""CIFAR-10: 60k 32x32 colour images in 10 classes (Krizhevsky, 2009).

Parsed from the binary release, so no pickle is ever loaded.
"""

import tarfile
from pathlib import Path

import numpy as np

from kiseki.datasets.base import DatasetInfo, RawFile, RawSplits

INFO = DatasetInfo(
    name="cifar10",
    title="CIFAR-10",
    image_shape=(32, 32, 3),
    class_names=(
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ),
    train_size=50000,
    validation_size=5000,
    default_model="vgg_mini",
    homepage="https://www.cs.toronto.edu/~kriz/cifar.html",
    files=(
        RawFile(
            "cifar-10-binary.tar.gz",
            "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
            "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd",
        ),
    ),
)

_RECORD = 1 + 32 * 32 * 3


def parse(raw_dir: Path) -> RawSplits:
    batches: dict[str, np.ndarray] = {}
    with tarfile.open(raw_dir / "cifar-10-binary.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if member.isfile() and name.endswith(".bin") and "batch" in name:
                handle = archive.extractfile(member)
                assert handle is not None
                batches[name] = np.frombuffer(handle.read(), dtype=np.uint8).reshape(-1, _RECORD)
    train = np.concatenate([batches[f"data_batch_{i}.bin"] for i in range(1, 6)])
    test = batches["test_batch.bin"]
    return RawSplits(
        train_x=_images(train),
        train_y=train[:, 0].copy(),
        test_x=_images(test),
        test_y=test[:, 0].copy(),
    )


def _images(records: np.ndarray) -> np.ndarray:
    # Each record stores the red, green and blue planes in turn, row-major.
    return records[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
