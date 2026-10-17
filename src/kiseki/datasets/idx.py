"""Parser for the gzip'd IDX files used by MNIST, Fashion-MNIST and KMNIST."""

import gzip
from pathlib import Path

import numpy as np

from kiseki.datasets.base import RawSplits

_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def read_idx(path: Path) -> np.ndarray:
    data = gzip.decompress(path.read_bytes())
    if data[0] != 0 or data[1] != 0 or data[2] != 0x08:
        raise ValueError(f"{path.name}: not an unsigned-byte IDX file")
    rank = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(rank))
    array = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * rank)
    return array.reshape(shape).copy()


def parse_idx_dir(raw_dir: Path, prefix: str = "") -> RawSplits:
    arrays = {key: read_idx(raw_dir / f"{prefix}{name}") for key, name in _FILES.items()}
    return RawSplits(
        train_x=arrays["train_x"][..., None],
        train_y=arrays["train_y"],
        test_x=arrays["test_x"][..., None],
        test_y=arrays["test_y"],
    )


def idx_file_names() -> tuple[str, ...]:
    return tuple(_FILES.values())
