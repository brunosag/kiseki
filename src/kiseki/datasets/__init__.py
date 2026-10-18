"""Dataset catalog. Add a dataset as one module here, then list it in `NAMES`.

A module defines `INFO: DatasetInfo` (pinned raw files, image shape, classes, default
model) and `parse(raw_dir) -> RawSplits`. See `mnist.py` for the smallest example.
"""

import importlib

from kiseki.datasets.base import DatasetInfo, Splits, cached, load_splits

NAMES = ("mnist", "fashion_mnist", "kmnist", "cifar10")


def _module(name: str):
    if name not in NAMES:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(NAMES)}")
    return importlib.import_module(f"kiseki.datasets.{name}")


def info(name: str) -> DatasetInfo:
    return _module(name).INFO


def load(name: str) -> Splits:
    module = _module(name)
    return load_splits(module.INFO, module.parse)


def is_cached(name: str) -> bool:
    return cached(info(name))


__all__ = ["NAMES", "DatasetInfo", "Splits", "info", "is_cached", "load"]
