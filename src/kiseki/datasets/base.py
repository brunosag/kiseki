"""Dataset contract: raw files with pinned hashes, parsed once into uint8 arrays.

A dataset module defines `INFO` and `parse(raw_dir) -> RawSplits`. `load(name)` downloads,
verifies, parses and caches the result as .npy files that later loads memory-map.
Images are uint8 NHWC; labels are uint8. Nothing here imports JAX.
"""

import hashlib
import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import msgspec
import numpy as np

from kiseki import paths


class RawFile(msgspec.Struct, frozen=True):
    name: str
    url: str
    sha256: str


class DatasetInfo(msgspec.Struct, frozen=True):
    name: str
    title: str
    image_shape: tuple[int, int, int]  # height, width, channels
    class_names: tuple[str, ...]
    train_size: int  # images in the official training split
    validation_size: int  # held out from training for validation by default
    default_model: str
    homepage: str
    files: tuple[RawFile, ...]


@dataclass(frozen=True, slots=True)
class RawSplits:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


@dataclass(frozen=True, slots=True)
class Splits:
    info: DatasetInfo
    train_x: np.ndarray  # (N, H, W, C) uint8
    train_y: np.ndarray  # (N,) uint8
    test_x: np.ndarray
    test_y: np.ndarray
    mean: np.ndarray  # (C,) float32, over train pixels scaled to [0, 1]
    std: np.ndarray


_ARRAYS = ("train_x", "train_y", "test_x", "test_y")


def cached(info: DatasetInfo, root: Path | None = None) -> bool:
    directory = (root or paths.datasets_dir()) / info.name
    return (directory / "stats.json").exists()


def load_splits(info: DatasetInfo, parse, root: Path | None = None) -> Splits:
    directory = (root or paths.datasets_dir()) / info.name
    stats_path = directory / "stats.json"
    if not stats_path.exists():
        raw_dir = directory / "raw"
        for raw in info.files:
            fetch(raw, raw_dir)
        raw_splits = parse(raw_dir)
        _check_shapes(info, raw_splits)
        for name in _ARRAYS:
            np.save(directory / f"{name}.npy", getattr(raw_splits, name))
        scaled = raw_splits.train_x.reshape(-1, info.image_shape[2]).astype(np.float64) / 255.0
        stats = {"mean": scaled.mean(axis=0).tolist(), "std": scaled.std(axis=0).tolist()}
        stats_path.write_text(json.dumps(stats))
        shutil.rmtree(raw_dir, ignore_errors=True)
    stats = json.loads(stats_path.read_text())
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in _ARRAYS}
    return Splits(
        info=info,
        mean=np.asarray(stats["mean"], dtype=np.float32),
        std=np.asarray(stats["std"], dtype=np.float32),
        **arrays,
    )


def fetch(raw: RawFile, directory: Path) -> Path:
    """Download one raw file unless a verified copy exists."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / raw.name
    if target.exists() and sha256(target) == raw.sha256:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    print(f"kiseki: downloading {raw.url}", file=sys.stderr)
    with urllib.request.urlopen(raw.url, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    digest = sha256(partial)
    if digest != raw.sha256:
        partial.unlink()
        raise ValueError(f"{raw.name}: sha256 {digest} does not match pinned {raw.sha256}")
    partial.replace(target)
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_shapes(info: DatasetInfo, splits: RawSplits) -> None:
    if splits.train_x.shape[0] != info.train_size:
        raise ValueError(f"{info.name}: expected {info.train_size} training images")
    for x, y in ((splits.train_x, splits.train_y), (splits.test_x, splits.test_y)):
        if x.dtype != np.uint8 or y.dtype != np.uint8:
            raise ValueError(f"{info.name}: arrays must be uint8")
        if x.shape[1:] != info.image_shape or x.shape[0] != y.shape[0]:
            raise ValueError(f"{info.name}: unexpected shapes {x.shape} / {y.shape}")
        if int(y.max()) >= len(info.class_names):
            raise ValueError(f"{info.name}: label out of range")
