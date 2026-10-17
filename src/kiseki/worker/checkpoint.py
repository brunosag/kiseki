"""Pytrees and networks on disk as safetensors.

A resume checkpoint is the training carry flattened by key path into
state/step-<n>.safetensors; state/manifest.json names the committed one and is replaced
last, so a crash mid-write leaves the previous checkpoint in force. A saved network is
models/<name>.safetensors with one tensor, "weights", and string metadata describing it.
"""

import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import msgspec
import numpy as np
from safetensors.numpy import load_file, save_file

from kiseki.store import write_atomic


def flatten(tree: Any) -> dict[str, np.ndarray]:
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in leaves}


def unflatten(like: Any, arrays: dict[str, np.ndarray]) -> Any:
    leaves, treedef = jax.tree_util.tree_flatten_with_path(like)
    restored = []
    for path, leaf in leaves:
        value = arrays[jax.tree_util.keystr(path)]
        if value.shape != leaf.shape or value.dtype != leaf.dtype:
            raise ValueError(
                f"checkpoint {jax.tree_util.keystr(path)}: {value.shape} {value.dtype}"
                f" does not match {leaf.shape} {leaf.dtype}"
            )
        restored.append(value)
    return treedef.unflatten(restored)


def _save(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, path)


def save_network(path: Path, weights: np.ndarray, **metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(
        path, {"weights": np.asarray(weights, np.float32)}, {k: str(v) for k, v in metadata.items()}
    )


def load_network(path: Path) -> np.ndarray:
    return load_file(str(path))["weights"]


class Manifest(msgspec.Struct):
    step: int
    file: str
    chunk_size: int


def save_state(directory: Path, step: int, chunk_size: int, tensors: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"step-{step}.safetensors"
    _save(directory / name, tensors, None)
    write_atomic(directory / "manifest.json", msgspec.json.encode(Manifest(step, name, chunk_size)))
    for old in directory.glob("step-*.safetensors"):
        if old.name != name:
            old.unlink()


def load_manifest(directory: Path) -> Manifest | None:
    path = directory / "manifest.json"
    if not path.exists():
        return None
    return msgspec.json.decode(path.read_bytes(), type=Manifest)


def load_state(directory: Path, manifest: Manifest) -> dict[str, np.ndarray]:
    return load_file(str(directory / manifest.file))


class Writer:
    """Writes files on one background thread so the device keeps training meanwhile."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kiseki-writer")
        self._pending: list[Future[None]] = []

    def submit(self, function, *args, **kwargs) -> None:
        running = []
        for future in self._pending:
            if future.done():
                future.result()  # re-raises a failed write
            else:
                running.append(future)
        self._pending = [*running, self._pool.submit(function, *args, **kwargs)]

    def flush(self) -> None:
        for future in self._pending:
            future.result()
        self._pending.clear()

    def close(self) -> None:
        self.flush()
        self._pool.shutdown()
