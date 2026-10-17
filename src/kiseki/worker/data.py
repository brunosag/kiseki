"""Datasets resident on the device, and minibatches drawn inside compiled code.

The train/validation split comes from `split_seed`, shared by every run on a dataset so
validation curves compare. Batches walk a fresh permutation of the training images each
epoch (seeded by the run), dropping the last partial batch, so a run is a pure function of
(seed, step) and resuming at any step reproduces the same batches.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from kiseki import datasets

Array = jax.Array


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["train_x", "train_y", "val_x", "val_y", "test_x", "test_y", "mean", "std"],
    meta_fields=["batch_size", "class_names"],
)
@dataclass(frozen=True)
class DeviceData:
    """A pytree, so compiled code takes it as an argument rather than baking it in."""

    train_x: Array  # (N, H, W, C) uint8; batches are normalized as they are drawn
    train_y: Array  # (N,) int32
    val_x: Array  # normalized float32
    val_y: Array
    test_x: Array  # normalized float32
    test_y: Array
    mean: Array  # (C,) float32
    std: Array
    batch_size: int
    class_names: tuple[str, ...]

    @property
    def batches_per_epoch(self) -> int:
        return self.train_x.shape[0] // self.batch_size

    def normalize(self, images: Array) -> Array:
        return (images.astype(jnp.float32) / 255.0 - self.mean) / self.std

    def order(self, key: Array, step: Array) -> Array:
        """The image order of the epoch containing `step`."""
        epoch = step // self.batches_per_epoch
        return jax.random.permutation(jax.random.fold_in(key, epoch), self.train_x.shape[0])

    def batch(self, order: Array, step: Array) -> tuple[Array, Array]:
        """The minibatch for a training step, as normalized images and labels."""
        position = step % self.batches_per_epoch
        index = jax.lax.dynamic_slice_in_dim(order, position * self.batch_size, self.batch_size)
        return self.normalize(self.train_x[index]), self.train_y[index]


def split_indices(train_size: int, validation_size: int, split_seed: int):
    order = np.random.default_rng(split_seed).permutation(train_size)
    return order[validation_size:], order[:validation_size]


def load(dataset: str, *, validation_size: int, split_seed: int, batch_size: int) -> DeviceData:
    splits = datasets.load(dataset)
    train_index, val_index = split_indices(splits.train_x.shape[0], validation_size, split_seed)
    train_x = np.asarray(splits.train_x)
    train_y = np.asarray(splits.train_y).astype(np.int32)
    put = jax.device_put

    def normalized(images: np.ndarray) -> Array:
        return put((images.astype(np.float32) / 255.0 - splits.mean) / splits.std)

    return DeviceData(
        train_x=put(train_x[train_index]),
        train_y=put(train_y[train_index]),
        val_x=normalized(train_x[val_index]),
        val_y=put(train_y[val_index]),
        test_x=normalized(np.asarray(splits.test_x)),
        test_y=put(np.asarray(splits.test_y).astype(np.int32)),
        mean=put(jnp.asarray(splits.mean)),
        std=put(jnp.asarray(splits.std)),
        batch_size=batch_size,
        class_names=splits.info.class_names,
    )
