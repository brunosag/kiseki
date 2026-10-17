"""Sequential networks as data. A `Network` is built from a layer list; everything a run or
an analysis needs (parameter layout, init, forward pass, activation taps, LRP) derives from
that list, so adding a model is a short declaration (see `small_cnn.py`).

A network's weights travel as one flat float32 vector of length `size`, which keeps
populations a plain (N, size) matrix. Images are NHWC; kernels are HWIO.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array


@dataclass(frozen=True)
class Conv:
    features: int
    kernel: int
    padding: str = "VALID"  # or "SAME"


@dataclass(frozen=True)
class Dense:
    features: int


@dataclass(frozen=True)
class ReLU:
    pass


@dataclass(frozen=True)
class MaxPool:
    window: int


@dataclass(frozen=True)
class AvgPool:
    window: int


@dataclass(frozen=True)
class Flatten:
    pass


Layer = Conv | Dense | ReLU | MaxPool | AvgPool | Flatten


@dataclass(frozen=True)
class Slot:
    """Where one weight tensor lives in the flat vector."""

    name: str
    shape: tuple[int, ...]
    offset: int
    fan_in: int
    gain: float

    @property
    def size(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class Network:
    name: str
    input_shape: tuple[int, int, int]
    layers: tuple[Layer, ...]
    slots: tuple[Slot, ...]
    names: tuple[str, ...]  # one per layer: conv1, relu1, pool1, flatten, dense1, ...
    size: int
    classes: int
    _param_index: dict[int, tuple[int, int]] = field(repr=False)  # layer -> (w slot, b slot)

    @property
    def penultimate(self) -> str:
        """Name of the activation that feeds the final dense layer."""
        return self.names[len(self.layers) - 2]

    def init(self, key: Array) -> Array:
        """Kaiming-normal weights (ReLU gain for hidden layers, 1 for the output), zero biases."""
        parts = []
        keys = jax.random.split(key, len(self.slots))
        for slot, slot_key in zip(self.slots, keys, strict=True):
            if slot.name.endswith(".b"):
                parts.append(jnp.zeros(slot.size, jnp.float32))
            else:
                std = slot.gain / math.sqrt(slot.fan_in)
                parts.append(std * jax.random.normal(slot_key, (slot.size,), jnp.float32))
        return jnp.concatenate(parts)

    def unflatten(self, flat: Array) -> list[Array]:
        return [
            lax.dynamic_slice_in_dim(flat, slot.offset, slot.size).reshape(slot.shape)
            for slot in self.slots
        ]

    def apply(self, flat: Array, x: Array, dtype=jnp.float32) -> Array:
        """Logits (B, classes) in float32; `dtype` sets the arithmetic of the forward pass."""
        return self.forward(flat, x, dtype=dtype)[0]

    def forward(
        self, flat: Array, x: Array, *, dtype=jnp.float32, taps: Sequence[str] = ()
    ) -> tuple[Array, dict[str, Array]]:
        tensors = [t.astype(dtype) for t in self.unflatten(flat)]
        h = x.astype(dtype)
        captured: dict[str, Array] = {}
        for index, layer in enumerate(self.layers):
            params = self._param_index.get(index)
            weights = (tensors[params[0]], tensors[params[1]]) if params else None
            h = apply_layer(layer, h, weights)
            if self.names[index] in taps:
                captured[self.names[index]] = h.astype(jnp.float32)
        return h.astype(jnp.float32), captured

    def layer_weights(self, flat: Array, index: int) -> tuple[Array, Array] | None:
        params = self._param_index.get(index)
        if params is None:
            return None
        tensors = self.unflatten(flat)
        return tensors[params[0]], tensors[params[1]]


def apply_layer(layer: Layer, h: Array, weights: tuple[Array, Array] | None) -> Array:
    match layer:
        case Conv():
            assert weights is not None
            w, b = weights
            out = lax.conv_general_dilated(
                h, w, (1, 1), layer.padding, dimension_numbers=("NHWC", "HWIO", "NHWC")
            )
            return out + b
        case Dense():
            assert weights is not None
            w, b = weights
            return h @ w + b
        case ReLU():
            return jax.nn.relu(h)
        case MaxPool():
            k = layer.window
            # A Python-scalar identity lets JAX recognize the max monoid, which has a gradient.
            return lax.reduce_window(h, -jnp.inf, lax.max, (1, k, k, 1), (1, k, k, 1), "VALID")
        case AvgPool():
            k = layer.window
            total = lax.reduce_window(h, 0.0, lax.add, (1, k, k, 1), (1, k, k, 1), "VALID")
            return total / (k * k)
        case Flatten():
            return h.reshape(h.shape[0], -1)


def network(
    name: str, input_shape: tuple[int, int, int], classes: int, layers: Sequence[Layer]
) -> Network:
    """Lay out parameters and name layers for a sequential stack ending in Dense(classes)."""
    layers = tuple(layers)
    if not isinstance(layers[-1], Dense) or layers[-1].features != classes:
        raise ValueError(f"{name}: the last layer must be Dense({classes})")
    slots: list[Slot] = []
    names: list[str] = []
    param_index: dict[int, tuple[int, int]] = {}
    counts: dict[str, int] = {}
    shape: tuple[int, ...] = input_shape
    offset = 0
    for index, layer in enumerate(layers):
        kind = {Conv: "conv", Dense: "dense", ReLU: "relu", MaxPool: "pool", AvgPool: "pool",
                Flatten: "flatten"}[type(layer)]
        counts[kind] = counts.get(kind, 0) + 1
        layer_name = kind if kind == "flatten" else f"{kind}{counts[kind]}"
        names.append(layer_name)
        if isinstance(layer, Conv | Dense):
            if isinstance(layer, Conv):
                h, w, c = shape
                w_shape = (layer.kernel, layer.kernel, c, layer.features)
                fan_in = layer.kernel * layer.kernel * c
                if layer.padding == "VALID":
                    h, w = h - layer.kernel + 1, w - layer.kernel + 1
                shape = (h, w, layer.features)
            else:
                (fan_in,) = shape
                w_shape = (fan_in, layer.features)
                shape = (layer.features,)
            gain = 1.0 if index == len(layers) - 1 else math.sqrt(2.0)
            weight = Slot(f"{layer_name}.w", w_shape, offset, fan_in, gain)
            offset += weight.size
            bias = Slot(f"{layer_name}.b", (layer.features,), offset, fan_in, gain)
            offset += bias.size
            param_index[index] = (len(slots), len(slots) + 1)
            slots += [weight, bias]
        elif isinstance(layer, MaxPool | AvgPool):
            h, w, c = shape
            shape = (h // layer.window, w // layer.window, c)
        elif isinstance(layer, Flatten):
            shape = (math.prod(shape),)
    return Network(
        name=name, input_shape=input_shape, layers=layers, slots=tuple(slots),
        names=tuple(names), size=offset, classes=classes, _param_index=param_index,
    )


Builder = Callable[[tuple[int, int, int], int], Network]
