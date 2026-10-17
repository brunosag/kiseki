"""The interface every optimizer implements (`kiseki/optimizers/<name>/impl.py: build`).

An optimizer is four pure functions over a state pytree, all traced into one compiled
training block (`kiseki.worker.train`):

- init(key) -> state
- step(state, key, x, y) -> (state, metrics): one generation or one SGD update on a
  normalized minibatch. metrics must hold scalar "loss" and "accuracy" for the network
  `current` returns (as measured before the update), plus the names in `extra_metrics`.
- current(state) -> flat weights of the network to validate and save.
- on_validation(state, improved) -> state: lets schedules react to validation (LEEA).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from kiseki.models.layers import Network
from kiseki.worker.population import Scorer

Array = jax.Array
State = Any


@dataclass(frozen=True)
class Problem:
    net: Network
    scorer: Scorer
    batch_size: int


@dataclass(frozen=True)
class Optimizer:
    init: Callable[[Array], State]
    step: Callable[[State, Array, Array, Array], tuple[State, dict[str, Array]]]
    current: Callable[[State], Array]
    on_validation: Callable[[State, Array], State] = lambda state, improved: state
    extra_metrics: tuple[str, ...] = ()
    validation_metrics: Callable[[State], dict[str, Array]] = lambda state: {}


def centered_ranks(values: Array) -> Array:
    """Ranks scaled to [-0.5, 0.5]; the largest value gets 0.5. Ties break by position."""
    count = values.shape[0]
    order = jnp.argsort(values)
    ranks = jnp.empty(count, jnp.float32).at[order].set(jnp.arange(count, dtype=jnp.float32))
    return ranks / max(count - 1, 1) - 0.5


def init_population(net: Network, key: Array, size: int) -> Array:
    return jax.vmap(net.init)(jax.random.split(key, size))
