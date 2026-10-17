"""Scoring many networks on one minibatch: the hot path of every evolutionary optimizer.

`Scorer.score(flat_population, x, y)` returns each network's mean cross-entropy and its
number of correct predictions. Networks are vmapped `chunk` at a time (lax.map), which
bounds activation memory; `kiseki.worker.tuning` picks the chunk per machine.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from kiseki.models.layers import Network

Array = jax.Array


def cross_entropy(logits: Array, labels: Array) -> tuple[Array, Array]:
    """Mean cross-entropy and correct-prediction count for one network's logits."""
    log_z = jax.nn.logsumexp(logits, axis=-1)
    picked = jnp.take_along_axis(logits, labels[:, None], axis=-1)[:, 0]
    loss = jnp.mean(log_z - picked)
    correct = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
    return loss, correct


@dataclass(frozen=True)
class Scorer:
    net: Network
    chunk: int
    dtype: jnp.dtype = jnp.float32

    def one(self, flat: Array, x: Array, y: Array) -> tuple[Array, Array]:
        return cross_entropy(self.net.apply(flat, x, self.dtype), y)

    def score(self, population: Array, x: Array, y: Array) -> tuple[Array, Array]:
        """(losses, corrects) for a (N, size) population; non-finite losses become +inf."""
        losses, correct = lax.map(
            lambda flat: self.one(flat, x, y),
            population,
            batch_size=min(self.chunk, population.shape[0]),
        )
        return jnp.where(jnp.isfinite(losses), losses, jnp.inf), correct

    def score_chunks(self, chunks: Array, live: Array, x: Array, y: Array) -> tuple[Array, Array]:
        """Score (n, chunk, size) blocks, skipping blocks whose `live` flag is false.

        Skipped blocks return +inf losses; callers substitute known scores for them.
        """

        def block(args):
            networks, alive = args
            return lax.cond(
                alive,
                lambda: jax.vmap(lambda f: self.one(f, x, y))(networks),
                lambda: (
                    jnp.full(networks.shape[0], jnp.inf, jnp.float32),
                    jnp.zeros(networks.shape[0], jnp.int32),
                ),
            )

        losses, correct = lax.map(block, (chunks, live))
        losses = losses.reshape(-1)
        return jnp.where(jnp.isfinite(losses), losses, jnp.inf), correct.reshape(-1)

    def evaluate(self, flat: Array, images: Array, labels: Array, batch: int = 2000):
        """Mean loss and accuracy of one network over a whole split (e.g. validation)."""
        count = images.shape[0]
        usable = count - count % batch if count > batch else count
        size = min(batch, count)
        xs = images[:usable].reshape(-1, size, *images.shape[1:])
        ys = labels[:usable].reshape(-1, size)

        def body(carry, xy):
            loss, correct = cross_entropy(self.net.apply(flat, xy[0]), xy[1])
            return (carry[0] + loss * size, carry[1] + correct), None

        (total, correct), _ = lax.scan(body, (jnp.float32(0), jnp.int32(0)), (xs, ys))
        if usable < count:
            loss, extra = cross_entropy(self.net.apply(flat, images[usable:]), labels[usable:])
            total, correct = total + loss * (count - usable), correct + extra
        return total / count, correct / count
