import jax
import jax.numpy as jnp
import pytest

from kiseki import models
from kiseki.analyses.lrp.impl import relevance


@pytest.mark.parametrize("name", models.NAMES)
def test_relevance_is_conserved_without_biases(name):
    """With zero biases and ε → 0 every rule conserves relevance: Σ input = target logit."""
    net = models.load(name, (28, 28, 1), 10)
    flat = net.init(jax.random.key(0))  # biases start at zero
    x = jax.random.uniform(jax.random.key(1), (8, 28, 28, 1), minval=-0.42, maxval=2.82)
    targets = net.apply(flat, x).argmax(1)
    r, logits = relevance(
        net,
        flat,
        x,
        targets,
        gamma=0.25,
        epsilon=1e-9,
        low=jnp.float32(-0.42),
        high=jnp.float32(2.82),
    )
    assert jnp.allclose(r.sum((1, 2, 3)), logits, rtol=5e-3)
