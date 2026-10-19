"""Models against hand-written forward passes."""

import jax
import jax.numpy as jnp

from kiseki import models


def test_mlp_bottleneck_matches_its_layers_by_hand():
    net = models.load("mlp_bottleneck", (28, 28, 1), 10)
    assert net.size == (784 * 16 + 16) + (16 * 128 + 128) + (128 * 10 + 10)
    flat = net.init(jax.random.key(0))
    flat = flat + 0.1 * jax.random.normal(jax.random.key(1), flat.shape)  # nonzero biases
    w1, b1, w2, b2, w3, b3 = net.unflatten(flat)
    x = jax.random.normal(jax.random.key(2), (64, 28, 28, 1))
    hidden = jax.nn.relu((x.reshape(64, 784) @ w1 + b1) @ w2 + b2)  # no ReLU on the 16
    expected = hidden @ w3 + b3
    assert jnp.allclose(net.apply(flat, x), expected, atol=1e-4, rtol=1e-5)
