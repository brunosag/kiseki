import jax
import jax.numpy as jnp
import numpy as np

from kiseki.analyses.agreement.impl import mcnemar_exact
from kiseki.analyses.embedding.impl import knn_accuracy, nearest, trustworthiness, tsne
from kiseki.analyses.loss_landscape.impl import direction
from kiseki.analyses.representations.impl import centered_gram, cka
from kiseki.models.layers import Conv, Dense, Flatten, ReLU, network


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # 1 vs 9 discordant: 2 · (1 + 10) / 1024
    assert np.isclose(mcnemar_exact(1, 9), 22 / 1024)


def test_cka_ignores_rotation_and_scale():
    x = jax.random.normal(jax.random.key(0), (200, 16))
    rotation, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(1), (16, 16)))
    other = jax.random.normal(jax.random.key(2), (200, 16))
    gram = centered_gram(x)
    assert jnp.isclose(cka(gram, centered_gram(3.0 * x @ rotation)), 1.0, atol=1e-4)
    assert cka(gram, centered_gram(other)) < 0.3


def test_tsne_separates_two_blobs():
    key_a, key_b = jax.random.split(jax.random.key(0))
    blobs = jnp.concatenate(
        [jax.random.normal(key_a, (150, 10)), jax.random.normal(key_b, (150, 10)) + 8.0]
    )
    labels = jnp.repeat(jnp.arange(2), 150)
    init = 1e-4 * jax.random.normal(jax.random.key(1), (300, 2))
    embedded = tsne(blobs, init, 20.0, 300)
    _, neighbours = nearest(embedded, 5)
    assert knn_accuracy(neighbours, labels) == 1.0
    assert trustworthiness(blobs, embedded, 5) > 0.9
    assert trustworthiness(blobs, blobs, 5) == 1.0


def test_directions_are_filter_normalized_and_spare_biases():
    net = network("t", (6, 6, 1), 3, [Conv(4, 3), ReLU(), Flatten(), Dense(3)])
    flat = net.init(jax.random.key(0))
    d = direction(net, flat, jax.random.key(1))
    for slot, w, v in zip(net.slots, net.unflatten(flat), net.unflatten(d), strict=True):
        if slot.name.endswith(".b"):
            assert not v.any()
            continue
        axes = tuple(range(w.ndim - 1))
        assert jnp.allclose(jnp.sqrt((v**2).sum(axes)), jnp.sqrt((w**2).sum(axes)), rtol=1e-5)
