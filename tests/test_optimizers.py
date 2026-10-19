"""Optimizers against independent expectations: a hand-written SGD update, ES gradient
estimates against autodiff, and every optimizer learning a separable toy problem."""

import jax
import jax.numpy as jnp
import msgspec
import pytest

from kiseki import optimizers
from kiseki.models.layers import Dense, Flatten, network
from kiseki.worker.optimizer import Problem, centered_ranks
from kiseki.worker.population import Scorer

NET = network("toy", (2, 2, 1), 2, [Flatten(), Dense(2)])


def toy_batch(key, count=256):
    """Class 1 when the first pixel exceeds the last: linearly separable."""
    x = jax.random.normal(key, (count, 2, 2, 1))
    y = (x[:, 0, 0, 0] > x[:, 1, 1, 0]).astype(jnp.int32)
    return x, y


def build(name, start=None, **params):
    params = msgspec.convert(params, optimizers.params_type(name))
    return optimizers.load_impl(name).build(params, Problem(NET, Scorer(NET, 64), 256, start))


def test_centered_ranks():
    assert centered_ranks(jnp.array([3.0, 1.0, 2.0])).tolist() == [0.5, -0.5, 0.0]


def test_sgd_step_matches_hand_written_update():
    opt = build("sgd", learning_rate=0.1, momentum=0.9, weight_decay=0.01)
    x, y = toy_batch(jax.random.key(0))
    state = opt.init(jax.random.key(1))
    w0 = state["weights"]

    def loss(flat):  # written out: dense layer and softmax cross-entropy
        w = flat[:8].reshape(4, 2)
        b = flat[8:]
        logits = x.reshape(-1, 4) @ w + b
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])

    g0 = jax.grad(loss)(w0) + 0.01 * w0
    state, metrics = opt.step(state, jax.random.key(2), x, y)
    assert jnp.allclose(state["weights"], w0 - 0.1 * g0, atol=1e-6)
    assert jnp.isclose(metrics["loss"], loss(w0), atol=1e-6)
    w1 = state["weights"]
    g1 = jax.grad(loss)(w1) + 0.01 * w1
    state, _ = opt.step(state, jax.random.key(3), x, y)
    assert jnp.allclose(state["weights"], w1 - 0.1 * (0.9 * g0 + g1), atol=1e-6)


def test_es_gradient_points_downhill():
    opt = build("openai_es", population_size=4000, sigma=0.02, learning_rate=0.01, weight_decay=0.0)
    x, y = toy_batch(jax.random.key(0))
    state = opt.init(jax.random.key(1))
    true = jax.grad(lambda f: Scorer(NET, 1).one(f, x, y)[0])(state["theta"])
    new, _ = opt.step(state, jax.random.key(2), x, y)
    moved = new["theta"] - state["theta"]  # Adam's first step: -lr · sign of the estimate
    agree = jnp.mean(jnp.sign(moved) == -jnp.sign(true))
    assert agree >= 0.8


def test_es_on_raw_losses_estimates_the_smoothed_gradient():
    sigma = 0.02
    opt = build(
        "openai_es",
        population_size=20000,
        sigma=sigma,
        learning_rate=1.0,
        weight_decay=0.0,
        shaping="none",
        update="sgd",
    )
    x, y = toy_batch(jax.random.key(0))
    state = opt.init(jax.random.key(1))
    true = jax.grad(lambda f: Scorer(NET, 1).one(f, x, y)[0])(state["theta"])
    new, metrics = opt.step(state, jax.random.key(2), x, y)
    estimate = state["theta"] - new["theta"]  # plain step at lr 1: the gradient estimate
    cosine = estimate @ true / (jnp.linalg.norm(estimate) * jnp.linalg.norm(true))
    assert cosine > 0.95
    assert jnp.allclose(jnp.linalg.norm(estimate), jnp.linalg.norm(true), rtol=0.1)
    assert 0 <= metrics["ties"] < 0.1  # float32 losses of 20k close networks can coincide


def test_es_on_raw_losses_gives_unusable_scores_no_weight():
    opt = build("openai_es", population_size=16, weight_decay=0.0, shaping="none", update="sgd")
    x, y = toy_batch(jax.random.key(0))
    state = opt.init(jax.random.key(1))
    state = state | {"theta": state["theta"] * 1e30}  # logits overflow: every score is unusable
    new, _ = opt.step(state, jax.random.key(2), x, y)
    assert jnp.array_equal(new["theta"], state["theta"])


def test_adam_with_l2_matches_its_first_step_by_hand():
    opt = build("adam", learning_rate=0.1, l2=0.5)
    x, y = toy_batch(jax.random.key(0))
    state = opt.init(jax.random.key(1))
    w0 = state["weights"]
    g = jax.grad(lambda f: Scorer(NET, 1).one(f, x, y)[0])(w0) + 0.5 * w0
    new, _ = opt.step(state, jax.random.key(2), x, y)
    # Adam's bias-corrected first step is -lr · g / (|g| + ε)
    assert jnp.allclose(new["weights"], w0 - 0.1 * g / (jnp.abs(g) + 1e-8), atol=1e-4)


@pytest.mark.parametrize(
    "distribution,p,kurtosis",
    [
        ("uniform", 1.0, 1.8),
        ("gaussian", 1.0, 3.0),
        ("laplace", 1.0, 6.0),
        ("uniform", 0.25, 7.2),
        ("gaussian", 0.25, 12.0),
        ("laplace", 0.25, 24.0),
    ],
)
def test_leea_mutation_distributions_match_in_variance_and_differ_in_shape(
    distribution, p, kurtosis
):
    opt = build(
        "leea",
        start=jnp.zeros(NET.size),
        population_size=8000,
        crossover_fraction=0.0,
        mutation_probability=p,
        initial_mutation_step=0.1,
        mutation_distribution=distribution,
    )
    x, y = toy_batch(jax.random.key(0))
    state, _ = opt.step(opt.init(jax.random.key(1)), jax.random.key(2), x, y)
    delta = state["population"].reshape(-1)  # every child mutated from the zero network
    variance = jnp.mean(delta**2)
    assert jnp.isclose(variance, p * 0.1**2 / 3, rtol=0.05)
    assert jnp.isclose(jnp.mean(delta**4) / variance**2, kurtosis, rtol=0.1)


@pytest.mark.parametrize(
    "name,params,steps",
    [
        ("sgd", {"learning_rate": 0.5}, 100),
        ("adam", {"learning_rate": 0.05}, 100),
        ("adam", {"learning_rate": 0.05, "weight_decay": 0.01}, 100),
        ("openai_es", {"population_size": 64}, 150),
        ("leea", {"population_size": 64}, 150),
        ("cosyne", {"population_size": 64}, 150),
    ],
)
def test_learns_a_separable_problem(name, params, steps):
    opt = build(name, **params)
    state = opt.init(jax.random.key(0))
    step = jax.jit(opt.step)
    for i in range(steps):
        x, y = toy_batch(jax.random.key(100 + i))
        state, _ = step(state, jax.random.key(i), x, y)
    x, y = toy_batch(jax.random.key(7), 2000)
    accuracy = Scorer(NET, 1).one(opt.current(state), x, y)[1] / 2000
    assert accuracy >= 0.9, f"{name}: {accuracy:.3f}"
