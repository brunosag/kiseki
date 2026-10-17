"""Minibatch SGD with heavy-ball momentum and L2 weight decay, on flat weights."""

import jax
import jax.numpy as jnp

from kiseki.optimizers.sgd import Params
from kiseki.worker.optimizer import Optimizer, Problem
from kiseki.worker.population import cross_entropy


def build(params: Params, problem: Problem) -> Optimizer:
    net = problem.net

    def loss_fn(flat, x, y):
        loss, correct = cross_entropy(net.apply(flat, x), y)
        return loss, correct

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def init(key):
        flat = net.init(key)
        return {"weights": flat, "velocity": jnp.zeros_like(flat)}

    def step(state, key, x, y):
        (loss, correct), grad = grad_fn(state["weights"], x, y)
        grad = grad + params.weight_decay * state["weights"]
        velocity = params.momentum * state["velocity"] + grad
        weights = state["weights"] - params.learning_rate * velocity
        metrics = {"loss": loss, "accuracy": correct / x.shape[0],
                   "grad_norm": jnp.linalg.norm(grad)}
        return {"weights": weights, "velocity": velocity}, metrics

    return Optimizer(
        init=init, step=step, current=lambda state: state["weights"],
        extra_metrics=("grad_norm",),
    )
