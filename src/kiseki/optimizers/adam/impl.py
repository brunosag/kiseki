"""AdamW on flat weights: g = ∇L (+ l2·θ), bias-corrected moments, then
θ -= η·(m̂/(√v̂ + ε) + λθ)."""

import jax
import jax.numpy as jnp

from kiseki.optimizers.adam import Params
from kiseki.worker.optimizer import Optimizer, Problem, initial
from kiseki.worker.population import cross_entropy


def build(params: Params, problem: Problem) -> Optimizer:
    net = problem.net

    def loss_fn(flat, x, y):
        return cross_entropy(net.apply(flat, x), y)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def init(key):
        flat = initial(problem, key)
        zeros = jnp.zeros_like(flat)
        return {"weights": flat, "m": zeros, "v": zeros, "t": jnp.int32(0)}

    def step(state, key, x, y):
        weights = state["weights"]
        (loss, correct), grad = grad_fn(weights, x, y)
        grad = grad + params.l2 * weights
        t = state["t"] + 1
        m = params.beta1 * state["m"] + (1 - params.beta1) * grad
        v = params.beta2 * state["v"] + (1 - params.beta2) * grad**2
        m_hat = m / (1 - params.beta1**t)
        v_hat = v / (1 - params.beta2**t)
        direction = m_hat / (jnp.sqrt(v_hat) + params.epsilon) + params.weight_decay * weights
        metrics = {
            "loss": loss,
            "accuracy": correct / x.shape[0],
            "grad_norm": jnp.linalg.norm(grad),
        }
        new = {"weights": weights - params.learning_rate * direction, "m": m, "v": v, "t": t}
        return new, metrics

    return Optimizer(
        init=init,
        step=step,
        current=lambda state: state["weights"],
        extra_metrics=("grad_norm",),
    )
