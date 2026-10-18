"""OpenAI-ES as one compiled step (Salimans, Ho, Chen, Sidor & Sutskever 2017).

For centre θ and n = N/2 noise vectors εᵢ ~ N(0, I):
1. score θ + σεᵢ and θ - σεᵢ on the minibatch;
2. utilities: centered ranks of -loss over all N scores (shaping "ranks"), or -loss
   itself (shaping "none");
3. gradient estimate of the expected utility: g = Σᵢ (u⁺ᵢ - u⁻ᵢ) εᵢ / N for ranks, and
   Σᵢ (u⁺ᵢ - u⁻ᵢ) εᵢ / (σN) for raw losses, an unbiased estimate of -∇E[L(θ + σε)];
4. Adam step (update "adam") or plain step θ -= η·(...) (update "sgd") on
   -g + weight_decay·θ.
The `ties` metric is the share of scores equal to another score, which ranks break by
position.
εᵢ is regenerated from fold_in(key, i) in both passes rather than stored, so memory stays
O(chunk·D) for any population size. The validated network is θ.
"""

import jax
import jax.numpy as jnp
from jax import lax

from kiseki.optimizers.openai_es import Params
from kiseki.worker.optimizer import Optimizer, Problem, centered_ranks, initial
from kiseki.worker.population import cross_entropy


def build(params: Params, problem: Problem) -> Optimizer:
    net = problem.net
    scorer = problem.scorer
    pairs = params.population_size // 2
    chunk = max(1, min(scorer.chunk // 2, pairs))
    rounds = -(-pairs // chunk)

    def noise(key, index):
        return jax.random.normal(jax.random.fold_in(key, index), (net.size,))

    def init(key):
        theta = initial(problem, key)
        zeros = jnp.zeros_like(theta)
        return {"theta": theta, "m": zeros, "v": zeros, "t": jnp.int32(0)}

    def step(state, key, x, y):
        theta = state["theta"]

        def score_pair(index):
            eps = params.sigma * noise(key, index)
            plus = scorer.one(theta + eps, x, y)
            minus = scorer.one(theta - eps, x, y)
            return plus[0], minus[0]

        loss_plus, loss_minus = lax.map(score_pair, jnp.arange(pairs), batch_size=chunk)
        scores = jnp.concatenate([loss_plus, loss_minus])
        scores = jnp.where(jnp.isfinite(scores), scores, jnp.inf)
        ordered = jnp.sort(scores)
        ties = jnp.mean(ordered[1:] == ordered[:-1])
        if params.shaping == "ranks":
            utility, scale = centered_ranks(-scores), 1.0
        else:
            utility, scale = -scores, 1.0 / params.sigma
        weight = utility[:pairs] - utility[pairs:]

        padding = rounds * chunk - pairs  # padded pairs weigh 0
        indices = jnp.pad(jnp.arange(pairs), (0, padding)).reshape(rounds, chunk)
        weights = jnp.pad(weight, (0, padding)).reshape(rounds, chunk)

        def add_noise(total, args):  # regenerate `chunk` perturbations, add their weighted sum
            round_indices, round_weights = args
            eps = jax.vmap(lambda i: noise(key, i))(round_indices)
            return total + round_weights @ eps, None

        total, _ = lax.scan(add_noise, jnp.zeros_like(theta), (indices, weights))
        grad = scale * total / params.population_size
        update = -grad + params.weight_decay * theta
        t = state["t"] + 1
        m = params.beta1 * state["m"] + (1 - params.beta1) * update
        v = params.beta2 * state["v"] + (1 - params.beta2) * update**2
        if params.update == "adam":
            m_hat = m / (1 - params.beta1**t)
            v_hat = v / (1 - params.beta2**t)
            new_theta = theta - params.learning_rate * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        else:
            new_theta = theta - params.learning_rate * update

        loss, correct = cross_entropy(net.apply(theta, x), y)
        metrics = {
            "loss": loss,
            "accuracy": correct / x.shape[0],
            "mean_loss": jnp.mean(jnp.where(jnp.isfinite(scores), scores, jnp.nan)),
            "grad_norm": jnp.linalg.norm(grad),
            "ties": ties,
        }
        return {"theta": new_theta, "m": m, "v": v, "t": t}, metrics

    return Optimizer(
        init=init,
        step=step,
        current=lambda state: state["theta"],
        extra_metrics=("mean_loss", "grad_norm", "ties"),
    )
