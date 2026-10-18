"""LEEA as one compiled generation (Morse & Stanley 2016).

Generation t, on minibatch B_t:
1. score every network: fitness f = 1 / (1 + loss), 0 for non-finite losses;
2. inheritance: each child adds (1 - λ) x its parent's previous fitness (the mean of both
   parents' for crossover children);
3. the top ρN by fitness form a roulette wheel weighted by fitness;
4. the first N(1 - crossover_fraction) children copy one parent and mutate each weight with
   probability p by U(-η, η); the rest take each weight from one of two distinct parents.
There is no elitism. The validated network is the generation's lowest-loss member. When
validation accuracy fails to improve `validation_patience` times running, η *= γ.
"""

import jax
import jax.numpy as jnp

from kiseki.optimizers.leea import Params
from kiseki.worker.optimizer import Optimizer, Problem, init_population


def build(params: Params, problem: Problem) -> Optimizer:
    size = params.population_size
    sexual = min(size, max(0, round(params.crossover_fraction * size)))
    asexual = size - sexual
    elite = min(size, max(2 if sexual else 1, round(size * params.retention_fraction)))
    keep = 1.0 - params.fitness_decay
    is_asexual = jnp.arange(size) < asexual

    def init(key):
        population = init_population(problem.net, key, size)
        return {
            "population": population,
            "fitness": jnp.zeros(size, jnp.float32),
            "parent_a": jnp.zeros(size, jnp.int32),
            "parent_b": jnp.zeros(size, jnp.int32),
            "first": jnp.array(True),
            "mutation_step": jnp.float32(params.initial_mutation_step),
            "patience": jnp.int32(0),
            "best": population[0],
        }

    def step(state, key, x, y):
        population = state["population"]
        losses, correct = problem.scorer.score(population, x, y)
        fitness = jnp.where(jnp.isfinite(losses), 1.0 / (1.0 + losses), 0.0)
        previous = state["fitness"]
        inherited = jnp.where(
            is_asexual,
            previous[state["parent_a"]],
            0.5 * (previous[state["parent_a"]] + previous[state["parent_b"]]),
        )
        fitness = fitness + jnp.where(state["first"], 0.0, keep * inherited)

        best = jnp.argmin(losses)
        wheel_fitness, wheel = jax.lax.top_k(fitness, elite)
        weights = jnp.clip(wheel_fitness, 0.0)
        usable = jnp.all(jnp.isfinite(weights)) & (weights.sum() > 0)
        logits = jnp.log(jnp.where(usable, weights, 1.0))
        k_asexual, k_first, k_second, k_mutate, k_cross = jax.random.split(key, 5)
        first = jax.random.categorical(k_first, logits, shape=(size,))
        # The second crossover parent is drawn from the wheel without the first one.
        masked = jnp.where(jnp.arange(elite)[None, :] == first[:, None], -jnp.inf, logits)
        second = jax.random.categorical(k_second, masked, axis=-1)
        parent_a = jnp.where(
            is_asexual, jax.random.categorical(k_asexual, logits, shape=(size,)), first
        )
        parent_a, parent_b = wheel[parent_a], wheel[second]

        # One uniform draw per weight decides whether it mutates (u < p) and, rescaled,
        # by how much: given u < p, 2u/p - 1 is uniform on (-1, 1).
        step_size = state["mutation_step"]
        p = params.mutation_probability
        u = jax.random.uniform(k_mutate, (asexual, problem.net.size))
        delta = jnp.where(u < p, (2.0 * u / max(p, 1e-12) - 1.0) * step_size, 0.0)
        mutants = population[parent_a[:asexual]] + delta
        take_first = jax.random.bernoulli(k_cross, 0.5, (sexual, problem.net.size))
        crossed = jnp.where(
            take_first, population[parent_a[asexual:]], population[parent_b[asexual:]]
        )

        metrics = {
            "loss": losses[best],
            "accuracy": correct[best] / x.shape[0],
            "mean_loss": jnp.mean(jnp.where(jnp.isfinite(losses), losses, jnp.nan)),
            "mutation_step": step_size,
        }
        state = state | {
            "population": jnp.concatenate([mutants, crossed]),
            "fitness": fitness,
            "parent_a": parent_a.astype(jnp.int32),
            "parent_b": parent_b.astype(jnp.int32),
            "first": jnp.array(False),
            "best": population[best],
        }
        return state, metrics

    def on_validation(state, improved):
        patience = jnp.where(improved, 0, state["patience"] + 1)
        decay = patience >= params.validation_patience
        return state | {
            "patience": jnp.where(decay, 0, patience).astype(jnp.int32),
            "mutation_step": jnp.where(
                decay, state["mutation_step"] * params.mutation_decay, state["mutation_step"]
            ),
        }

    return Optimizer(
        init=init,
        step=step,
        current=lambda state: state["best"],
        on_validation=on_validation,
        extra_metrics=("mean_loss", "mutation_step"),
        validation_metrics=lambda state: {"mutation_step": state["mutation_step"]},
    )
