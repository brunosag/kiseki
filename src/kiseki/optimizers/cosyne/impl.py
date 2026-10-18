"""CoSyNE as one compiled generation (Gomez, Schmidhuber & Miikkulainen 2008), following
the EvoTorch formulation the previous kiseki used.

Generation t, on minibatch B_t:
1. score the population (N networks);
2. elites: the best ⌊N·elitism_ratio⌋; parents: the best ⌊N/4⌋;
3. 2·|parents| tournaments among parents pick mates; each pair makes two children by
   one-point (or simulated binary) crossover, then Gaussian mutation;
4. permutation: each weight position is shuffled among the networks selected for it, a
   network being selected with probability 1 - (r + 1/2)^(1/D) for centered rank r (best
   r = 1/2: never; worst r = -1/2: always);
5. elites ∪ children ∪ permuted are pooled and the best N, scored on B_t, survive.

Scores already known on B_t are reused instead of recomputed: elites keep theirs, and so do
permuted networks that came out unchanged (about half of them), whose rounds of scoring are skipped.
"""

import jax
import jax.numpy as jnp
from jax import lax

from kiseki.optimizers.cosyne import Params
from kiseki.worker.optimizer import Optimizer, Problem, centered_ranks, init_population

_COLUMN_GROUP = 4096  # weight positions permuted at once; bounds the sort's memory


def build(params: Params, problem: Problem) -> Optimizer:
    size = params.population_size
    dims = problem.net.size
    elites = min(size, int(size * params.elitism_ratio))
    parents = size // 4
    pairs = parents  # 2·parents tournaments make `parents` pairs, two children each
    scorer = problem.scorer
    chunk = min(scorer.chunk, size)

    def init(key):
        population = init_population(problem.net, key, size)
        return {"population": population}

    def crossover(key, first, second):
        if params.sbx_eta is None:
            point = jax.random.randint(key, (pairs, 1), 1, dims)
            mask = jnp.arange(dims)[None, :] >= point
            return jnp.concatenate([jnp.where(mask, first, second), jnp.where(mask, second, first)])
        eta = params.sbx_eta
        u = jax.random.uniform(key, first.shape)
        beta = jnp.where(
            u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        )
        return jnp.concatenate(
            [
                0.5 * ((1 + beta) * first + (1 - beta) * second),
                0.5 * ((1 + beta) * second + (1 - beta) * first),
            ]
        )

    def mutate(key, children):
        if params.mutation_stdev <= 0 or params.mutation_probability <= 0:
            return children
        k_noise, k_mask = jax.random.split(key)
        noise = params.mutation_stdev * jax.random.normal(k_noise, children.shape)
        if params.mutation_probability >= 1:
            return children + noise
        mask = jax.random.uniform(k_mask, children.shape) <= params.mutation_probability
        return children + jnp.where(mask, noise, 0.0)

    def permute(key, population, losses):
        """Shuffle each weight position among the networks selected for it."""
        if params.permute_all:
            probability = jnp.ones(size)
        else:
            rank = centered_ranks(-losses)  # best network: 0.5
            probability = 1.0 - (rank + 0.5) ** (1.0 / dims)
        padded = -(-dims // _COLUMN_GROUP) * _COLUMN_GROUP
        columns = jnp.pad(population, ((0, 0), (0, padded - dims))).T.reshape(
            -1, _COLUMN_GROUP, size
        )
        rows = jnp.arange(size)

        def block(args):
            values, block_key = args  # values: (columns, networks)
            k_select, k_order = jax.random.split(block_key)
            selected = jax.random.uniform(k_select, values.shape) <= probability
            # Selected networks first, in random order; unselected ones after, in order.
            keys = jnp.where(selected, jax.random.uniform(k_order, values.shape), 2.0 + rows / size)
            source = jnp.argsort(keys, axis=-1)
            # A selected network takes the value at its position in the selected list.
            slot = jnp.where(
                selected,
                jnp.cumsum(selected, axis=-1) - 1,
                selected.sum(-1, keepdims=True) + jnp.cumsum(~selected, axis=-1) - 1,
            )
            return jnp.take_along_axis(values, jnp.take_along_axis(source, slot, -1), -1)

        keys = jax.random.split(key, columns.shape[0])
        shuffled = lax.map(block, (columns, keys))
        return shuffled.reshape(padded, size).T[:, :dims]

    def score_skipping_unchanged(candidates, original, known_losses, known_correct, x, y):
        changed = jnp.any(candidates != original, axis=1)
        order = jnp.argsort(~changed, stable=True)  # changed networks first
        rounds = -(-size // chunk)
        pad = rounds * chunk - size
        ordered = jnp.pad(candidates[order], ((0, pad), (0, 0))).reshape(rounds, chunk, dims)
        live = jnp.pad(changed[order], (0, pad)).reshape(rounds, chunk).any(axis=1)
        losses, correct = scorer.score_chunks(ordered, live, x, y)
        losses = jnp.empty(size).at[order].set(losses[:size])
        correct = jnp.empty(size, jnp.int32).at[order].set(correct[:size])
        return (
            jnp.where(changed, losses, known_losses),
            jnp.where(changed, correct, known_correct),
        )

    def step(state, key, x, y):
        population = state["population"]
        losses, correct = scorer.score(population, x, y)
        order = jnp.argsort(losses)
        k_tour, k_cross, k_mutate, k_perm = jax.random.split(key, 4)

        # Parents are sorted best-first, so a tournament's winner is its smallest position.
        contestants = jax.random.randint(k_tour, (2 * pairs, params.tournament_size), 0, parents)
        winners = order[contestants.min(axis=1)]
        children = mutate(
            k_mutate, crossover(k_cross, population[winners[:pairs]], population[winners[pairs:]])
        )
        child_losses, child_correct = scorer.score(children, x, y)

        permuted = permute(k_perm, population, losses)
        perm_losses, perm_correct = score_skipping_unchanged(
            permuted, population, losses, correct, x, y
        )

        pool = jnp.concatenate([population[order[:elites]], children, permuted])
        pool_losses = jnp.concatenate([losses[order[:elites]], child_losses, perm_losses])
        pool_correct = jnp.concatenate([correct[order[:elites]], child_correct, perm_correct])
        survivors = jnp.argsort(pool_losses)[:size]
        metrics = {
            "loss": pool_losses[survivors[0]],
            "accuracy": pool_correct[survivors[0]] / x.shape[0],
            "mean_loss": jnp.mean(jnp.where(jnp.isfinite(losses), losses, jnp.nan)),
            "permuted_changed": jnp.mean(jnp.any(permuted != population, axis=1)),
        }
        return {"population": pool[survivors]}, metrics

    return Optimizer(
        init=init,
        step=step,
        current=lambda state: state["population"][0],
        extra_metrics=("mean_loss", "permuted_changed"),
    )
