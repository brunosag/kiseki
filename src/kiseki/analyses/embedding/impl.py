"""Maps of each model's penultimate representation: PCA, and exact t-SNE on the GPU.

t-SNE (van der Maaten & Hinton 2008) here is exact, not Barnes-Hut: input affinities from
each point's 3·perplexity nearest neighbours (perplexity matched by bisection), then every
pairwise repulsion computed each iteration, row-blocked to bound memory. On a GPU that is
faster than Barnes-Hut on a CPU at these sizes (n ≤ 10k) and has no approximation error.
Optimisation follows the reference implementation: early exaggeration 12 for 250
iterations, momentum 0.5 then 0.8, delta-bar-delta gains, learning rate max(n/48, 50),
initialised from PCA (Kobak & Berens 2019).

Each model gets its own map: penultimate spaces of different networks are not aligned, so
one joint map would mostly separate the models from each other.
"""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import lax
from matplotlib.lines import Line2D

from kiseki.analyses.embedding import TITLE, Params
from kiseki.worker.report import Context, Section, md_table

EXAGGERATION = 12.0
EXAGGERATION_ITERATIONS = 250
ROW_BLOCK = 1024


def squared_distances(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)


def nearest(features: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """Squared distances and indices of each point's k nearest other points."""
    n = features.shape[0]
    d2 = squared_distances(features, features).at[jnp.arange(n), jnp.arange(n)].set(jnp.inf)
    negative, index = lax.top_k(-d2, k)
    return -negative, index


def pca(features: jax.Array, components: int) -> tuple[jax.Array, jax.Array]:
    centered = features - features.mean(0)
    _, singular, vt = jnp.linalg.svd(centered, full_matrices=False)
    explained = singular**2 / (singular**2).sum()
    return centered @ vt[:components].T, explained[:components]


def affinities(distances: jax.Array, index: jax.Array, perplexity: float) -> jax.Array:
    """Symmetric joint probabilities P (dense n×n) from kNN squared distances."""
    n = distances.shape[0]
    distances = distances - distances[:, :1]  # nearest at 0: avoids underflow in exp
    target = jnp.log(perplexity)

    def entropy(beta):
        p = jnp.exp(-distances * beta[:, None])
        total = p.sum(1)
        return jnp.log(total) + beta * (distances * p).sum(1) / total, p / total[:, None]

    def bisect(_, bounds):
        low, high, beta = bounds
        h, _ = entropy(beta)
        too_flat = h > target  # entropy too high: sharpen (raise beta)
        low = jnp.where(too_flat, beta, low)
        high = jnp.where(too_flat, high, beta)
        beta = jnp.where(jnp.isinf(high), beta * 2.0, (low + high) / 2.0)
        return low, high, beta

    start = (jnp.zeros(n), jnp.full(n, jnp.inf), jnp.ones(n))
    _, _, beta = lax.fori_loop(0, 64, bisect, start)
    _, conditional = entropy(beta)
    p = jnp.zeros((n, n)).at[jnp.arange(n)[:, None], index].set(conditional)
    return (p + p.T) / (2.0 * n)


def tsne(features: jax.Array, init: jax.Array, perplexity: float, iterations: int) -> jax.Array:
    n = features.shape[0]
    k = min(n - 1, int(3 * perplexity))
    distances, index = nearest(features, k)
    p = affinities(distances, index, perplexity)
    learning_rate = max(n / EXAGGERATION / 4.0, 50.0)
    pad = -(-n // ROW_BLOCK) * ROW_BLOCK - n
    starts = jnp.arange(0, n + pad, ROW_BLOCK)
    p_padded = jnp.pad(p, ((0, pad), (0, 0)))

    def kernel_rows(y, start):
        rows = lax.dynamic_slice_in_dim(jnp.pad(y, ((0, pad), (0, 0))), start, ROW_BLOCK)
        kernel = 1.0 / (1.0 + squared_distances(rows, y))
        own = (start + jnp.arange(ROW_BLOCK))[:, None] == jnp.arange(n)[None, :]
        valid = (start + jnp.arange(ROW_BLOCK)) < n
        return rows, jnp.where(own | ~valid[:, None], 0.0, kernel)

    def gradient(y, exaggeration):
        total = lax.map(lambda s: kernel_rows(y, s)[1].sum(), starts).sum()

        def block(start):
            rows, kernel = kernel_rows(y, start)
            pq = (
                exaggeration * lax.dynamic_slice_in_dim(p_padded, start, ROW_BLOCK) - kernel / total
            ) * kernel
            return 4.0 * (pq.sum(1)[:, None] * rows - pq @ y)

        return lax.map(block, starts).reshape(-1, 2)[:n]

    def body(carry, iteration):
        y, update, gains = carry
        early = iteration < EXAGGERATION_ITERATIONS
        grad = gradient(y, jnp.where(early, EXAGGERATION, 1.0))
        same = jnp.sign(grad) == jnp.sign(update)
        gains = jnp.maximum(jnp.where(same, gains * 0.8, gains + 0.2), 0.01)
        momentum = jnp.where(early, 0.5, 0.8)
        update = momentum * update - learning_rate * gains * grad
        return (y + update, update, gains), None

    start = (init, jnp.zeros_like(init), jnp.ones_like(init))
    (y, _, _), _ = lax.scan(body, start, jnp.arange(iterations))
    return y


def knn_accuracy(index: jax.Array, labels: jax.Array) -> float:
    """Leave-one-out majority vote of the k nearest neighbours."""
    votes = jax.nn.one_hot(labels[index], int(labels.max()) + 1).sum(1)
    return float((votes.argmax(1) == labels).mean())


def trustworthiness(features: jax.Array, embedded: jax.Array, k: int) -> float:
    """How well the map's neighbourhoods exist in the original space (1: all do)."""
    n = features.shape[0]
    original = squared_distances(features, features).at[jnp.arange(n), jnp.arange(n)].set(-1.0)
    _, original_nn = lax.top_k(-original, k + 1)  # includes self (distance -1)
    _, embedded_nn = nearest(embedded, k)
    # Rank in the original space of each map neighbour: how many points are closer.
    picked = jnp.take_along_axis(original, embedded_nn, 1)
    ranks = (original[:, None, :] < picked[:, :, None]).sum(-1)  # self counts as rank 0
    inside = (embedded_nn[:, :, None] == original_nn[:, None, :]).any(-1)
    penalty = jnp.where(inside, 0, ranks - k).sum()
    return float(1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * penalty)


def balanced_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, count // len(classes))
    chosen = [rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in classes]
    return np.sort(np.concatenate(chosen))


def run(params: Params, context: Context) -> Section:
    indices = balanced_indices(context.labels, min(params.samples, context.size), context.seed)
    labels = jnp.asarray(context.labels[indices])
    n = len(indices)
    perplexity = min(params.perplexity, (n - 1) / 3)
    k = max(1, min(params.neighbors, (n - 1) // 2))  # trustworthiness is defined for k < n/2
    run_tsne = jax.jit(tsne, static_argnums=(2, 3))
    summary, rows = {}, []
    data: dict[str, Any] = {
        "indices": context.save_array("embedding-indices", indices),
        "labels": context.save_array("embedding-labels", context.labels[indices]),
    }
    arrays = [data["indices"], data["labels"]]
    count = len(context.subjects)
    figure, axes = plt.subplots(count, 2, figsize=(9, 4.4 * count), squeeze=False)
    for number, subject in enumerate(context.subjects):
        features = jnp.asarray(context.evaluate(subject).features[indices], jnp.float32)
        components = min(params.pca_components, features.shape[1], n)
        reduced, explained = pca(features, components)
        init = reduced[:, :2] / reduced[:, 0].std() * 1e-4
        embedded = run_tsne(reduced, init, perplexity, params.iterations)
        _, feature_nn = nearest(features, k)
        _, map_nn = nearest(embedded, k)
        result = {
            "knn_accuracy_features": knn_accuracy(feature_nn, labels),
            "knn_accuracy_tsne": knn_accuracy(map_nn, labels),
            "trustworthiness": trustworthiness(reduced, embedded, k),
            "pca_explained_2": float(explained[:2].sum()),
            "features": int(features.shape[1]),
        }
        summary[subject.label] = result
        names = {
            kind: context.save_array(f"embedding-{number}-{kind}", value)
            for kind, value in (
                ("tsne", embedded),
                ("pca", reduced[:, :2]),
                ("predictions", context.evaluate(subject).predictions[indices]),
            )
        }
        data[subject.label] = names
        arrays += list(names.values())
        rows.append(
            [
                subject.label,
                result["features"],
                100 * result["knn_accuracy_features"],
                100 * result["knn_accuracy_tsne"],
                result["trustworthiness"],
                100 * result["pca_explained_2"],
            ]
        )
        for axis, (kind, values) in zip(
            axes[number], (("PCA", reduced), ("t-SNE", embedded)), strict=True
        ):
            points = np.asarray(values)
            axis.scatter(
                points[:, 0],
                points[:, 1],
                c=np.asarray(labels),
                cmap="tab10",
                s=2,
                vmin=0,
                vmax=9,
                linewidths=0,
            )
            axis.set_title(f"{subject.label}: {kind}", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
    handles = [
        Line2D([], [], marker="o", linestyle="", color=plt.get_cmap("tab10")(i), markersize=4)
        for i in range(len(context.class_names))
    ]
    figure.legend(handles, context.class_names, loc="lower center", ncol=5, fontsize=8)
    figure.subplots_adjust(bottom=0.08)
    markdown = (
        f"{n} images, class-balanced, the same for every model. kNN accuracy: "
        f"leave-one-out {k}-nearest-neighbour vote, in the penultimate space and on "
        "the t-SNE map; trustworthiness: share of map neighbourhoods that are real.\n\n"
        + md_table(
            [
                "model",
                "features",
                f"{k}-NN acc % (features)",
                f"{k}-NN acc % (t-SNE)",
                "trustworthiness",
                "PCA var % (2 PCs)",
            ],
            rows,
        )
    )
    return Section(
        name="embedding",
        title=TITLE,
        summary=summary,
        data=data,
        markdown=markdown,
        figures=[context.save_figure(figure, "embedding")],
        arrays=arrays,
    )
