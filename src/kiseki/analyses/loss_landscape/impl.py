"""Loss around each model and between models, scored as populations on the GPU.

Planes (Li et al. 2018, "Visualizing the Loss Landscape of Neural Nets"): two random
Gaussian directions, filter-normalized so each conv filter and dense unit moves in
proportion to its own norm (biases stay put), and the loss over the grid
θ + α·d1 + β·d2, α, β ∈ [-radius, radius]. The raw directions come from the report's
seed, so models of one architecture are probed along the same directions.

Lines (Frankle et al. 2020, "Linear Mode Connectivity and the Lottery Ticket Hypothesis"):
the loss along (1 - t)·θ_a + t·θ_b, past both ends by `line_margin`. The barrier is the
largest rise above the straight line between the two end losses, for t in [0, 1]; near
zero means one basin.

Each grid or line point is one network, so the whole plane is scored the way training
scores a population: chunks of networks vmapped over the same images.
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import lax

from kiseki.analyses.loss_landscape import TITLE, Params
from kiseki.models.layers import Network
from kiseki.worker.population import cross_entropy
from kiseki.worker.report import Context, Section, md_table
from kiseki.worker.train import device_limit

ACCURACY_TOLERANCE = 0.01  # basin share: points within 1 pp of the model's accuracy


def direction(net: Network, flat: jax.Array, key: jax.Array) -> jax.Array:
    """A random direction scaled per filter (last axis) to the weights' norms; zero biases."""
    parts = []
    keys = jax.random.split(key, len(net.slots))
    for slot, tensor, slot_key in zip(net.slots, net.unflatten(flat), keys, strict=True):
        if slot.name.endswith(".b"):
            parts.append(jnp.zeros(slot.size, jnp.float32))
            continue
        d = jax.random.normal(slot_key, slot.shape, jnp.float32)
        axes = tuple(range(tensor.ndim - 1))
        norm_w = jnp.sqrt((tensor**2).sum(axes, keepdims=True))
        norm_d = jnp.sqrt((d**2).sum(axes, keepdims=True))
        parts.append((d * norm_w / jnp.maximum(norm_d, 1e-12)).reshape(-1))
    return jnp.concatenate(parts)


def chunk_for(net: Network, images: int, limit: int) -> int:
    """Networks per vmapped chunk: activations for `images` within ~30% of the device."""
    if not limit:
        return 8
    flat = jax.ShapeDtypeStruct((net.size,), jnp.float32)
    x = jax.ShapeDtypeStruct((images, *net.input_shape), jnp.float32)
    taps = jax.eval_shape(lambda f, b: net.forward(f, b, taps=net.names)[1], flat, x)
    per_network = 4 * sum(math.prod(t.shape) for t in taps.values())
    return max(1, min(64, 2 ** int(math.log2(max(1.0, 0.3 * limit / per_network)))))


def score_along(net: Network, chunk: int):
    """Compiled (origin, directions (k, size), coefficients (n, k), x, y) -> losses, accuracies."""

    def score(origin, directions, coefficients, x, y):
        def one(c):
            loss, correct = cross_entropy(net.apply(origin + c @ directions, x), y)
            return loss, correct / y.shape[0]

        losses, accuracies = lax.map(
            one, coefficients, batch_size=min(chunk, coefficients.shape[0])
        )
        return jnp.where(jnp.isfinite(losses), losses, jnp.inf), accuracies

    return jax.jit(score)


def sample(context: Context, params: Params) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(context.seed + 2)
    if params.data == "train":
        count = int(context.data.train_x.shape[0])
        index = jnp.asarray(np.sort(rng.permutation(count)[: params.samples]))
        return context.data.normalize(context.data.train_x[index]), context.data.train_y[index]
    index = jnp.asarray(np.sort(rng.permutation(context.size)[: params.samples]))
    return context.x[index], context.y[index]


def ring(grid_radius: np.ndarray, values: np.ndarray, r: float, width: float) -> float:
    near = np.abs(grid_radius - r) <= width
    return float(values[near].mean())


def run(params: Params, context: Context) -> Section:
    x, y = sample(context, params)
    limit = device_limit()
    axis = np.linspace(-params.radius, params.radius, params.resolution)
    alpha, beta = np.meshgrid(axis, axis, indexing="xy")
    plane = jnp.asarray(np.stack([alpha.ravel(), beta.ravel()], 1), jnp.float32)
    radius = np.hypot(alpha, beta)
    step = axis[1] - axis[0]
    key = jax.random.key(context.seed)
    scorers: dict[str, Callable] = {}

    def scorer(net: Network):
        if net.name not in scorers:
            scorers[net.name] = score_along(net, chunk_for(net, x.shape[0], limit))
        return scorers[net.name]

    summary: dict = {}
    data: dict = {"axis": axis.tolist(), "data": params.data, "samples": int(x.shape[0])}
    arrays, planes = [], {}
    rows = []
    for number, subject in enumerate(context.subjects):
        net, flat = subject.net, subject.weights
        directions = jnp.stack([direction(net, flat, jax.random.fold_in(key, i)) for i in range(2)])
        score = scorer(net)
        losses, accuracies = score(flat, directions, plane, x, y)
        (center_loss,), (center_accuracy,) = score(flat, directions, jnp.zeros((1, 2)), x, y)
        loss_grid = np.asarray(losses).reshape(alpha.shape)
        accuracy_grid = np.asarray(accuracies).reshape(alpha.shape)
        center_loss, center_accuracy = float(center_loss), float(center_accuracy)
        width = max(step, 0.05 * params.radius)
        disk = radius <= params.radius
        result = {
            "loss": center_loss,
            "accuracy": center_accuracy,
            "rise_half": ring(radius, loss_grid, params.radius / 2, width) - center_loss,
            "rise_edge": ring(radius, loss_grid, params.radius, width) - center_loss,
            "basin_share": float(
                (accuracy_grid[disk] >= center_accuracy - ACCURACY_TOLERANCE).mean()
            ),
        }
        summary[subject.label] = result
        planes[number] = loss_grid
        data[subject.label] = {
            "loss": context.save_array(
                f"loss_landscape-{number}-plane-loss", loss_grid.astype(np.float32)
            ),
            "accuracy": context.save_array(
                f"loss_landscape-{number}-plane-accuracy", accuracy_grid.astype(np.float32)
            ),
        }
        arrays += list(data[subject.label].values())
        rows.append(
            [
                subject.label,
                center_loss,
                100 * center_accuracy,
                result["rise_half"],
                result["rise_edge"],
                100 * result["basin_share"],
            ]
        )

    t = np.linspace(-params.line_margin, 1 + params.line_margin, params.line_points)
    inside = (t >= 0) & (t <= 1)
    lines, pair_rows = [], []
    for first, other in context.pairs():
        if first.net.name != other.net.name:
            continue
        score = scorer(first.net)
        directions = (other.weights - first.weights)[None, :]
        coefficients = jnp.asarray(np.concatenate([t, [0.0, 1.0]])[:, None], jnp.float32)
        losses, accuracies = (
            np.asarray(v) for v in score(first.weights, directions, coefficients, x, y)
        )
        (loss_a, loss_b), (accuracy_a, accuracy_b) = losses[-2:], accuracies[-2:]
        losses, accuracies = losses[:-2], accuracies[:-2]
        loss_barrier = float(
            max(0.0, (losses[inside] - ((1 - t) * loss_a + t * loss_b)[inside]).max())
        )
        accuracy_barrier = float(
            max(0.0, (((1 - t) * accuracy_a + t * accuracy_b)[inside] - accuracies[inside]).max())
        )
        name = f"{first.label} → {other.label}"
        distance = float(jnp.linalg.norm(other.weights - first.weights))
        summary.setdefault("lines", {})[name] = {
            "loss_barrier": loss_barrier,
            "accuracy_barrier": accuracy_barrier,
            "distance": distance,
        }
        path = context.save_array(
            f"loss_landscape-line-{len(lines)}",
            np.stack([t, losses, accuracies], 1).astype(np.float32),
        )
        arrays.append(path)
        data.setdefault("lines", {})[name] = path
        lines.append((first, other, losses, accuracies))
        pair_rows.append([name, distance, loss_barrier, 100 * accuracy_barrier])

    figures = []
    count = len(context.subjects)
    figure, axes = plt.subplots(1, count, figsize=(4.2 * count, 3.8), squeeze=False, sharey=True)
    finite = np.concatenate([g[np.isfinite(g)] for g in planes.values()])
    low = np.log10(max(1e-4, finite.min()))
    high = np.log10(max(10**low * 1.01, np.percentile(finite, 98)))
    levels = np.linspace(low, high, 25)
    fills = []
    for number, subject in enumerate(context.subjects):
        ax = axes[0][number]
        shown = np.log10(np.clip(planes[number], 10**low, 10**high))
        fills.append(ax.contourf(alpha, beta, shown, levels=levels, cmap="viridis"))
        ax.contour(alpha, beta, shown, levels=levels[::4], colors="white", linewidths=0.4)
        ax.plot(0, 0, marker="+", color="red", markersize=10)
        ax.set_title(subject.label, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("α")
        if number == 0:
            ax.set_ylabel("β")
    figure.colorbar(
        fills[-1],
        ax=axes[0].tolist(),
        label="log10 loss",
        shrink=0.85,
    )
    figures.append(context.save_figure(figure, "loss_landscape-planes"))
    if lines:
        figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 3.6))
        for first, other, losses, accuracies in lines:
            label = f"{first.label} → {other.label}"
            loss_axis.plot(t, losses, color=other.color, label=label)
            accuracy_axis.plot(t, 100 * accuracies, color=other.color, label=label)
        for ax in (loss_axis, accuracy_axis):
            for edge in (0, 1):
                ax.axvline(edge, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlabel("t (0: first model, 1: other)")
        loss_axis.set_yscale("log")
        loss_axis.set_ylabel("loss")
        accuracy_axis.set_ylabel("accuracy %")
        accuracy_axis.legend(fontsize=8)
        figures.append(context.save_figure(figure, "loss_landscape-lines"))

    markdown = (
        f"Loss on {int(x.shape[0])} {params.data} images. Planes: {params.resolution}² "
        f"points within ±{params.radius:g} filter-normalized units. Rise: mean loss "
        f"increase on the circle at r/2 and r (lower: flatter); basin: share of the "
        f"disk within {100 * ACCURACY_TOLERANCE:g} pp of the model's accuracy.\n\n"
        + md_table(["model", "loss", "accuracy %", "rise r/2", "rise r", "basin %"], rows)
    )
    if pair_rows:
        markdown += (
            "\n\nLines between models of one architecture. Barrier: largest rise above "
            "the straight line between the end points, t ∈ [0, 1] (≈0: one basin).\n\n"
            + md_table(["pair", "distance", "loss barrier", "accuracy barrier pp"], pair_rows)
        )
    return Section(
        name="loss_landscape",
        title=TITLE,
        summary=summary,
        data=data,
        markdown=markdown,
        figures=figures,
        arrays=arrays,
    )
