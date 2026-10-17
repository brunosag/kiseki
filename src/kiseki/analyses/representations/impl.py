"""What each layer computes, and how similar two models' layers are.

Activation statistics are over `samples` images. Similarity is linear CKA (Kornblith et
al. 2019) between every layer of one model and every layer of the other, from n×n Gram
matrices, so layers of any width compare at the same cost.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from kiseki.analyses.representations import TITLE, Params
from kiseki.models.layers import Network
from kiseki.worker.report import Context, Section, md_table


def layer_names(net: Network) -> list[str]:
    """Layers worth comparing: each ReLU output, and the logits."""
    return [n for n in net.names if n.startswith("relu")] + [net.names[-1]]


def centered_gram(features: jax.Array) -> jax.Array:
    gram = features @ features.T
    mean_rows = gram.mean(axis=0, keepdims=True)
    return gram - mean_rows - mean_rows.T + gram.mean()


def cka(first: jax.Array, second: jax.Array) -> jax.Array:
    """Linear CKA between two centered Gram matrices."""
    return jnp.vdot(first, second) / (jnp.linalg.norm(first) * jnp.linalg.norm(second))


def activations(net: Network, flat, x, names):
    def one(batch):
        logits, taps = net.forward(flat, batch, taps=tuple(n for n in names if n != net.names[-1]))
        taps = dict(taps)
        taps[net.names[-1]] = logits
        return {n: taps[n].reshape(batch.shape[0], -1) for n in names}

    chunks = [one(x[i : i + 500]) for i in range(0, x.shape[0], 500)]
    return {n: jnp.concatenate([c[n] for c in chunks]) for n in names}


def stats(values: jax.Array, bins: int) -> dict:
    """Statistics of one layer's (samples, units) activations."""
    flat = values.reshape(-1)
    sample = flat[:: max(1, flat.size // 1_000_000)]  # quantiles from at most ~1M values
    q05, q50, q95 = (float(v) for v in jnp.quantile(sample, jnp.array([0.05, 0.5, 0.95])))
    counts, edges = jnp.histogram(sample, bins=bins)
    return {
        "units": int(values.shape[1]),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "zeros": float((flat == 0).mean()),
        "dead_units": float((values.max(axis=0) <= 0).mean()),
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "histogram": {"edges": np.asarray(edges).tolist(), "counts": np.asarray(counts).tolist()},
    }


def run(params: Params, context: Context) -> Section:
    count = min(params.samples, context.size)
    order = np.random.default_rng(context.seed).permutation(context.size)[:count]
    x = context.x[jnp.asarray(np.sort(order))]
    summary, data, rows = {}, {}, []
    grams: dict[str, dict[str, jax.Array]] = {}
    for subject in context.subjects:
        names = layer_names(subject.net)
        acts = activations(subject.net, subject.weights, x, names)
        layers = {}
        for name in names:
            layers[name] = stats(acts[name], params.bins)
            rows.append(
                [
                    subject.label,
                    name,
                    layers[name]["units"],
                    100 * layers[name]["zeros"],
                    100 * layers[name]["dead_units"],
                    layers[name]["mean"],
                    layers[name]["q95"],
                ]
            )
        data[subject.label] = {"layers": layers}
        summary[subject.label] = {"dead_units": {n: layers[n]["dead_units"] for n in names}}
        grams[subject.label] = {n: centered_gram(acts[n]) for n in names}
    markdown = [
        f"Over {count} images. zeros: share of activations exactly 0; dead: share of "
        "units never active on any image.",
        "",
        md_table(["model", "layer", "units", "zeros %", "dead %", "mean", "q95"], rows),
    ]
    figures = []
    for first, second in context.pairs():
        key = f"{first.label} vs {second.label}"
        first_names, second_names = list(grams[first.label]), list(grams[second.label])
        matrix = np.array(
            [
                [float(cka(grams[first.label][a], grams[second.label][b])) for b in second_names]
                for a in first_names
            ]
        )
        data[key] = {"cka": matrix.tolist(), "rows": first_names, "columns": second_names}
        diagonal = {
            a: float(matrix[i, second_names.index(a)])
            for i, a in enumerate(first_names)
            if a in second_names
        }
        summary[key] = {"cka_same_layer": diagonal}
        markdown += [
            "",
            f"**{key}**, linear CKA of matching layers (1: same representation up "
            "to rotation and scale):",
            "",
            md_table(["layer", "CKA"], [[n, v] for n, v in diagonal.items()]),
        ]
        figure, axis = plt.subplots(
            figsize=(1 + 0.6 * len(second_names), 1 + 0.6 * len(first_names))
        )
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(second_names)), second_names, rotation=45, fontsize=8)
        axis.set_yticks(range(len(first_names)), first_names, fontsize=8)
        axis.set_xlabel(second.label)
        axis.set_ylabel(first.label)
        for i in range(len(first_names)):
            for j in range(len(second_names)):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] < 0.6 else "black",
                )
        figure.colorbar(image, ax=axis, shrink=0.8)
        axis.set_title("linear CKA", fontsize=10)
        slug = f"cka-{first.ref.run_id[-4:]}-{second.ref.run_id[-4:]}"
        figures.append(context.save_figure(figure, slug))
    return Section(
        name="representations",
        title=TITLE,
        summary=summary,
        data=data,
        markdown="\n".join(markdown),
        figures=figures,
    )
