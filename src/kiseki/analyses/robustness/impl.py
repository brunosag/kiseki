"""Accuracy as the test images are corrupted. Perturbations act in pixel space ([0, 1]) and
are identical for every model (noise from a fixed key per level)."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kiseki.analyses.robustness import TITLE, Params
from kiseki.worker.report import Context, Section, accuracy_and_loss, md_table


def perturb(kind: str, pixels: jax.Array, level: float, key: jax.Array) -> jax.Array:
    match kind:
        case "noise":
            out = pixels + level * jax.random.normal(key, pixels.shape)
        case "brightness":
            out = pixels + level
        case "contrast":
            mean = pixels.mean(axis=(1, 2, 3), keepdims=True)
            out = mean + level * (pixels - mean)
        case "occlusion":
            _, height, width, _ = pixels.shape
            side = round(level * min(height, width))
            top, left = (height - side) // 2, (width - side) // 2
            rows = (jnp.arange(height) >= top) & (jnp.arange(height) < top + side)
            cols = (jnp.arange(width) >= left) & (jnp.arange(width) < left + side)
            mask = rows[:, None] & cols[None, :]
            out = jnp.where(mask[None, :, :, None], 0.0, pixels)
        case _:
            raise ValueError(kind)
    return jnp.clip(out, 0.0, 1.0)


def run(params: Params, context: Context) -> Section:
    pixels = context.pixels(context.x)
    kinds = {
        "noise": params.noise,
        "brightness": params.brightness,
        "contrast": params.contrast,
        "occlusion": params.occlusion,
    }
    evaluators = {s.label: accuracy_and_loss(s.net) for s in context.subjects}
    data: dict = {kind: {"levels": list(levels)} for kind, levels in kinds.items()}
    summary: dict = {}
    key = jax.random.key(context.seed)
    for kind, levels in kinds.items():
        for subject in context.subjects:
            data[kind][subject.label] = {"accuracy": [], "loss": []}
        for index, level in enumerate(levels):
            images = context.normalize(perturb(kind, pixels, level, jax.random.fold_in(key, index)))
            for subject in context.subjects:
                accuracy, loss = evaluators[subject.label](subject.weights, images, context.y)
                data[kind][subject.label]["accuracy"].append(float(accuracy))
                data[kind][subject.label]["loss"].append(float(loss))
    for subject in context.subjects:
        summary[subject.label] = {
            f"{kind}_mean_accuracy": sum(data[kind][subject.label]["accuracy"]) / len(levels)
            for kind, levels in kinds.items()
        }
    figure, axes = plt.subplots(1, len(kinds), figsize=(3.4 * len(kinds), 3), squeeze=False)
    rows = []
    for axis, (kind, levels) in zip(axes[0], kinds.items(), strict=True):
        for subject in context.subjects:
            accuracy = data[kind][subject.label]["accuracy"]
            axis.plot(
                levels, accuracy, marker="o", markersize=3, color=subject.color, label=subject.label
            )
            rows.append([kind, subject.label, *[f"{100 * a:.1f}" for a in accuracy]])
        axis.set_title(kind, fontsize=10)
        axis.set_xlabel("level")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy")
    axes[0][0].legend(fontsize=7)
    lines = []
    for kind, levels in kinds.items():
        lines += [
            f"**{kind}** (levels {', '.join(f'{v:g}' for v in levels)}): accuracy %",
            "",
            md_table(["model", *[f"{v:g}" for v in levels]], [r[1:] for r in rows if r[0] == kind]),
            "",
        ]
    return Section(
        name="robustness",
        title=TITLE,
        summary=summary,
        data=data,
        markdown="\n".join(lines),
        figures=[context.save_figure(figure, "robustness")],
    )
