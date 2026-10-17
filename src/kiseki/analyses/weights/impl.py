"""Weights layer by layer, and how far apart two models' weights are."""

import matplotlib.pyplot as plt
import numpy as np

from kiseki.analyses.weights import TITLE, Params
from kiseki.worker.report import Context, Section, md_table


def run(params: Params, context: Context) -> Section:
    summary, data = {}, {}
    host = {s.label: np.asarray(s.weights, np.float64) for s in context.subjects}
    rows = []
    for subject in context.subjects:
        flat = host[subject.label]
        layers = []
        for slot in subject.net.slots:
            values = flat[slot.offset : slot.offset + slot.size]
            layers.append(
                {
                    "name": slot.name,
                    "size": slot.size,
                    "norm": float(np.linalg.norm(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "abs_max": float(np.abs(values).max()),
                }
            )
            rows.append(
                [
                    subject.label,
                    slot.name,
                    slot.size,
                    layers[-1]["norm"],
                    layers[-1]["std"],
                    layers[-1]["abs_max"],
                ]
            )
        data[subject.label] = {"layers": layers}
        summary[subject.label] = {"norm": float(np.linalg.norm(flat))}
    markdown = [md_table(["model", "tensor", "size", "L2 norm", "std", "max |w|"], rows)]

    if context.same_architecture:
        for first, second in context.pairs():
            a, b = host[first.label], host[second.label]
            key = f"{first.label} vs {second.label}"
            per_layer, pair_rows = [], []
            for slot in first.net.slots:
                x = a[slot.offset : slot.offset + slot.size]
                y = b[slot.offset : slot.offset + slot.size]
                distance = float(np.linalg.norm(x - y))
                scale = (np.linalg.norm(x) + np.linalg.norm(y)) / 2
                cosine = float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))
                per_layer.append(
                    {
                        "name": slot.name,
                        "distance": distance,
                        "relative_distance": distance / max(scale, 1e-12),
                        "cosine": cosine,
                    }
                )
                pair_rows.append([slot.name, distance, distance / max(scale, 1e-12), cosine])
            summary[key] = {
                "distance": float(np.linalg.norm(a - b)),
                "cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
            }
            data[key] = {"layers": per_layer}
            markdown += [
                "",
                f"**{key}**: distance {summary[key]['distance']:.4g}, cosine "
                f"{summary[key]['cosine']:.3f} (weights are only comparable up to "
                "permutation of hidden units).",
                "",
                md_table(["tensor", "L2 distance", "relative", "cosine"], pair_rows),
            ]

    weight_slots = [slot for slot in context.subjects[0].net.slots if slot.name.endswith(".w")]
    figure, axes = plt.subplots(
        1, len(weight_slots), figsize=(2.6 * len(weight_slots), 2.6), squeeze=False
    )
    for axis, slot in zip(axes[0], weight_slots, strict=True):
        for subject in context.subjects:
            match = next((s for s in subject.net.slots if s.name == slot.name), None)
            if match is None:
                continue
            values = host[subject.label][match.offset : match.offset + match.size]
            axis.hist(
                values,
                bins=params.bins,
                histtype="step",
                density=True,
                color=subject.color,
                label=subject.label,
            )
        axis.set_title(slot.name, fontsize=9)
        axis.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7)
    return Section(
        name="weights",
        title=TITLE,
        summary=summary,
        data=data,
        markdown="\n".join(markdown),
        figures=[context.save_figure(figure, "weights")],
    )
