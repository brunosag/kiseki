"""Accuracy, loss and per-class metrics of each model on the split."""

import matplotlib.pyplot as plt
import numpy as np

from kiseki.analyses.metrics import TITLE, Params
from kiseki.worker.report import Context, Section, md_table


def confusion(labels: np.ndarray, predictions: np.ndarray, classes: int) -> np.ndarray:
    return np.bincount(labels * classes + predictions, minlength=classes * classes).reshape(
        classes, classes
    )


def per_class(matrix: np.ndarray) -> dict[str, np.ndarray]:
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0)
    support = matrix.sum(axis=1)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def run(params: Params, context: Context) -> Section:
    classes = len(context.class_names)
    summary, data, rows = {}, {}, []
    matrices = {}
    for subject in context.subjects:
        evaluation = context.evaluate(subject)
        matrix = confusion(context.labels, evaluation.predictions, classes)
        matrices[subject.label] = matrix
        scores = per_class(matrix)
        macro_f1 = float(scores["f1"].mean())
        summary[subject.label] = {
            "accuracy": evaluation.accuracy,
            "loss": evaluation.loss,
            "macro_f1": macro_f1,
        }
        data[subject.label] = {
            "confusion": matrix.tolist(),
            "per_class": [
                {
                    "class": name,
                    "precision": float(scores["precision"][i]),
                    "recall": float(scores["recall"][i]),
                    "f1": float(scores["f1"][i]),
                    "support": int(scores["support"][i]),
                }
                for i, name in enumerate(context.class_names)
            ],
        }
        rows.append(
            [
                subject.label,
                100 * evaluation.accuracy,
                evaluation.loss,
                macro_f1,
                context.size - int(evaluation.correct.sum()),
            ]
        )

    lines = [md_table(["model", "accuracy %", "loss", "macro F1", "errors"], rows), ""]
    header = ["class", *[f"F1 {s.label}" for s in context.subjects]]
    f1_rows = [
        [name, *[data[s.label]["per_class"][i]["f1"] for s in context.subjects]]
        for i, name in enumerate(context.class_names)
    ]
    lines.append(md_table(header, f1_rows))

    count = len(context.subjects)
    figure, axes = plt.subplots(1, count, figsize=(4.2 * count, 3.8), squeeze=False)
    for axis, subject in zip(axes[0], context.subjects, strict=True):
        matrix = matrices[subject.label].astype(float)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        np.fill_diagonal(normalized, np.nan)  # errors only: the diagonal would dominate
        axis.imshow(normalized, cmap="Reds")
        axis.set_title(f"{subject.label}: errors by class", fontsize=9)
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
        ticks = range(classes)
        axis.set_xticks(ticks, [n[:6] for n in context.class_names], rotation=90, fontsize=7)
        axis.set_yticks(ticks, [n[:6] for n in context.class_names], fontsize=7)
    figures = [context.save_figure(figure, "confusion")]
    if count >= 2:
        first, second = context.subjects[:2]
        data["confusion_difference"] = (matrices[first.label] - matrices[second.label]).tolist()
    return Section(
        name="metrics",
        title=TITLE,
        summary=summary,
        data=data,
        markdown="\n".join(lines),
        figures=figures,
    )
