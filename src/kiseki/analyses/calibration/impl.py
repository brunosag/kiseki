"""Calibration: does a model's confidence match how often it is right?"""

import matplotlib.pyplot as plt
import numpy as np

from kiseki.analyses.calibration import TITLE, Params
from kiseki.worker.report import Context, Section, md_table


def reliability(confidence: np.ndarray, correct: np.ndarray, bins: int) -> dict:
    index = np.minimum((confidence * bins).astype(int), bins - 1)  # [k/bins, (k+1)/bins)
    count = np.bincount(index, minlength=bins)
    total_confidence = np.bincount(index, weights=confidence, minlength=bins)
    total_correct = np.bincount(index, weights=correct.astype(float), minlength=bins)
    safe = np.maximum(count, 1)
    mean_confidence = np.where(count > 0, total_confidence / safe, np.nan)
    accuracy = np.where(count > 0, total_correct / safe, np.nan)
    gaps = np.abs(np.nan_to_num(accuracy) - np.nan_to_num(mean_confidence))
    ece = float((count / confidence.size * gaps).sum())
    mce = float(gaps[count > 0].max()) if count.any() else 0.0
    return {"count": count, "confidence": mean_confidence, "accuracy": accuracy, "ece": ece,
            "mce": mce}


def run(params: Params, context: Context) -> Section:
    labels = context.labels
    summary, data, rows = {}, {}, []
    figure, (diagram, histogram) = plt.subplots(1, 2, figsize=(10, 4))
    centers = (np.arange(params.bins) + 0.5) / params.bins
    for subject in context.subjects:
        evaluation = context.evaluate(subject)
        probabilities = evaluation.probabilities.astype(np.float64)
        confidence = probabilities.max(axis=1)
        curve = reliability(confidence, evaluation.correct, params.bins)
        one_hot = np.eye(probabilities.shape[1])[labels]
        brier = float(((probabilities - one_hot) ** 2).sum(axis=1).mean())
        nll = evaluation.loss
        mean_confidence = float(confidence.mean())
        summary[subject.label] = {"ece": curve["ece"], "mce": curve["mce"], "brier": brier,
                                  "nll": nll, "mean_confidence": mean_confidence,
                                  "accuracy": evaluation.accuracy}
        data[subject.label] = {
            "bins": [{"lower": i / params.bins, "upper": (i + 1) / params.bins,
                      "count": int(curve["count"][i]),
                      "confidence": None if np.isnan(curve["confidence"][i]) else float(curve["confidence"][i]),
                      "accuracy": None if np.isnan(curve["accuracy"][i]) else float(curve["accuracy"][i])}
                     for i in range(params.bins)],
        }
        rows.append([subject.label, 100 * curve["ece"], brier, nll, 100 * mean_confidence,
                     100 * evaluation.accuracy])
        diagram.plot(centers, curve["accuracy"], marker="o", markersize=3, color=subject.color,
                     label=subject.label)
        histogram.hist(confidence, bins=params.bins, range=(0, 1), histtype="step",
                       color=subject.color, label=subject.label)
    diagram.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    diagram.set_xlabel("confidence")
    diagram.set_ylabel("accuracy")
    diagram.set_title("Reliability", fontsize=10)
    diagram.legend(fontsize=8)
    histogram.set_yscale("log")
    histogram.set_xlabel("confidence")
    histogram.set_title("Confidence distribution", fontsize=10)
    markdown = md_table(["model", "ECE %", "Brier", "NLL", "mean confidence %", "accuracy %"], rows)
    markdown += ("\n\nECE: mean |accuracy − confidence| over equal-width bins, weighted by "
                 "bin size; below accuracy means overconfident.")
    return Section(name="calibration", title=TITLE, summary=summary, data=data,
                   markdown=markdown, figures=[context.save_figure(figure, "calibration")])
