"""Static figures of runs (matplotlib, PNG), for agents and reports."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kiseki.store import Run


def run_label(run: Run) -> str:
    spec = run.spec()
    return f"{spec.label or spec.optimizer} {run.id[-4:]}"


def curves(
    runs: list[Run],
    output: Path,
    *,
    metric: str = "val_accuracy",
    x: str = "step",
    log_x: bool = False,
) -> Path:
    """One line per run of `metric` (a train or val column) against step or elapsed time."""
    source = "val" if metric.startswith("val_") or x == "elapsed" else "train"
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=120)
    for run in runs:
        rows = list(run.iter_rows(source))
        if not rows:
            continue
        if source == "train" and len(rows) > 4000:  # thin dense step metrics
            stride = len(rows) // 2000
            rows = rows[::stride]
        axis.plot(
            [r[x] for r in rows],
            [r.get(metric, float("nan")) for r in rows],
            label=run_label(run),
            linewidth=1.2,
        )
    axis.set_xlabel("elapsed seconds" if x == "elapsed" else "step")
    axis.set_ylabel(metric)
    if log_x:
        axis.set_xscale("log")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return output
