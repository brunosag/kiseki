"""Running a report: load the networks, evaluate each once over the split, run the analyses.

Analyses receive a `Context`:
- `subjects`: the networks (Network, weights on device, label, the run's spec);
- `x`, `y`: the split's normalized images and labels on device; `pixels`/`normalize`
  convert to and from [0, 1] pixel space (for perturbations and pictures);
- `evaluate(subject)`: logits, probabilities, predictions and penultimate features over
  the split, computed once and shared;
- `save_figure`, `save_array`: write under the report and return the relative path.

An analysis returns a `Section`: headline numbers (`summary`), details (`data`), prose with
tables (`markdown`), and the files it wrote. report.json and report.md collect them.
"""

import time
import traceback
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import msgspec
import numpy as np
from jax import lax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kiseki import analyses, datasets, models
from kiseki.models.layers import Network
from kiseki.report import Report, ReportSpec, ReportStatus
from kiseki.spec import ModelRef, RunSpec
from kiseki.store import Run
from kiseki.worker import checkpoint
from kiseki.worker import data as device_data
from kiseki.worker.population import cross_entropy

Array = jax.Array
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b")
BATCH = 1000


@dataclass
class Subject:
    ref: ModelRef
    label: str
    spec: RunSpec
    net: Network
    weights: Array
    color: str


@dataclass
class Evaluation:
    logits: np.ndarray  # (N, classes)
    probabilities: np.ndarray
    predictions: np.ndarray
    correct: np.ndarray  # bool
    loss: float
    accuracy: float
    features: np.ndarray  # penultimate activations, (N, F)


@dataclass
class Section:
    name: str
    title: str
    summary: dict[str, Any]
    data: dict[str, Any]
    markdown: str
    figures: list[str] = field(default_factory=list)
    arrays: list[str] = field(default_factory=list)


def batched(function, x: Array, batch: int = BATCH):
    """Apply `function` to `x` in batches inside one compiled loop; outputs concatenated."""
    count = x.shape[0]
    full = count - count % batch
    parts = []
    if full:
        mapped = lax.map(function, x[:full].reshape(-1, batch, *x.shape[1:]))
        parts.append(jax.tree.map(lambda a: a.reshape(full, *a.shape[2:]), mapped))
    if full < count:
        parts.append(function(x[full:]))
    if len(parts) == 1:
        return parts[0]
    return jax.tree.map(lambda *a: jnp.concatenate(a), *parts)


class Context:
    def __init__(self, report: Report, spec: ReportSpec, subjects: list[Subject],
                 data: device_data.DeviceData, info: datasets.DatasetInfo):
        self.report = report
        self.spec = spec
        self.subjects = subjects
        self.data = data
        self.info = info
        self.class_names = info.class_names
        self.x = data.test_x if spec.split == "test" else data.val_x
        self.y = data.test_y if spec.split == "test" else data.val_y
        self.labels = np.asarray(self.y)
        self.seed = spec.seed
        self._evaluations: dict[str, Evaluation] = {}

    @property
    def size(self) -> int:
        return int(self.x.shape[0])

    @cached_property
    def same_architecture(self) -> bool:
        return len({s.net.name for s in self.subjects}) == 1

    def pairs(self) -> list[tuple[Subject, Subject]]:
        first = self.subjects[0]
        return [(first, other) for other in self.subjects[1:]]

    def pixels(self, x: Array) -> Array:
        return x * self.data.std + self.data.mean

    def normalize(self, pixels: Array) -> Array:
        return (pixels - self.data.mean) / self.data.std

    def evaluate(self, subject: Subject) -> Evaluation:
        key = str(subject.ref)
        if key not in self._evaluations:
            logits, features = _forward(subject.net)(subject.weights, self.x)
            logits = np.asarray(logits)
            shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            predictions = logits.argmax(axis=1)
            correct = predictions == self.labels
            self._evaluations[key] = Evaluation(
                logits=logits, probabilities=probabilities, predictions=predictions,
                correct=correct,
                loss=float(-log_p[np.arange(self.size), self.labels].mean()),
                accuracy=float(correct.mean()), features=np.asarray(features),
            )
        return self._evaluations[key]

    def save_figure(self, figure, name: str) -> str:
        relative = f"figures/{name}.png"
        path = self.report.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        return relative

    def save_array(self, name: str, value) -> str:
        relative = f"arrays/{name}.npy"
        path = self.report.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(value))
        return relative


_forwards: dict[str, Any] = {}


def _forward(net: Network):
    """Compiled (weights, images) -> (logits, penultimate features), batched."""
    if net.name not in _forwards:
        def forward(flat, x):
            def one(batch):
                logits, taps = net.forward(flat, batch, taps=(net.penultimate,))
                features = taps[net.penultimate]
                return logits, features.reshape(features.shape[0], -1)
            return batched(one, x)
        _forwards[net.name] = jax.jit(forward)
    return _forwards[net.name]


def accuracy_and_loss(net: Network):
    """Compiled (weights, images, labels) -> (accuracy, mean loss), batched."""
    def evaluate(flat, x, y):
        def one(batch):
            images, labels = batch
            loss, correct = cross_entropy(net.apply(flat, images), labels)
            return loss * labels.shape[0], correct
        count = x.shape[0]
        full = count - count % BATCH
        losses, corrects = lax.map(one, (x[:full].reshape(-1, BATCH, *x.shape[1:]),
                                         y[:full].reshape(-1, BATCH)))
        total, correct = losses.sum(), corrects.sum()
        if full < count:
            loss, extra = one((x[full:], y[full:]))
            total, correct = total + loss, correct + extra
        return correct / count, total / count
    return jax.jit(evaluate)


def load_subjects(spec: ReportSpec) -> tuple[list[Subject], RunSpec]:
    subjects = []
    first: RunSpec | None = None
    for index, text in enumerate(spec.models):
        ref = ModelRef.parse(text)
        run = Run.find(ref.run_id)
        run_spec = run.spec()
        if first is None:
            first = run_spec
        elif run_spec.dataset != first.dataset:
            raise ValueError(f"{ref} is on {run_spec.dataset}, not {first.dataset}")
        info = datasets.info(run_spec.dataset)
        path = run.models_dir / f"{ref.name}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"{ref}: no {path.name}; have {', '.join(run.model_names())}"
                                    " (remote runs: kiseki pull --models)")
        net = models.load(run_spec.model, info.image_shape, len(info.class_names))
        weights = jnp.asarray(checkpoint.load_network(path))
        label = spec.labels[index] if spec.labels else (
            f"{run_spec.label or run_spec.optimizer} {run.id[-4:]}" +
            (f" {ref.name}" if ref.name != "best" else ""))
        subjects.append(Subject(ref=ModelRef(run.id, ref.name), label=label, spec=run_spec,
                                net=net, weights=weights, color=COLORS[index % len(COLORS)]))
    assert first is not None
    return subjects, first


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def build(report: Report) -> int:
    spec = report.spec()
    status = ReportStatus(state="running", stage="loading")
    report.write_status(status)
    subjects, first = load_subjects(spec)
    info = datasets.info(first.dataset)
    assert first.validation_size is not None
    data = device_data.load(first.dataset, validation_size=first.validation_size,
                            split_seed=first.split_seed, batch_size=first.batch_size)
    context = Context(report, spec, subjects, data, info)
    sections: list[Section] = []
    for name, raw in spec.analyses.items():
        status.stage = name
        report.write_status(status)
        started = time.perf_counter()
        params = msgspec.convert(raw, analyses.params_type(name))
        section = analyses.load_impl(name).run(params, context)
        status.seconds[name] = round(time.perf_counter() - started, 2)
        report.write_status(status)
        print(f"{name}: {status.seconds[name]:.1f}s", flush=True)
        sections.append(section)
    write_outputs(report, context, sections)
    status.state, status.stage = "completed", ""
    report.write_status(status)
    return 0


def _plain(value: Any) -> Any:
    """numpy and JAX values in analysis output, as JSON-ready Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray | jax.Array):
        return np.asarray(value).tolist()
    raise NotImplementedError(f"cannot encode {type(value).__name__}")


def write_outputs(report: Report, context: Context, sections: list[Section]) -> None:
    subjects = [{
        "ref": str(s.ref), "label": s.label, "color": s.color, "dataset": s.spec.dataset,
        "model": s.spec.model, "optimizer": s.spec.optimizer, "params": s.spec.params,
        "seed": s.spec.seed,
    } for s in context.subjects]
    document = {
        "id": report.id, "spec": context.spec, "split": context.spec.split,
        "dataset": context.info.name, "class_names": list(context.class_names),
        "subjects": subjects,
        "sections": [msgspec.structs.asdict(s) if isinstance(s, msgspec.Struct) else s.__dict__
                     for s in sections],
    }
    (report.path / "report.json").write_bytes(
        msgspec.json.format(msgspec.json.encode(document, enc_hook=_plain), indent=2) + b"\n")
    header = [
        f"# Report {report.id}",
        "",
        f"{context.info.title}, {context.spec.split} split ({context.size} images).",
        "",
        md_table(["model", "run", "optimizer", "network"],
                 [[s.label, str(s.ref), s.spec.optimizer, s.spec.model] for s in context.subjects]),
    ]
    body = []
    for section in sections:
        body += ["", f"## {section.title}", "", section.markdown.strip()]
        body += [f"\n![{section.title}]({figure})" for figure in section.figures]
    (report.path / "report.md").write_text("\n".join(header + body) + "\n")


def main(report: Report) -> int:
    try:
        return build(report)
    except BaseException as error:
        traceback.print_exc()
        status = report.status()
        status.state, status.error = "failed", f"{type(error).__name__}: {error}"
        report.write_status(status)
        return 1
