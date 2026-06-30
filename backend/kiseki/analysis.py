from __future__ import annotations

import hashlib
import json
import secrets
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Literal

import numpy as np
import torch
from captum.attr import LRP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .checkpoint import CheckpointSaver, checkpoint_summary_from_metadata
from .data import CIFAR10_MEAN, CIFAR10_STD, DataLoaderFactory, test_loader
from .dataset_types import DatasetName
from .models import build_model
from .schemas import (
    AnalysisActivationLayerStats,
    AnalysisActivationReport,
    AnalysisCalibration,
    AnalysisCalibrationBin,
    AnalysisClassAverageRelevance,
    AnalysisComparisonJobRequest,
    AnalysisComparisonJobStatus,
    AnalysisComparisonParams,
    AnalysisComparisonReport,
    AnalysisCurves,
    AnalysisEmbeddingPoint,
    AnalysisEmbeddingProjection,
    AnalysisEmbeddings,
    AnalysisHistogram,
    AnalysisLrpReport,
    AnalysisLrpSample,
    AnalysisModelMetrics,
    AnalysisOverlap,
    AnalysisPerClassMetric,
    AnalysisRobustnessCurve,
    AnalysisRobustnessPoint,
    AnalysisRuntimeReport,
    AnalysisTableRow,
    AnalysisWeightLayerComparison,
    CheckpointSelection,
    CheckpointSummary,
    ExperimentConfig,
    ExperimentStatus,
)


ANALYSIS_VERSION = "comparison-v3"
CLASS_COUNT = 10
LRP_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 512
HISTOGRAM_BINS = 20
EMBEDDING_TSNE_SIDE_LIMIT = 2000
EMBEDDING_PCA_TOTAL_LIMIT = 2000

MNIST_CLASS_LABELS = tuple(str(label) for label in range(CLASS_COUNT))
FASHION_MNIST_CLASS_LABELS = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
CIFAR10_CLASS_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class AnalysisComparisonError(ValueError):
    pass


class AnalysisParameterError(ValueError):
    pass


@dataclass(slots=True)
class ModelEvaluation:
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    mean_loss: float
    embeddings: np.ndarray
    activations: dict[str, np.ndarray]


@dataclass(slots=True)
class AnalysisJob:
    status: AnalysisComparisonJobStatus
    report: AnalysisComparisonReport | None = None
    subscribers: set[Queue[AnalysisComparisonJobStatus]] = field(default_factory=set)


class AnalysisService:
    def __init__(
        self,
        *,
        data_loader_factory: DataLoaderFactory,
        checkpoint_saver: CheckpointSaver,
        is_experiment_running: Callable[[], bool] | None = None,
    ) -> None:
        self.data_loader_factory = data_loader_factory
        self.checkpoint_saver = checkpoint_saver
        self.is_experiment_running = is_experiment_running or (lambda: False)
        self.lock = threading.Lock()
        self.jobs: dict[str, AnalysisJob] = {}

    def checkpoint_summaries(self) -> list[CheckpointSummary]:
        return self.checkpoint_saver.list_analysis_summaries()

    def create_comparison_job(
        self,
        request: AnalysisComparisonJobRequest,
    ) -> AnalysisComparisonJobStatus:
        left_payload = self._load_payload(request.left)
        right_payload = self._load_payload(request.right)
        left_config = ExperimentConfig.model_validate(left_payload["config"])
        right_config = ExperimentConfig.model_validate(right_payload["config"])
        validate_comparable_checkpoints(left_config, right_config)

        left_summary = checkpoint_summary_from_payload(
            self.checkpoint_saver,
            left_payload,
            request.left,
        )
        right_summary = checkpoint_summary_from_payload(
            self.checkpoint_saver,
            right_payload,
            request.right,
        )
        fingerprints = {
            "left": checkpoint_fingerprint(self.checkpoint_saver, request.left),
            "right": checkpoint_fingerprint(self.checkpoint_saver, request.right),
        }
        cache_key = comparison_cache_key(request)
        cached = read_cached_report(self.cache_path(cache_key))

        if cached is not None and not request.force_recompute:
            stale_sides = stale_cache_sides(cached, fingerprints)
            report = AnalysisComparisonReport.model_validate(cached["report"])
            return self._store_completed_job(
                cache_state="stale" if stale_sides else "fresh",
                message=(
                    "Cached report is stale; recompute is available."
                    if stale_sides
                    else "Loaded cached report."
                ),
                report=report,
                stale_sides=stale_sides,
            )

        job_id = secrets.token_urlsafe(12)
        cache_state: Literal["miss", "recomputed"] = (
            "recomputed" if request.force_recompute else "miss"
        )
        status = AnalysisComparisonJobStatus(
            job_id=job_id,
            status="queued",
            progress=0.0,
            stage="load/cache",
            message="Queued comparison job.",
            cache_state=cache_state,
        )
        with self.lock:
            self.jobs[job_id] = AnalysisJob(status=status)

        worker = threading.Thread(
            target=self._run_comparison_job,
            args=(
                job_id,
                request,
                cache_key,
                fingerprints,
                left_summary,
                right_summary,
            ),
            daemon=True,
        )
        worker.start()
        return status

    def get_comparison_job(self, job_id: str) -> AnalysisComparisonJobStatus:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.status.model_copy(deep=True)

    def get_comparison_report(self, job_id: str) -> AnalysisComparisonReport:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.report is None:
                raise AnalysisComparisonError("comparison report is not available")
            return job.report.model_copy(deep=True)

    def comparison_events(self, job_id: str) -> Iterator[str]:
        queue: Queue[AnalysisComparisonJobStatus] = Queue()
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.subscribers.add(queue)
            initial_status = job.status.model_copy(deep=True)

        yield format_sse("status", initial_status)
        try:
            while True:
                try:
                    status = queue.get(timeout=15)
                except Empty:
                    yield ": heartbeat\n\n"
                    continue

                event_type = "completed" if status.status == "completed" else status.status
                yield format_sse(event_type, status)
                if status.status in {"completed", "failed"}:
                    return
        finally:
            with self.lock:
                job = self.jobs.get(job_id)
                if job is not None:
                    job.subscribers.discard(queue)

    def cache_path(self, cache_key: str) -> Path:
        return self.checkpoint_saver.directory / "_analysis_cache" / f"{cache_key}.json"

    def _load_payload(self, selection: CheckpointSelection) -> dict[str, Any]:
        return self.checkpoint_saver.load(
            selection.run_id,
            selection.kind,
            map_location="cpu",
        )

    def _store_completed_job(
        self,
        *,
        cache_state: Literal["fresh", "stale"],
        message: str,
        report: AnalysisComparisonReport,
        stale_sides: list[Literal["left", "right"]],
    ) -> AnalysisComparisonJobStatus:
        job_id = secrets.token_urlsafe(12)
        status = AnalysisComparisonJobStatus(
            job_id=job_id,
            status="completed",
            progress=1.0,
            stage="persist",
            message=message,
            cache_state=cache_state,
            stale_sides=stale_sides,
            report_available=True,
        )
        with self.lock:
            self.jobs[job_id] = AnalysisJob(status=status, report=report)
        return status

    def _run_comparison_job(
        self,
        job_id: str,
        request: AnalysisComparisonJobRequest,
        cache_key: str,
        fingerprints: dict[str, str],
        left_summary: CheckpointSummary,
        right_summary: CheckpointSummary,
    ) -> None:
        try:
            device = resolve_analysis_device(self.is_experiment_running())

            def progress(stage: str, message: str, value: float) -> None:
                self._update_job(
                    job_id,
                    status="running",
                    progress=value,
                    stage=stage,
                    message=message,
                )

            progress("load/cache", "Loading checkpoints.", 0.04)
            report = build_comparison_report(
                request,
                left_summary=left_summary,
                right_summary=right_summary,
                data_loader_factory=self.data_loader_factory,
                checkpoint_saver=self.checkpoint_saver,
                device=device,
                progress=progress,
            )
            progress("persist", "Writing report cache.", 0.98)
            write_cached_report(
                self.cache_path(cache_key),
                report=report,
                fingerprints=fingerprints,
                request=request,
            )
            self._update_job(
                job_id,
                status="completed",
                progress=1.0,
                stage="persist",
                message="Comparison complete.",
                report=report,
            )
        except Exception as exc:  # pragma: no cover - surfaced through API status.
            self._update_job(
                job_id,
                status="failed",
                progress=1.0,
                stage="failed",
                message="Comparison failed.",
                error=str(exc),
            )

    def _update_job(
        self,
        job_id: str,
        *,
        status: Literal["queued", "running", "completed", "failed"],
        progress: float,
        stage: str,
        message: str,
        report: AnalysisComparisonReport | None = None,
        error: str | None = None,
    ) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job.status.status = status
            job.status.progress = clamp_float(progress, 0.0, 1.0)
            job.status.stage = stage
            job.status.message = message
            if report is not None:
                job.report = report
                job.status.report_available = True
            if error is not None:
                job.status.error = error
            snapshot = job.status.model_copy(deep=True)
            subscribers = tuple(job.subscribers)

        for queue in subscribers:
            queue.put(snapshot)


def build_comparison_report(
    request: AnalysisComparisonJobRequest,
    *,
    left_summary: CheckpointSummary,
    right_summary: CheckpointSummary,
    data_loader_factory: DataLoaderFactory,
    checkpoint_saver: CheckpointSaver,
    device: torch.device,
    progress: Callable[[str, str, float], None],
) -> AnalysisComparisonReport:
    left_payload = checkpoint_saver.load(
        request.left.run_id,
        request.left.kind,
        map_location="cpu",
    )
    right_payload = checkpoint_saver.load(
        request.right.run_id,
        request.right.kind,
        map_location="cpu",
    )
    left_config = ExperimentConfig.model_validate(left_payload["config"])
    right_config = ExperimentConfig.model_validate(right_payload["config"])
    validate_comparable_checkpoints(left_config, right_config)

    dataset = left_config.dataset
    labels_names = class_names(dataset)
    progress("inference", "Loading test set.", 0.08)
    inputs, labels = collect_test_inputs(test_loader(data_loader_factory, dataset))
    if labels.numel() == 0:
        raise AnalysisComparisonError("test loader did not return any samples")

    left_model = load_model(left_payload, dataset, device)
    right_model = load_model(right_payload, dataset, device)
    left_label, right_label = comparison_side_labels(left_summary, right_summary)

    progress("inference", f"Running {left_label} over the full test set.", 0.14)
    left_eval = evaluate_model(left_model, inputs, labels, device=device)
    progress("inference", f"Running {right_label} over the full test set.", 0.25)
    right_eval = evaluate_model(right_model, inputs, labels, device=device)

    progress("metrics", "Computing metrics, confusion, and calibration.", 0.34)
    left_metrics = model_metrics(left_eval, labels_names, request.params.calibration_bins)
    right_metrics = model_metrics(right_eval, labels_names, request.params.calibration_bins)
    confusion_delta = (
        np.asarray(left_metrics.confusion_matrix, dtype=np.int64)
        - np.asarray(right_metrics.confusion_matrix, dtype=np.int64)
    ).tolist()
    overlap = overlap_report(
        labels.numpy(),
        left_eval.predictions,
        right_eval.predictions,
    )

    progress("embeddings", "Generating joint PCA and t-SNE projections.", 0.46)
    embeddings = embedding_report(
        left_eval,
        right_eval,
        params=request.params,
    )

    progress("LRP", "Computing aggregate and representative LRP maps.", 0.62)
    lrp = lrp_report(
        left_model,
        right_model,
        inputs,
        labels,
        left_eval,
        right_eval,
        dataset=dataset,
        params=request.params,
        device=device,
    )

    progress("activation/weights", "Summarizing activations and weights.", 0.76)
    activations = AnalysisActivationReport(
        left=activation_stats(left_eval.activations),
        right=activation_stats(right_eval.activations),
    )
    weights = weight_comparisons(left_model, right_model)

    progress("robustness", "Evaluating robustness curves.", 0.86)
    robustness = robustness_report(
        left_model,
        right_model,
        inputs,
        labels,
        dataset=dataset,
        params=request.params,
        device=device,
    )

    left_status = ExperimentStatus.model_validate(left_payload["status"])
    right_status = ExperimentStatus.model_validate(right_payload["status"])
    generated_at = datetime.now(UTC).isoformat()
    return AnalysisComparisonReport(
        analysis_version=ANALYSIS_VERSION,
        generated_at=generated_at,
        analysis_device=device.type,
        left=left_summary,
        right=right_summary,
        params=request.params,
        metadata=metadata_rows(left_summary, right_summary),
        comparability=comparability_rows(left_config, right_config),
        curves={
            "left": curves_from_status(left_status),
            "right": curves_from_status(right_status),
        },
        metrics={"left": left_metrics, "right": right_metrics},
        confusion_difference=confusion_delta,
        overlap=overlap,
        embeddings=embeddings,
        lrp=lrp,
        activations=activations,
        weights=weights,
        robustness=robustness,
        runtime=AnalysisRuntimeReport(
            rows=runtime_rows(left_summary, right_summary, left_status, right_status)
        ),
    )


def collect_test_inputs(loader: torch.utils.data.DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_inputs, batch_labels in loader:
        inputs.append(batch_inputs.detach().cpu())
        labels.append(torch.as_tensor(batch_labels, dtype=torch.long).detach().cpu())
    if not inputs:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(inputs), torch.cat(labels)


def load_model(
    payload: dict[str, Any],
    dataset: DatasetName,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(dataset)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
) -> ModelEvaluation:
    all_predictions: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    all_embeddings: list[torch.Tensor] = []
    activation_batches: dict[str, list[torch.Tensor]] = defaultdict(list)
    loss_sum = 0.0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels in tensor_batches(inputs, labels, EVAL_BATCH_SIZE):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs)
            probabilities = torch.softmax(logits, dim=1)
            loss_sum += float(
                torch.nn.functional.cross_entropy(logits, batch_labels, reduction="sum").cpu()
            )
            total += int(batch_labels.numel())
            all_predictions.append(logits.argmax(dim=1).detach().cpu())
            all_probabilities.append(probabilities.detach().cpu())

            final_hidden = getattr(model, "final_hidden", None)
            if not callable(final_hidden):
                raise AnalysisComparisonError("model does not expose final_hidden activations")
            all_embeddings.append(final_hidden(batch_inputs).detach().cpu())

            named_activations = getattr(model, "named_activations", None)
            if callable(named_activations):
                for name, activation in named_activations(batch_inputs).items():
                    activation_batches[str(name)].append(activation.detach().flatten().cpu())

    if total == 0:
        raise AnalysisComparisonError("test loader did not return any samples")

    activations = {
        name: torch.cat(values).numpy()
        for name, values in activation_batches.items()
        if values
    }
    return ModelEvaluation(
        labels=labels.numpy(),
        predictions=torch.cat(all_predictions).numpy(),
        probabilities=torch.cat(all_probabilities).numpy(),
        mean_loss=loss_sum / total,
        embeddings=torch.cat(all_embeddings).numpy(),
        activations=activations,
    )


def evaluate_accuracy_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[float, float]:
    correct = 0
    loss_sum = 0.0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels in tensor_batches(inputs, labels, EVAL_BATCH_SIZE):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs)
            loss_sum += float(
                torch.nn.functional.cross_entropy(logits, batch_labels, reduction="sum").cpu()
            )
            correct += int((logits.argmax(dim=1) == batch_labels).sum().cpu())
            total += int(batch_labels.numel())
    return 100.0 * correct / max(total, 1), loss_sum / max(total, 1)


def tensor_batches(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for start in range(0, labels.numel(), batch_size):
        end = min(start + batch_size, labels.numel())
        yield inputs[start:end], labels[start:end]


def model_metrics(
    evaluation: ModelEvaluation,
    labels_names: tuple[str, ...],
    calibration_bins: int,
) -> AnalysisModelMetrics:
    labels = evaluation.labels
    predictions = evaluation.predictions
    class_indices = list(range(CLASS_COUNT))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_indices,
        zero_division=0,
    )
    confusion = confusion_matrix(labels, predictions, labels=class_indices)
    accuracy = 100.0 * float(np.mean(predictions == labels))
    return AnalysisModelMetrics(
        accuracy=safe_float(accuracy),
        loss=safe_float(evaluation.mean_loss),
        macro_f1=safe_float(float(np.mean(f1))),
        per_class_f1=[
            AnalysisPerClassMetric(
                label=label,
                name=labels_names[label],
                precision=safe_float(float(precision[label])),
                recall=safe_float(float(recall[label])),
                f1=safe_float(float(f1[label])),
                support=int(support[label]),
            )
            for label in class_indices
        ],
        confusion_matrix=confusion.astype(int).tolist(),
        calibration=calibration_report(evaluation.probabilities, labels, calibration_bins),
    )


def calibration_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bin_count: int,
) -> AnalysisCalibration:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    confidences = clipped.max(axis=1)
    predictions = clipped.argmax(axis=1)
    correct = predictions == labels
    one_hot = np.eye(CLASS_COUNT, dtype=np.float64)[labels]
    brier_score = float(np.mean(np.sum((clipped - one_hot) ** 2, axis=1)))
    nll = float(-np.mean(np.log(clipped[np.arange(labels.shape[0]), labels])))
    bins: list[AnalysisCalibrationBin] = []
    ece = 0.0

    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if index == bin_count - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidences[mask].mean())
            mean_accuracy = float(correct[mask].mean())
            ece += (count / labels.shape[0]) * abs(mean_accuracy - mean_confidence)
        else:
            mean_confidence = 0.0
            mean_accuracy = 0.0
        bins.append(
            AnalysisCalibrationBin(
                lower=safe_float(lower),
                upper=safe_float(upper),
                count=count,
                confidence=safe_float(mean_confidence),
                accuracy=safe_float(mean_accuracy),
            )
        )

    return AnalysisCalibration(
        bins=bins,
        ece=safe_float(ece),
        brier_score=safe_float(brier_score),
        nll=safe_float(nll),
    )


def overlap_report(
    labels: np.ndarray,
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
) -> AnalysisOverlap:
    left_correct = left_predictions == labels
    right_correct = right_predictions == labels
    both_error = ~left_correct & ~right_correct
    both_error_same = both_error & (left_predictions == right_predictions)
    both_error_different = both_error & (left_predictions != right_predictions)
    rows = [
        ("correct_both", left_correct & right_correct),
        ("left_only_correct", left_correct & ~right_correct),
        ("right_only_correct", ~left_correct & right_correct),
        ("error_both_same_prediction", both_error_same),
        ("error_both_different_prediction", both_error_different),
        ("disagreements", left_predictions != right_predictions),
    ]
    return AnalysisOverlap(
        total=int(labels.shape[0]),
        correct_both=int((left_correct & right_correct).sum()),
        left_only_correct=int((left_correct & ~right_correct).sum()),
        right_only_correct=int((~left_correct & right_correct).sum()),
        error_both=int(both_error.sum()),
        disagreements=int((left_predictions != right_predictions).sum()),
        both_error_same_prediction=int(both_error_same.sum()),
        both_error_different_prediction=int(both_error_different.sum()),
        upset=[{"set": name, "count": int(mask.sum())} for name, mask in rows],
    )


def embedding_report(
    left_eval: ModelEvaluation,
    right_eval: ModelEvaluation,
    *,
    params: AnalysisComparisonParams,
) -> AnalysisEmbeddings:
    features = np.concatenate([left_eval.embeddings, right_eval.embeddings], axis=0)
    validate_tsne_params(params, sample_count=features.shape[0])
    pca_coordinates = pca_embedding(features)
    tsne_coordinates = tsne_embedding(features, params)
    split = left_eval.embeddings.shape[0]
    pca_left = embedding_points(pca_coordinates[:split], left_eval)
    pca_right = embedding_points(pca_coordinates[split:], right_eval)
    tsne_left = embedding_points(tsne_coordinates[:split], left_eval)
    tsne_right = embedding_points(tsne_coordinates[split:], right_eval)
    return AnalysisEmbeddings(
        pca=sample_embedding_projection(
            pca_left,
            pca_right,
            coordinates=pca_coordinates,
            total_limit=EMBEDDING_PCA_TOTAL_LIMIT,
        ),
        tsne=sample_embedding_projection(
            tsne_left,
            tsne_right,
            coordinates=tsne_coordinates,
            side_limit=EMBEDDING_TSNE_SIDE_LIMIT,
        ),
    )


def pca_embedding(features: np.ndarray) -> np.ndarray:
    feature_matrix = np.asarray(features, dtype=np.float32)
    if feature_matrix.shape[0] < 2:
        return np.zeros((feature_matrix.shape[0], 2), dtype=np.float32)
    components = min(2, feature_matrix.shape[0], feature_matrix.shape[1])
    coordinates = PCA(n_components=components, random_state=0).fit_transform(feature_matrix)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(coordinates.shape[0])])
    return coordinates


def tsne_embedding(features: np.ndarray, params: AnalysisComparisonParams) -> np.ndarray:
    feature_matrix = np.asarray(features, dtype=np.float32)
    if feature_matrix.shape[0] < 2:
        return np.zeros((feature_matrix.shape[0], 2), dtype=np.float32)

    component_count = min(
        params.tsne_pca_components,
        feature_matrix.shape[0],
        feature_matrix.shape[1],
    )
    if component_count >= 2:
        feature_matrix = PCA(
            n_components=component_count,
            random_state=params.tsne_seed,
        ).fit_transform(feature_matrix)

    learning_rate: float | str
    if params.tsne_learning_rate_mode == "auto":
        learning_rate = "auto"
    else:
        learning_rate = float(params.tsne_learning_rate)

    return TSNE(
        n_components=2,
        perplexity=params.tsne_perplexity,
        max_iter=params.tsne_max_iter,
        learning_rate=learning_rate,
        angle=params.tsne_angle,
        method="barnes_hut",
        init="random",
        random_state=params.tsne_seed,
    ).fit_transform(feature_matrix)


def validate_tsne_params(params: AnalysisComparisonParams, *, sample_count: int) -> None:
    if sample_count < 2:
        raise AnalysisParameterError("sample_count must be at least 2")
    if params.tsne_perplexity >= sample_count:
        raise AnalysisParameterError("tsne_perplexity must be less than sample_count")


def embedding_points(
    coordinates: np.ndarray,
    evaluation: ModelEvaluation,
) -> list[AnalysisEmbeddingPoint]:
    return [
        AnalysisEmbeddingPoint(
            index=index,
            x=safe_float(float(point[0])),
            y=safe_float(float(point[1])),
            label=int(label),
            prediction=int(prediction),
            correct=bool(label == prediction),
        )
        for index, (point, label, prediction) in enumerate(
            zip(coordinates, evaluation.labels, evaluation.predictions, strict=True)
        )
    ]


def sample_embedding_projection(
    left: list[AnalysisEmbeddingPoint],
    right: list[AnalysisEmbeddingPoint],
    *,
    coordinates: np.ndarray,
    side_limit: int | None = None,
    total_limit: int | None = None,
) -> AnalysisEmbeddingProjection:
    if side_limit is not None:
        sampled_left = sample_embedding_points(left, side_limit)
        sampled_right = sample_embedding_points(right, side_limit)
    elif total_limit is not None:
        sampled_left, sampled_right = sample_embedding_points_by_total(
            left,
            right,
            total_limit,
        )
    else:
        sampled_left = sort_embedding_points(left)
        sampled_right = sort_embedding_points(right)

    return AnalysisEmbeddingProjection(
        left=sampled_left,
        right=sampled_right,
        left_total=len(left),
        right_total=len(right),
        x_domain=padded_embedding_domain(coordinates[:, 0]),
        y_domain=padded_embedding_domain(coordinates[:, 1]),
    )


def sample_embedding_points(
    points: list[AnalysisEmbeddingPoint],
    limit: int,
) -> list[AnalysisEmbeddingPoint]:
    if len(points) <= limit:
        return sort_embedding_points(points)
    return sample_grouped_embedding_items(
        points,
        limit,
        key=lambda point: f"{point.label}:{int(point.correct)}",
    )


def sample_embedding_points_by_total(
    left: list[AnalysisEmbeddingPoint],
    right: list[AnalysisEmbeddingPoint],
    limit: int,
) -> tuple[list[AnalysisEmbeddingPoint], list[AnalysisEmbeddingPoint]]:
    combined: list[tuple[Literal["left", "right"], AnalysisEmbeddingPoint]] = [
        ("left", point) for point in left
    ] + [("right", point) for point in right]
    if len(combined) <= limit:
        return sort_embedding_points(left), sort_embedding_points(right)

    sampled = sample_grouped_embedding_items(
        combined,
        limit,
        key=lambda item: f"{item[0]}:{item[1].label}:{int(item[1].correct)}",
    )
    sampled_left = [point for side, point in sampled if side == "left"]
    sampled_right = [point for side, point in sampled if side == "right"]
    return sort_embedding_points(sampled_left), sort_embedding_points(sampled_right)


def sample_grouped_embedding_items[T](
    items: list[T],
    limit: int,
    *,
    key: Callable[[T], str],
) -> list[T]:
    if limit <= 0 or not items:
        return []

    groups: dict[str, list[T]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)

    sorted_groups = [
        {"key": group_key, "items": group_items, "quota": 0}
        for group_key, group_items in sorted(groups.items())
    ]
    if len(sorted_groups) >= limit:
        return [
            item
            for group in sorted_groups[:limit]
            for item in evenly_spaced_sample(group["items"], 1)
        ]

    for group in sorted_groups:
        group["quota"] = 1

    remaining = limit - len(sorted_groups)
    total_capacity = sum(max(0, len(group["items"]) - 1) for group in sorted_groups)
    remainders = []
    for group in sorted_groups:
        capacity = max(0, len(group["items"]) - 1)
        exact_share = (capacity / total_capacity) * remaining if total_capacity else 0.0
        extra = min(capacity, int(np.floor(exact_share)))
        group["quota"] += extra
        remainders.append(
            {
                "capacity": capacity - extra,
                "fraction": exact_share - np.floor(exact_share),
                "group": group,
            }
        )

    remaining = limit - sum(int(group["quota"]) for group in sorted_groups)
    for item in sorted(
        (item for item in remainders if item["capacity"] > 0),
        key=lambda value: (-float(value["fraction"]), str(value["group"]["key"])),
    ):
        if remaining <= 0:
            break
        item["group"]["quota"] += 1
        remaining -= 1

    return [
        item
        for group in sorted_groups
        for item in evenly_spaced_sample(group["items"], int(group["quota"]))
    ]


def evenly_spaced_sample[T](items: list[T], count: int) -> list[T]:
    if count <= 0:
        return []
    if len(items) <= count:
        return items
    if count == 1:
        return [items[(len(items) - 1) // 2]]
    return [
        items[round((index * (len(items) - 1)) / (count - 1))]
        for index in range(count)
    ]


def sort_embedding_points(
    points: list[AnalysisEmbeddingPoint],
) -> list[AnalysisEmbeddingPoint]:
    return sorted(points, key=lambda point: (point.label, int(point.correct), point.index))


def padded_embedding_domain(values: np.ndarray) -> tuple[float, float]:
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.shape[0] == 0:
        return (0.0, 1.0)

    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    if minimum == maximum:
        padding = max(1.0, abs(minimum) * 0.05)
    else:
        padding = (maximum - minimum) * 0.05
    return (safe_float(minimum - padding), safe_float(maximum + padding))


def lrp_report(
    left_model: torch.nn.Module,
    right_model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    left_eval: ModelEvaluation,
    right_eval: ModelEvaluation,
    *,
    dataset: DatasetName,
    params: AnalysisComparisonParams,
    device: torch.device,
) -> AnalysisLrpReport:
    sample_indices = representative_lrp_indices(
        labels.numpy(),
        left_eval.predictions,
        right_eval.predictions,
        params.lrp_gallery_sample_count,
    )
    average_indices = class_average_indices(labels.numpy(), labels.numel())
    all_indices = sorted(set(sample_indices).union(*average_indices.values()))
    left_relevance = relevance_maps(left_model, inputs, labels, all_indices, device=device)
    right_relevance = relevance_maps(right_model, inputs, labels, all_indices, device=device)
    labels_names = class_names(dataset)

    samples = [
        AnalysisLrpSample(
            index=index,
            label=int(labels[index]),
            label_name=labels_names[int(labels[index])],
            group=sample_group(
                int(labels[index]),
                int(left_eval.predictions[index]),
                int(right_eval.predictions[index]),
            ),
            left_prediction=int(left_eval.predictions[index]),
            right_prediction=int(right_eval.predictions[index]),
            left_confidence=safe_float(float(left_eval.probabilities[index].max())),
            right_confidence=safe_float(float(right_eval.probabilities[index].max())),
            image=rgb_image(inputs[index], dataset),
            left_relevance=normalized_signed_relevance(left_relevance[index]),
            right_relevance=normalized_signed_relevance(right_relevance[index]),
            difference_relevance=normalized_signed_relevance(
                left_relevance[index] - right_relevance[index]
            ),
        )
        for index in sample_indices
    ]

    class_averages: list[AnalysisClassAverageRelevance] = []
    height = int(inputs.shape[-2])
    width = int(inputs.shape[-1])
    empty_map = torch.zeros((height, width), dtype=torch.float32)
    for label in range(CLASS_COUNT):
        indices = average_indices[label]
        if indices:
            left_average = torch.stack([left_relevance[index] for index in indices]).mean(dim=0)
            right_average = torch.stack([right_relevance[index] for index in indices]).mean(dim=0)
        else:
            left_average = empty_map
            right_average = empty_map
        class_averages.append(
            AnalysisClassAverageRelevance(
                label=label,
                name=labels_names[label],
                left_relevance=normalized_signed_relevance(left_average),
                right_relevance=normalized_signed_relevance(right_average),
                difference_relevance=normalized_signed_relevance(left_average - right_average),
            )
        )

    return AnalysisLrpReport(samples=samples, class_averages=class_averages)


def relevance_maps(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    indices: list[int],
    *,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    if not indices:
        return {}
    lrp = LRP(model)
    output: dict[int, torch.Tensor] = {}
    for start in range(0, len(indices), LRP_BATCH_SIZE):
        batch_indices = indices[start : start + LRP_BATCH_SIZE]
        batch_inputs = inputs[batch_indices].to(device).clone().detach().requires_grad_(True)
        batch_targets = labels[batch_indices].to(device)
        model.zero_grad(set_to_none=True)
        try:
            attributions = lrp.attribute(batch_inputs, target=batch_targets)
        except Exception:
            attributions = gradient_relevance(model, batch_inputs, batch_targets)
        if not isinstance(attributions, torch.Tensor):
            raise AnalysisComparisonError("LRP returned multiple attribution tensors")

        for index, attribution in zip(batch_indices, attributions.detach().cpu(), strict=True):
            output[index] = torch.nan_to_num(attribution.float().sum(dim=0))
    return output


def gradient_relevance(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if inputs.grad is not None:
        inputs.grad.zero_()
    logits = model(inputs)
    selected = logits.gather(1, targets.unsqueeze(1)).sum()
    selected.backward()
    gradients = inputs.grad
    if gradients is None:
        return torch.zeros_like(inputs)
    return gradients * inputs


def representative_lrp_indices(
    labels: np.ndarray,
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    sample_count: int,
) -> list[int]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, (label, left_prediction, right_prediction) in enumerate(
        zip(labels, left_predictions, right_predictions, strict=True)
    ):
        buckets[sample_group(int(label), int(left_prediction), int(right_prediction))].append(index)

    priorities = [
        "disagreement",
        "left_only_correct",
        "right_only_correct",
        "error_both_different_prediction",
        "error_both_same_prediction",
        "correct_both",
    ]
    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < min(sample_count, labels.shape[0]):
        added = False
        for group in priorities:
            bucket = buckets[group]
            while bucket and bucket[0] in selected_set:
                bucket.pop(0)
            if bucket and len(selected) < sample_count:
                index = bucket.pop(0)
                selected.append(index)
                selected_set.add(index)
                added = True
        if not added:
            break
    return selected


def class_average_indices(labels: np.ndarray, limit_per_class: int) -> dict[int, list[int]]:
    indices: dict[int, list[int]] = {}
    for label in range(CLASS_COUNT):
        class_indices = np.where(labels == label)[0].tolist()
        indices[label] = class_indices[:limit_per_class]
    return indices


def sample_group(label: int, left_prediction: int, right_prediction: int) -> str:
    left_correct = left_prediction == label
    right_correct = right_prediction == label
    if left_prediction != right_prediction:
        if left_correct:
            return "left_only_correct"
        if right_correct:
            return "right_only_correct"
        return "disagreement"
    if left_correct and right_correct:
        return "correct_both"
    return "error_both_same_prediction"


def rgb_image(input_tensor: torch.Tensor, dataset: DatasetName) -> list[list[list[float]]]:
    image = denormalize_inputs(input_tensor.unsqueeze(0), dataset)[0]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = torch.nan_to_num(image).clamp(0.0, 1.0)
    return image.permute(1, 2, 0).tolist()


def normalized_signed_relevance(relevance: torch.Tensor) -> list[list[float]]:
    normalized = torch.nan_to_num(relevance.detach().cpu().float())
    max_abs = float(normalized.abs().max())
    if max_abs > 0.0:
        normalized = normalized / max_abs
    return normalized.clamp(-1.0, 1.0).tolist()


def activation_stats(activations: dict[str, np.ndarray]) -> list[AnalysisActivationLayerStats]:
    return [
        activation_layer_stats(name, values)
        for name, values in sorted(activations.items(), key=lambda item: item[0])
    ]


def activation_layer_stats(name: str, values: np.ndarray) -> AnalysisActivationLayerStats:
    safe_values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    if safe_values.size == 0:
        safe_values = np.zeros(1, dtype=np.float64)
    counts, edges = np.histogram(safe_values, bins=HISTOGRAM_BINS)
    return AnalysisActivationLayerStats(
        name=name,
        sparsity=safe_float(float(np.mean(np.abs(safe_values) <= 1e-8))),
        mean=safe_float(float(np.mean(safe_values))),
        std=safe_float(float(np.std(safe_values))),
        q05=safe_float(float(np.quantile(safe_values, 0.05))),
        q50=safe_float(float(np.quantile(safe_values, 0.50))),
        q95=safe_float(float(np.quantile(safe_values, 0.95))),
        histogram=AnalysisHistogram(
            bins=[safe_float(float(edge)) for edge in edges],
            counts=[int(count) for count in counts],
        ),
    )


def weight_comparisons(
    left_model: torch.nn.Module,
    right_model: torch.nn.Module,
) -> list[AnalysisWeightLayerComparison]:
    right_state = right_model.state_dict()
    comparisons: list[AnalysisWeightLayerComparison] = []
    for name, left_tensor in left_model.state_dict().items():
        right_tensor = right_state.get(name)
        if right_tensor is None:
            continue
        if left_tensor.shape != right_tensor.shape or not torch.is_floating_point(left_tensor):
            continue
        left_flat = left_tensor.detach().cpu().float().flatten()
        right_flat = right_tensor.detach().cpu().float().flatten()
        left_norm = float(torch.linalg.vector_norm(left_flat))
        right_norm = float(torch.linalg.vector_norm(right_flat))
        distance = float(torch.linalg.vector_norm(left_flat - right_flat))
        denominator = max((left_norm + right_norm) / 2.0, 1e-12)
        comparisons.append(
            AnalysisWeightLayerComparison(
                name=name,
                left_norm=safe_float(left_norm),
                right_norm=safe_float(right_norm),
                distance=safe_float(distance),
                relative_distance=safe_float(distance / denominator),
            )
        )
    return comparisons


def robustness_report(
    left_model: torch.nn.Module,
    right_model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    dataset: DatasetName,
    params: AnalysisComparisonParams,
    device: torch.device,
) -> list[AnalysisRobustnessCurve]:
    pixel_inputs = denormalize_inputs(inputs, dataset)
    perturbations = [
        ("gaussian_noise", params.robustness_noise_levels),
        ("brightness", params.robustness_brightness_levels),
        ("center_cutout", params.robustness_cutout_levels),
    ]
    curves: list[AnalysisRobustnessCurve] = []
    for perturbation, levels in perturbations:
        points: list[AnalysisRobustnessPoint] = []
        for level in levels:
            perturbed = perturb_pixels(pixel_inputs, perturbation, level)
            normalized = normalize_inputs(perturbed, dataset)
            left_accuracy, left_loss = evaluate_accuracy_loss(
                left_model,
                normalized,
                labels,
                device=device,
            )
            right_accuracy, right_loss = evaluate_accuracy_loss(
                right_model,
                normalized,
                labels,
                device=device,
            )
            points.append(
                AnalysisRobustnessPoint(
                    level=safe_float(level),
                    left_accuracy=safe_float(left_accuracy),
                    right_accuracy=safe_float(right_accuracy),
                    left_loss=safe_float(left_loss),
                    right_loss=safe_float(right_loss),
                )
            )
        curves.append(AnalysisRobustnessCurve(perturbation=perturbation, points=points))
    return curves


def perturb_pixels(inputs: torch.Tensor, perturbation: str, level: float) -> torch.Tensor:
    outputs = inputs.clone()
    if perturbation == "gaussian_noise" and level > 0:
        generator = torch.Generator().manual_seed(17 + int(level * 10_000))
        outputs = outputs + torch.randn(outputs.shape, generator=generator) * level
    elif perturbation == "brightness":
        outputs = outputs + level
    elif perturbation == "center_cutout" and level > 0:
        height = outputs.shape[-2]
        width = outputs.shape[-1]
        size = int(round(min(height, width) * clamp_float(level, 0.0, 1.0)))
        if size > 0:
            top = max((height - size) // 2, 0)
            left = max((width - size) // 2, 0)
            outputs[..., top : top + size, left : left + size] = 0.0
    return outputs.clamp(0.0, 1.0)


def denormalize_inputs(inputs: torch.Tensor, dataset: DatasetName) -> torch.Tensor:
    outputs = inputs.detach().cpu().float()
    if dataset == "cifar10":
        mean = torch.tensor(CIFAR10_MEAN, dtype=outputs.dtype).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD, dtype=outputs.dtype).view(1, 3, 1, 1)
        outputs = outputs * std + mean
    return outputs.clamp(0.0, 1.0)


def normalize_inputs(inputs: torch.Tensor, dataset: DatasetName) -> torch.Tensor:
    outputs = inputs.detach().cpu().float()
    if dataset == "cifar10":
        mean = torch.tensor(CIFAR10_MEAN, dtype=outputs.dtype).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD, dtype=outputs.dtype).view(1, 3, 1, 1)
        outputs = (outputs - mean) / std
    return outputs


def comparison_side_labels(
    left: CheckpointSummary,
    right: CheckpointSummary,
) -> tuple[str, str]:
    if left.optimizer != right.optimizer:
        return left.optimizer, right.optimizer
    return f"{left.optimizer} 1", f"{right.optimizer} 2"


def metadata_rows(
    left: CheckpointSummary,
    right: CheckpointSummary,
) -> list[AnalysisTableRow]:
    return [
        AnalysisTableRow(label="Dataset", left=format_dataset(left.dataset), right=format_dataset(right.dataset)),
        AnalysisTableRow(label="Optimizer", left=left.optimizer, right=right.optimizer),
        AnalysisTableRow(label="Seed", left=str(left.seed), right=str(right.seed)),
        AnalysisTableRow(
            label="Batch size",
            left=str(left.config.batch_size),
            right=str(right.config.batch_size),
        ),
        AnalysisTableRow(label="Steps", left=str(left.step), right=str(right.step)),
        AnalysisTableRow(
            label="Saved at",
            left=left.saved_at,
            right=right.saved_at,
        ),
    ]


def comparability_rows(
    left_config: ExperimentConfig,
    right_config: ExperimentConfig,
) -> list[AnalysisTableRow]:
    same_dataset = left_config.dataset == right_config.dataset
    same_architecture = type(build_model(left_config.dataset)) is type(build_model(right_config.dataset))
    return [
        AnalysisTableRow(
            label="Dataset",
            left=format_dataset(left_config.dataset),
            right=format_dataset(right_config.dataset),
            comparable=same_dataset,
        ),
        AnalysisTableRow(
            label="Architecture",
            left=type(build_model(left_config.dataset)).__name__,
            right=type(build_model(right_config.dataset)).__name__,
            comparable=same_architecture,
        ),
    ]


def runtime_rows(
    left: CheckpointSummary,
    right: CheckpointSummary,
    left_status: ExperimentStatus,
    right_status: ExperimentStatus,
) -> list[AnalysisTableRow]:
    return [
        AnalysisTableRow(
            label="Elapsed time",
            left=format_optional_seconds(left.total_elapsed_seconds),
            right=format_optional_seconds(right.total_elapsed_seconds),
        ),
        AnalysisTableRow(
            label="Device",
            left=left.device_name or left.device,
            right=right.device_name or right.device,
        ),
        AnalysisTableRow(
            label="Peak memory",
            left=last_memory_history_value(left_status.history.memory_mb),
            right=last_memory_history_value(right_status.history.memory_mb),
        ),
    ]


def curves_from_status(status: ExperimentStatus) -> AnalysisCurves:
    return AnalysisCurves(
        training_loss=list(status.history.loss),
        validation_accuracy=curve_points(status.history.acc),
        training_accuracy=curve_points(status.history.train_acc),
        validation_loss=curve_points(status.history.val_loss),
    )


def curve_points(points: list[Any]) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    for point in points:
        if hasattr(point, "model_dump"):
            dumped = point.model_dump()
            output.append({"i": int(dumped["i"]), "value": float(dumped["value"])})
        elif isinstance(point, dict):
            output.append({"i": int(point["i"]), "value": float(point["value"])})
    return output


def validate_comparable_checkpoints(
    left_config: ExperimentConfig,
    right_config: ExperimentConfig,
) -> None:
    if left_config.dataset != right_config.dataset:
        raise AnalysisComparisonError("checkpoints must use the same dataset")
    if type(build_model(left_config.dataset)) is not type(build_model(right_config.dataset)):
        raise AnalysisComparisonError("checkpoints must use the same architecture")


def comparison_cache_key(request: AnalysisComparisonJobRequest) -> str:
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "left": request.left.model_dump(mode="json"),
        "right": request.right.model_dump(mode="json"),
        "params": request.params.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_fingerprint(saver: CheckpointSaver, selection: CheckpointSelection) -> str:
    path = saver.pt_path(selection.run_id, selection.kind)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cached_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("analysis_version") != ANALYSIS_VERSION:
            return None
        AnalysisComparisonReport.model_validate(payload.get("report"))
        return payload
    except Exception:
        return None


def write_cached_report(
    path: Path,
    *,
    report: AnalysisComparisonReport,
    fingerprints: dict[str, str],
    request: AnalysisComparisonJobRequest,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at": report.generated_at,
        "left": request.left.model_dump(mode="json"),
        "right": request.right.model_dump(mode="json"),
        "params": request.params.model_dump(mode="json"),
        "fingerprints": fingerprints,
        "report": report.model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stale_cache_sides(
    cached: dict[str, Any],
    fingerprints: dict[str, str],
) -> list[Literal["left", "right"]]:
    cached_fingerprints = cached.get("fingerprints")
    if not isinstance(cached_fingerprints, dict):
        return ["left", "right"]
    stale: list[Literal["left", "right"]] = []
    for side in ("left", "right"):
        if cached_fingerprints.get(side) != fingerprints[side]:
            stale.append(side)
    return stale


def checkpoint_summary_from_payload(
    saver: CheckpointSaver,
    payload: dict[str, Any],
    selection: CheckpointSelection,
) -> CheckpointSummary:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = saver.load_metadata(selection.run_id, selection.kind)
    return checkpoint_summary_from_metadata(metadata, selection.run_id, selection.kind)


def resolve_analysis_device(experiment_running: bool) -> torch.device:
    if torch.cuda.is_available() and not experiment_running:
        return torch.device("cuda")
    return torch.device("cpu")


def format_sse(event_type: str, payload: AnalysisComparisonJobStatus) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload.model_dump(mode='json'))}\n\n"


def class_names(dataset: DatasetName) -> tuple[str, ...]:
    if dataset == "fashion_mnist":
        return FASHION_MNIST_CLASS_LABELS
    if dataset == "cifar10":
        return CIFAR10_CLASS_LABELS
    return MNIST_CLASS_LABELS


def format_dataset(dataset: str) -> str:
    if dataset == "fashion_mnist":
        return "Fashion MNIST"
    if dataset == "cifar10":
        return "CIFAR-10"
    return dataset.upper()


def format_optional_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    return f"{seconds:.2f}s"


def format_memory_mb(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value >= 1024:
        return f"{value / 1024:.3g} GB"
    return f"{value:.0f} MB"


def last_memory_history_value(points: list[Any]) -> str:
    if not points:
        return "n/a"
    value = getattr(points[-1], "value", None)
    if value is None:
        return "n/a"
    return format_memory_mb(float(value))


def safe_float(value: float) -> float:
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    if not np.isfinite(value):
        return minimum
    return min(maximum, max(minimum, float(value)))
