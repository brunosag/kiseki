from __future__ import annotations

import secrets

import numpy as np
import torch
from captum.attr import LRP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .checkpoint import CheckpointSaver, checkpoint_summary_from_metadata
from .data import CIFAR10_MEAN, CIFAR10_STD, DataLoaderFactory, test_loader
from .models import build_model
from .schemas import (
    CheckpointSummary,
    ExperimentConfig,
    LRPAnalysisRequest,
    LRPAnalysisResponse,
    LRPSample,
    TSNEAnalysisRequest,
    TSNEAnalysisResponse,
    TSNEParams,
    TSNEPoint,
)


class TSNEParameterError(ValueError):
    pass


class LRPParameterError(ValueError):
    pass


LRP_SEED_UPPER_BOUND = 2_147_483_647


class AnalysisService:
    def __init__(
        self,
        *,
        data_loader_factory: DataLoaderFactory,
        checkpoint_saver: CheckpointSaver,
    ) -> None:
        self.data_loader_factory = data_loader_factory
        self.checkpoint_saver = checkpoint_saver

    def checkpoint_summaries(self) -> list[CheckpointSummary]:
        return self.checkpoint_saver.list_analysis_summaries()

    def tsne(self, request: TSNEAnalysisRequest) -> TSNEAnalysisResponse:
        payload = self.checkpoint_saver.load(
            request.checkpoint.run_id,
            request.checkpoint.kind,
            map_location="cpu",
        )
        config = ExperimentConfig.model_validate(payload["config"])
        seed = config.seed if request.params.seed is None else request.params.seed
        params = request.params.model_copy(update={"seed": seed})

        model = build_model(config.dataset)
        model.load_state_dict(payload["model_state"])
        model.eval()

        features, labels, predictions = collect_final_hidden_activations(
            model,
            test_loader(self.data_loader_factory, config.dataset),
        )
        validate_tsne_params(params, sample_count=features.shape[0])
        coordinates = tsne_embedding(features, params)

        return TSNEAnalysisResponse(
            checkpoint=checkpoint_summary_from_payload(
                self.checkpoint_saver,
                payload,
                run_id=request.checkpoint.run_id,
                kind=request.checkpoint.kind,
            ),
            params=params,
            points=[
                TSNEPoint(
                    x=float(point[0]),
                    y=float(point[1]),
                    label=int(label),
                    prediction=int(prediction),
                    correct=bool(label == prediction),
                )
                for point, label, prediction in zip(
                    coordinates,
                    labels,
                    predictions,
                    strict=True,
                )
            ],
        )

    def lrp(self, request: LRPAnalysisRequest) -> LRPAnalysisResponse:
        payload = self.checkpoint_saver.load(
            request.checkpoint.run_id,
            request.checkpoint.kind,
            map_location="cpu",
        )
        config = ExperimentConfig.model_validate(payload["config"])
        seed = (
            request.params.seed
            if request.params.seed is not None
            else secrets.randbelow(LRP_SEED_UPPER_BOUND + 1)
        )
        params = request.params.model_copy(update={"seed": seed})

        model = build_model(config.dataset)
        model.load_state_dict(payload["model_state"])
        model.eval()

        inputs, labels = collect_lrp_inputs(
            test_loader(self.data_loader_factory, config.dataset)
        )
        selected_indices = balanced_sample_indices(
            labels.tolist(),
            min(params.sample_count, labels.numel()),
            seed=seed,
        )
        if not selected_indices:
            raise LRPParameterError("test loader did not return any samples")

        selected_inputs = inputs[selected_indices].clone().requires_grad_(True)
        selected_labels = labels[selected_indices]

        with torch.no_grad():
            logits = model(selected_inputs)
            predictions = logits.argmax(dim=1)
            scores = logits.gather(1, predictions.unsqueeze(1)).squeeze(1)

        model.zero_grad(set_to_none=True)
        attributions, deltas = LRP(model).attribute(
            selected_inputs,
            target=predictions,
            return_convergence_delta=True,
        )
        if not isinstance(attributions, torch.Tensor):
            raise LRPParameterError("LRP returned multiple attribution tensors")

        return LRPAnalysisResponse(
            checkpoint=checkpoint_summary_from_payload(
                self.checkpoint_saver,
                payload,
                run_id=request.checkpoint.run_id,
                kind=request.checkpoint.kind,
            ),
            params=params,
            samples=[
                LRPSample(
                    index=int(source_index),
                    label=int(label),
                    prediction=int(prediction),
                    target=int(prediction),
                    correct=bool(label == prediction),
                    score=float(score),
                    delta=float(delta),
                    image=rgb_image(sample_input, config.dataset),
                    relevance=normalized_signed_relevance(attribution),
                )
                for source_index, label, prediction, score, delta, sample_input, attribution in zip(
                    selected_indices,
                    selected_labels.tolist(),
                    predictions.tolist(),
                    scores.tolist(),
                    deltas.detach().cpu().tolist(),
                    selected_inputs,
                    attributions.detach().cpu(),
                    strict=True,
                )
            ],
        )


def collect_final_hidden_activations(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in loader:
            final_hidden = getattr(model, "final_hidden", None)
            if not callable(final_hidden):
                raise TSNEParameterError("model does not expose final_hidden activations")
            hidden = final_hidden(inputs)
            logits = model(inputs)
            feature_batches.append(hidden.detach().cpu())
            label_batches.append(targets.detach().cpu())
            prediction_batches.append(logits.argmax(dim=1).detach().cpu())

    if not feature_batches:
        raise TSNEParameterError("test loader did not return any samples")

    return (
        torch.cat(feature_batches).numpy(),
        torch.cat(label_batches).numpy(),
        torch.cat(prediction_batches).numpy(),
    )


def collect_lrp_inputs(
    loader: torch.utils.data.DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []

    for inputs, targets in loader:
        input_batches.append(inputs.detach().cpu())
        label_batches.append(torch.as_tensor(targets, dtype=torch.long).detach().cpu())

    if not input_batches:
        raise LRPParameterError("test loader did not return any samples")

    return torch.cat(input_batches), torch.cat(label_batches)


def balanced_sample_indices(
    labels: list[int],
    sample_count: int,
    *,
    seed: int,
    class_count: int = 10,
) -> list[int]:
    target_count = min(sample_count, len(labels))
    rng = np.random.default_rng(seed)
    desired_counts = [target_count // class_count] * class_count
    for label in range(target_count % class_count):
        desired_counts[label] += 1

    buckets: list[list[int]] = [[] for _ in range(class_count)]
    for index, label in enumerate(labels):
        if 0 <= label < class_count:
            buckets[label].append(index)

    selected: list[int] = []
    selected_set: set[int] = set()
    for label, desired_count in enumerate(desired_counts):
        bucket = rng.permutation(buckets[label]).tolist()
        for index in bucket[:desired_count]:
            selected.append(index)
            selected_set.add(index)

    for index in rng.permutation(len(labels)).tolist():
        if len(selected) >= target_count:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)

    return selected


def rgb_image(input_tensor: torch.Tensor, dataset: str) -> list[list[list[float]]]:
    image = input_tensor.detach().cpu().float()
    if dataset == "cifar10":
        mean = torch.tensor(CIFAR10_MEAN, dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor(CIFAR10_STD, dtype=image.dtype).view(3, 1, 1)
        image = image * std + mean

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    image = torch.nan_to_num(image).clamp(0.0, 1.0)
    return image.permute(1, 2, 0).tolist()


def normalized_signed_relevance(attribution: torch.Tensor) -> list[list[float]]:
    relevance = attribution.detach().cpu().float().sum(dim=0)
    relevance = torch.nan_to_num(relevance)
    max_abs = float(relevance.abs().max())
    if max_abs > 0.0:
        relevance = relevance / max_abs
    return relevance.clamp(-1.0, 1.0).tolist()


def validate_tsne_params(params: TSNEParams, *, sample_count: int) -> None:
    if sample_count < 2:
        raise TSNEParameterError("sample_count must be at least 2")
    if params.perplexity >= sample_count:
        raise TSNEParameterError("perplexity must be less than sample_count")


def tsne_embedding(features: np.ndarray, params: TSNEParams) -> np.ndarray:
    feature_matrix = np.asarray(features, dtype=np.float32)
    if params.use_pca:
        component_count = min(
            params.pca_components,
            feature_matrix.shape[0],
            feature_matrix.shape[1],
        )
        feature_matrix = PCA(
            n_components=component_count,
            random_state=params.seed,
        ).fit_transform(feature_matrix)

    learning_rate: float | str
    if params.learning_rate_mode == "auto":
        learning_rate = "auto"
    else:
        learning_rate = float(params.learning_rate)

    return TSNE(
        n_components=2,
        perplexity=params.perplexity,
        max_iter=params.max_iter,
        learning_rate=learning_rate,
        angle=params.angle,
        method="barnes_hut",
        init="random",
        random_state=params.seed,
    ).fit_transform(feature_matrix)


def checkpoint_summary_from_payload(
    saver: CheckpointSaver,
    payload: dict,
    *,
    run_id: str,
    kind: str,
) -> CheckpointSummary:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = saver.load_metadata(run_id, kind)  # type: ignore[arg-type]
    return checkpoint_summary_from_metadata(metadata, run_id, kind)  # type: ignore[arg-type]
