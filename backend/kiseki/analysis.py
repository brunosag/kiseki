from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .checkpoint import CheckpointSaver, checkpoint_summary_from_metadata
from .data import DataLoaderFactory
from .models import CNN2C2DMNIST
from .schemas import (
    CheckpointSummary,
    ExperimentConfig,
    TSNEAnalysisRequest,
    TSNEAnalysisResponse,
    TSNEParams,
    TSNEPoint,
)


class TSNEParameterError(ValueError):
    pass


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

        model = CNN2C2DMNIST()
        model.load_state_dict(payload["model_state"])
        model.eval()

        features, labels, predictions = collect_final_hidden_activations(
            model,
            self.data_loader_factory.mnist_test(),
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


def collect_final_hidden_activations(
    model: CNN2C2DMNIST,
    loader: torch.utils.data.DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in loader:
            hidden = model.named_activations(inputs, ("fc1_relu",))["fc1_relu"]
            logits = model.fc2(hidden)
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
