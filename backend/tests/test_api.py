import os
import time
from datetime import UTC, datetime

import numpy as np
import torch
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader, TensorDataset

from kiseki import analysis
from kiseki.api import create_app
from kiseki.checkpoint import CheckpointSaver
from kiseki import experiment
from kiseki.experiment import (
    ExperimentManager,
    build_run_id,
    format_sse,
    resolve_device,
    seed_everything,
)
from kiseki.dataset_types import DatasetName
from kiseki.models import CIFARResNet20, build_model
from kiseki.schemas import ETA, ETA_0, ExperimentConfig, ExperimentStatus
from kiseki.optimizers import SGDConfig, SGDRunner


class SyntheticLoaderFactory:
    def mnist(self, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
        generator = torch.Generator().manual_seed(seed)
        inputs = torch.randn(24, 1, 28, 28, generator=generator)
        targets = torch.randint(0, 10, (24,), generator=generator)
        dataset = TensorDataset(inputs, targets)
        return (
            DataLoader(dataset, batch_size=batch_size, shuffle=True),
            DataLoader(dataset, batch_size=batch_size, shuffle=False),
        )

    def mnist_test(self) -> DataLoader:
        generator = torch.Generator().manual_seed(123)
        inputs = torch.randn(12, 1, 28, 28, generator=generator)
        targets = torch.arange(12) % 10
        return DataLoader(TensorDataset(inputs, targets), batch_size=5, shuffle=False)

    def cifar10(self, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
        generator = torch.Generator().manual_seed(seed)
        inputs = torch.randn(24, 3, 32, 32, generator=generator)
        targets = torch.randint(0, 10, (24,), generator=generator)
        dataset = TensorDataset(inputs, targets)
        return (
            DataLoader(dataset, batch_size=batch_size, shuffle=True),
            DataLoader(dataset, batch_size=batch_size, shuffle=False),
        )

    def cifar10_test(self) -> DataLoader:
        generator = torch.Generator().manual_seed(123)
        inputs = torch.randn(12, 3, 32, 32, generator=generator)
        targets = torch.arange(12) % 10
        return DataLoader(TensorDataset(inputs, targets), batch_size=5, shuffle=False)


class SlowSGDRunner(SGDRunner):
    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        time.sleep(0.02)
        return super().step(inputs, targets)


def wait_for_status(client: TestClient, predicate, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    status = client.get("/api/experiments/status").json()
    while not predicate(status) and time.monotonic() < deadline:
        time.sleep(0.05)
        status = client.get("/api/experiments/status").json()
    return status


def save_api_checkpoint(
    saver: CheckpointSaver,
    *,
    run_id: str,
    saved_at: str,
    step: int,
    kind: str = "latest",
    iterations: int = 6,
    accuracy: float | None = 12.0,
    dataset: DatasetName = "mnist",
) -> None:
    config = ExperimentConfig(
        dataset=dataset,
        device="cpu",
        seed=19,
        batch_size=4,
        iterations=iterations,
        target_acc=100.0,
        optimizer="SGD",
    )
    status = ExperimentStatus(
        run_id=run_id,
        optimizer="SGD",
        current_step=step,
        current_loss=0.75,
        total_elapsed_seconds=123.4,
        best_acc=accuracy or 0.0,
        last_checkpoint_step=step,
        last_checkpoint_acc=accuracy,
        last_checkpoint_saved_at=saved_at,
        checkpoint_path=str(saver.latest_pt_path(run_id)),
        best_checkpoint_acc=accuracy if kind == "best" else None,
        best_checkpoint_step=step if kind == "best" else None,
        best_checkpoint_saved_at=saved_at if kind == "best" else None,
        best_checkpoint_path=str(saver.best_pt_path(run_id)) if kind == "best" else None,
    )
    saver.save(
        model=build_model(dataset),
        status=status,
        config=config,
        optimizer="SGD",
        run_id=run_id,
        optimizer_params={"SGD": {ETA: 0.02}},
        saved_at=saved_at,
        kind=kind,
    )


def test_api_lists_only_complete_latest_checkpoints_sorted_newest_first(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="old-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
    )
    save_api_checkpoint(
        saver,
        run_id="new-run",
        saved_at="2026-06-20T12:02:00+00:00",
        step=4,
    )
    save_api_checkpoint(
        saver,
        run_id="hidden-best-run",
        saved_at="2026-06-20T12:01:00+00:00",
        step=3,
        kind="best",
    )
    incomplete_dir = tmp_path / "incomplete-run"
    incomplete_dir.mkdir()
    (incomplete_dir / "latest.json").write_text("{}", encoding="utf-8")

    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))

    response = client.get("/api/checkpoints")

    assert response.status_code == 200
    checkpoints = response.json()
    assert [(item["run_id"], item["kind"]) for item in checkpoints] == [
        ("new-run", "latest"),
        ("old-run", "latest"),
    ]
    assert checkpoints[0]["saved_at"] == "2026-06-20T12:02:00+00:00"
    assert checkpoints[0]["step"] == 4
    assert checkpoints[0]["optimizer"] == "SGD"
    assert checkpoints[0]["dataset"] == "mnist"
    assert checkpoints[0]["seed"] == 19
    assert checkpoints[0]["device"] == "cpu"
    assert checkpoints[0]["deterministic"] is False
    assert checkpoints[0]["accuracy"] == 12.0
    assert checkpoints[0]["current_loss"] == 0.75
    assert checkpoints[0]["total_elapsed_seconds"] == 123.4
    assert checkpoints[0]["reproducibility_status"].startswith("best-effort")
    assert checkpoints[0]["config"]["iterations"] == 6
    assert checkpoints[0]["optimizer_params"] == {"SGD": {ETA: 0.02}}


def test_api_analysis_checkpoint_list_prefers_best_then_latest(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="latest-only-run",
        saved_at="2026-06-20T12:02:00+00:00",
        step=4,
    )
    save_api_checkpoint(
        saver,
        run_id="best-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
    )
    save_api_checkpoint(
        saver,
        run_id="best-run",
        saved_at="2026-06-20T12:03:00+00:00",
        step=5,
        kind="best",
    )
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))

    response = client.get("/api/checkpoints", params={"mode": "analysis"})

    assert response.status_code == 200
    assert [(item["run_id"], item["kind"]) for item in response.json()] == [
        ("best-run", "best"),
        ("latest-only-run", "latest"),
    ]


def test_api_schema_includes_supported_datasets(tmp_path) -> None:
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    response = client.get("/api/schema")

    assert response.status_code == 200
    assert response.json()["config_schema"]["dataset"]["options"] == [
        {"label": "MNIST", "value": "mnist"},
        {"label": "CIFAR-10", "value": "cifar10"},
    ]


def test_api_tsne_validates_params_loads_checkpoint_and_returns_points(
    tmp_path,
    monkeypatch,
) -> None:
    class FakePCA:
        def __init__(self, n_components: int, random_state: int) -> None:
            self.n_components = n_components
            self.random_state = random_state

        def fit_transform(self, features):
            return features[:, : self.n_components]

    class FakeTSNE:
        def __init__(self, **kwargs) -> None:
            assert kwargs["method"] == "barnes_hut"
            assert kwargs["learning_rate"] == 25.0
            assert kwargs["random_state"] == 7
            self.kwargs = kwargs

        def fit_transform(self, features):
            return np.column_stack(
                (
                    np.arange(features.shape[0], dtype=np.float32),
                    -np.arange(features.shape[0], dtype=np.float32),
                )
            )

    monkeypatch.setattr(analysis, "PCA", FakePCA)
    monkeypatch.setattr(analysis, "TSNE", FakeTSNE)
    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="analysis-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
    )
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))
    before_status = client.get("/api/experiments/status").json()

    response = client.post(
        "/api/analysis/tsne",
        json={
            "checkpoint": {"run_id": "analysis-run", "kind": "latest"},
            "params": {
                "perplexity": 5,
                "max_iter": 250,
                "learning_rate_mode": "numeric",
                "learning_rate": 25,
                "angle": 0.3,
                "pca_components": 2,
                "seed": 7,
                "use_pca": True,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["checkpoint"]["run_id"] == "analysis-run"
    assert payload["params"]["seed"] == 7
    assert len(payload["points"]) == 12
    assert set(payload["points"][0]) == {"x", "y", "label", "prediction", "correct"}
    assert payload["points"][0]["x"] == 0.0
    assert payload["points"][1]["y"] == -1.0
    assert client.get("/api/experiments/status").json() == before_status

    invalid = client.post(
        "/api/analysis/tsne",
        json={
            "checkpoint": {"run_id": "analysis-run", "kind": "latest"},
            "params": {
                "perplexity": 5,
                "learning_rate_mode": "numeric",
                "learning_rate": 0,
            },
        },
    )
    assert invalid.status_code == 422


def test_api_tsne_uses_cifar_checkpoint_model_and_test_loader(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeTSNE:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def fit_transform(self, features):
            assert features.shape == (12, 64)
            return np.column_stack(
                (
                    np.arange(features.shape[0], dtype=np.float32),
                    np.arange(features.shape[0], dtype=np.float32),
                )
            )

    class TrackingLoaderFactory(SyntheticLoaderFactory):
        def __init__(self) -> None:
            self.cifar10_test_calls = 0

        def cifar10_test(self) -> DataLoader:
            self.cifar10_test_calls += 1
            return super().cifar10_test()

    monkeypatch.setattr(analysis, "TSNE", FakeTSNE)
    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="cifar-analysis-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
        dataset="cifar10",
    )
    data_loader_factory = TrackingLoaderFactory()
    manager = ExperimentManager(
        data_loader_factory=data_loader_factory,
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))

    response = client.post(
        "/api/analysis/tsne",
        json={
            "checkpoint": {"run_id": "cifar-analysis-run", "kind": "latest"},
            "params": {
                "perplexity": 5,
                "max_iter": 250,
                "angle": 0.3,
                "seed": 7,
                "use_pca": False,
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["checkpoint"]["dataset"] == "cifar10"
    assert len(response.json()["points"]) == 12
    assert data_loader_factory.cifar10_test_calls == 1


def test_api_lrp_returns_balanced_predicted_class_samples_for_mnist(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="lrp-analysis-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
    )
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))
    before_status = client.get("/api/experiments/status").json()

    response = client.post(
        "/api/analysis/lrp",
        json={
            "checkpoint": {"run_id": "lrp-analysis-run", "kind": "latest"},
            "params": {"sample_count": 10, "seed": 8},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["checkpoint"]["run_id"] == "lrp-analysis-run"
    assert payload["params"] == {"sample_count": 10, "seed": 8}
    assert len(payload["samples"]) == 10
    assert [sample["label"] for sample in payload["samples"]] == list(range(10))
    assert [sample["index"] for sample in payload["samples"]] == [
        10,
        11,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    assert all(sample["target"] == sample["prediction"] for sample in payload["samples"])

    sample = payload["samples"][0]
    assert set(sample) == {
        "index",
        "label",
        "prediction",
        "target",
        "correct",
        "score",
        "delta",
        "image",
        "relevance",
    }
    assert sample["correct"] == (sample["label"] == sample["prediction"])
    assert np.isfinite(sample["score"])
    assert np.isfinite(sample["delta"])
    assert len(sample["image"]) == 28
    assert len(sample["image"][0]) == 28
    assert len(sample["image"][0][0]) == 3
    assert len(sample["relevance"]) == 28
    assert len(sample["relevance"][0]) == 28
    image_values = np.asarray(sample["image"])
    relevance_values = np.asarray(sample["relevance"])
    assert image_values.min() >= 0.0
    assert image_values.max() <= 1.0
    assert relevance_values.min() >= -1.0
    assert relevance_values.max() <= 1.0
    assert client.get("/api/experiments/status").json() == before_status

    invalid = client.post(
        "/api/analysis/lrp",
        json={
            "checkpoint": {"run_id": "lrp-analysis-run", "kind": "latest"},
            "params": {"sample_count": 0},
        },
    )
    assert invalid.status_code == 422


def test_api_lrp_uses_cifar_checkpoint_model_and_returns_denormalized_rgb(
    tmp_path,
) -> None:
    class TrackingLoaderFactory(SyntheticLoaderFactory):
        def __init__(self) -> None:
            self.cifar10_test_calls = 0
            generator = torch.Generator().manual_seed(321)
            self.raw_inputs = torch.rand(12, 3, 32, 32, generator=generator)

        def cifar10_test(self) -> DataLoader:
            self.cifar10_test_calls += 1
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
            normalized_inputs = (self.raw_inputs - mean) / std
            targets = torch.arange(12) % 10
            return DataLoader(
                TensorDataset(normalized_inputs, targets),
                batch_size=5,
                shuffle=False,
            )

    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="cifar-lrp-analysis-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
        dataset="cifar10",
    )
    data_loader_factory = TrackingLoaderFactory()
    manager = ExperimentManager(
        data_loader_factory=data_loader_factory,
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))

    response = client.post(
        "/api/analysis/lrp",
        json={
            "checkpoint": {"run_id": "cifar-lrp-analysis-run", "kind": "latest"},
            "params": {"sample_count": 10, "seed": 8},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["checkpoint"]["dataset"] == "cifar10"
    assert len(payload["samples"]) == 10
    assert [sample["label"] for sample in payload["samples"]] == list(range(10))
    assert all(sample["target"] == sample["prediction"] for sample in payload["samples"])
    assert data_loader_factory.cifar10_test_calls == 1

    sample = payload["samples"][0]
    assert len(sample["image"]) == 32
    assert len(sample["image"][0]) == 32
    assert len(sample["image"][0][0]) == 3
    assert len(sample["relevance"]) == 32
    assert len(sample["relevance"][0]) == 32
    source_index = sample["index"]
    np.testing.assert_allclose(
        sample["image"][0][0],
        data_loader_factory.raw_inputs[source_index, :, 0, 0].tolist(),
        atol=1e-6,
    )
    relevance_values = np.asarray(sample["relevance"])
    assert relevance_values.min() >= -1.0
    assert relevance_values.max() <= 1.0


def test_lrp_balanced_sample_indices_are_seeded_and_class_balanced() -> None:
    labels = [label for label in range(10) for _ in range(5)]

    first = analysis.balanced_sample_indices(labels, 20, seed=1)
    second = analysis.balanced_sample_indices(labels, 20, seed=1)
    resampled = analysis.balanced_sample_indices(labels, 20, seed=2)

    assert first == second
    assert first != resampled
    assert [labels[index] for index in first] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        8,
        8,
        9,
        9,
    ]


def test_api_delete_checkpoint_run_removes_latest_and_hidden_best(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="delete-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
    )
    save_api_checkpoint(
        saver,
        run_id="delete-run",
        saved_at="2026-06-20T12:01:00+00:00",
        step=3,
        kind="best",
    )
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))

    response = client.delete("/api/checkpoints/delete-run")

    assert response.status_code == 204
    assert not (tmp_path / "delete-run").exists()
    assert client.get("/api/checkpoints").json() == []


def test_api_delete_checkpoint_run_rejects_missing_running_and_paused_current(
    tmp_path,
    monkeypatch,
) -> None:
    def build_slow_runner(optimizer_name, model, raw_params, *, device, seed, **kwargs):
        assert optimizer_name == "SGD"
        return SlowSGDRunner(model, SGDConfig(eta=float(raw_params.get(ETA, 0.01))))

    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="delete-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
    )
    monkeypatch.setattr(experiment, "build_optimizer_runner", build_slow_runner)
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))

    assert client.delete("/api/checkpoints/missing-run").status_code == 404

    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 11,
            "batch_size": 4,
            "iterations": 100,
            "target_acc": 100.0,
            "optimizer": "SGD",
            "checkpoint_interval": 0,
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }
    assert client.post("/api/experiments/start", json=payload).status_code == 200
    running = wait_for_status(client, lambda status: status["current_step"] >= 1)

    assert client.delete("/api/checkpoints/delete-run").status_code == 409

    assert client.post("/api/experiments/pause").status_code == 200
    paused = wait_for_status(client, lambda status: status["is_paused"])
    assert paused["run_id"] == running["run_id"]
    assert client.delete(f"/api/checkpoints/{paused['run_id']}").status_code == 409
    assert client.post("/api/experiments/stop").status_code == 200


def test_api_start_from_checkpoint_uses_saved_payload(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="resume-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
        kind="best",
        iterations=4,
    )
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))

    response = client.post(
        "/api/experiments/start",
        json={
            "config": {
                "dataset": "mnist",
                "device": "cpu",
                "seed": 999,
                "batch_size": 4,
                "iterations": 1,
                "target_acc": 100.0,
                "optimizer": "LEEA",
            },
            "opt_params": {"LEEA": {"N": 4, ETA_0: 0.2}},
            "checkpoint": {"run_id": "resume-run", "kind": "best"},
        },
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == "resume-run"
    assert response.json()["optimizer"] == "SGD"
    status = wait_for_status(client, lambda status: not status["is_running"])
    assert status["run_id"] == "resume-run"
    assert status["current_step"] == 4
    assert status["requested_device"] == "cpu"


def test_api_load_checkpoint_sets_paused_status_and_resume_uses_selected_kind(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="resume-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
        kind="best",
        iterations=4,
    )
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))

    load_response = client.post(
        "/api/checkpoints/load",
        json={"run_id": "resume-run", "kind": "best"},
    )

    assert load_response.status_code == 200
    loaded = load_response.json()
    assert loaded["is_running"] is False
    assert loaded["is_paused"] is True
    assert loaded["run_id"] == "resume-run"
    assert loaded["optimizer"] == "SGD"
    assert loaded["current_step"] == 2
    assert loaded["total_elapsed_seconds"] == 123.4

    resume_response = client.post("/api/experiments/resume")
    assert resume_response.status_code == 200
    status = wait_for_status(client, lambda status: not status["is_running"])
    assert status["run_id"] == "resume-run"
    assert status["current_step"] == 4


def test_api_reset_clears_loaded_checkpoint_status(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="loaded-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
    )
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))

    load_response = client.post(
        "/api/checkpoints/load",
        json={"run_id": "loaded-run", "kind": "latest"},
    )
    assert load_response.status_code == 200
    assert load_response.json()["is_paused"] is True

    reset_response = client.post("/api/experiments/reset")

    assert reset_response.status_code == 200
    reset = reset_response.json()
    assert reset["is_running"] is False
    assert reset["is_paused"] is False
    assert reset["run_id"] is None
    assert reset["current_step"] == 0
    assert reset["current_loss"] == 0.0
    assert reset["best_acc"] == 0.0
    assert reset["total_elapsed_seconds"] == 0.0
    assert reset["history"] == {"loss": [], "acc": [], "mutation_step": []}
    assert reset["checkpoint_path"] is None
    assert client.get("/api/experiments/status").json() == reset


def test_api_start_from_missing_checkpoint_returns_404(tmp_path) -> None:
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    response = client.post(
        "/api/experiments/start",
        json={"checkpoint": {"run_id": "missing-run", "kind": "latest"}},
    )

    assert response.status_code == 404
    assert "missing-run" in response.json()["detail"]

    response = client.post(
        "/api/checkpoints/load",
        json={"run_id": "missing-run", "kind": "latest"},
    )

    assert response.status_code == 404
    assert "missing-run" in response.json()["detail"]


def test_api_checkpoint_start_conflicts_and_load_replaces_paused_state(
    tmp_path,
    monkeypatch,
) -> None:
    def build_slow_runner(optimizer_name, model, raw_params, *, device, seed, **kwargs):
        assert optimizer_name == "SGD"
        return SlowSGDRunner(model, SGDConfig(eta=float(raw_params.get(ETA, 0.01))))

    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="resume-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=2,
    )
    monkeypatch.setattr(experiment, "build_optimizer_runner", build_slow_runner)
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))
    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 11,
            "batch_size": 4,
            "iterations": 100,
            "target_acc": 100.0,
            "optimizer": "SGD",
            "checkpoint_interval": 0,
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }
    checkpoint_payload = {"checkpoint": {"run_id": "resume-run", "kind": "latest"}}

    assert client.post("/api/experiments/start", json=payload).status_code == 200
    wait_for_status(client, lambda status: status["current_step"] >= 1)
    assert client.post("/api/experiments/start", json=checkpoint_payload).status_code == 409
    assert client.post("/api/experiments/reset").status_code == 409
    assert (
        client.post("/api/checkpoints/load", json=checkpoint_payload["checkpoint"]).status_code
        == 409
    )

    assert client.post("/api/experiments/pause").status_code == 200
    wait_for_status(client, lambda status: status["is_paused"])
    assert client.post("/api/experiments/start", json=checkpoint_payload).status_code == 409
    load_response = client.post("/api/checkpoints/load", json=checkpoint_payload["checkpoint"])
    assert load_response.status_code == 200
    assert load_response.json()["is_paused"] is True
    assert load_response.json()["run_id"] == "resume-run"
    assert client.post("/api/experiments/stop").status_code == 200


def test_api_start_status_stop_flow(tmp_path) -> None:
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 3,
            "batch_size": 4,
            "iterations": 100,
            "target_acc": 100.0,
            "optimizer": "SGD",
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }
    response = client.post("/api/experiments/start", json=payload)

    assert response.status_code == 200
    assert response.json()["is_running"] is True

    deadline = time.monotonic() + 5
    status = client.get("/api/experiments/status").json()
    while status["current_step"] == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
        status = client.get("/api/experiments/status").json()

    assert status["current_step"] > 0
    assert status["requested_device"] == "cpu"
    assert status["device"] == "cpu"
    assert status["device_name"] == "cpu"
    assert status["total_elapsed_seconds"] > 0
    assert status["last_iteration_seconds"] > 0
    assert status["current_mutation_step"] is None
    assert status["history"]["mutation_step"] == []

    stop_response = client.post("/api/experiments/stop")
    assert stop_response.status_code == 200
    wait_for_status(client, lambda status: not status["is_running"])


def test_api_start_cifar10_experiment_uses_cifar_loader_and_model(
    tmp_path,
    monkeypatch,
) -> None:
    class TrackingLoaderFactory(SyntheticLoaderFactory):
        def __init__(self) -> None:
            self.cifar10_calls: list[tuple[int, int]] = []

        def cifar10(self, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
            self.cifar10_calls.append((batch_size, seed))
            return super().cifar10(batch_size, seed)

    captured: dict[str, torch.nn.Module] = {}

    def build_tracking_runner(optimizer_name, model, raw_params, *, device, seed, **kwargs):
        assert optimizer_name == "SGD"
        captured["model"] = model
        return SGDRunner(model, SGDConfig(eta=float(raw_params.get(ETA, 0.01))))

    monkeypatch.setattr(experiment, "build_optimizer_runner", build_tracking_runner)
    data_loader_factory = TrackingLoaderFactory()
    manager = ExperimentManager(
        data_loader_factory=data_loader_factory,
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    response = client.post(
        "/api/experiments/start",
        json={
            "config": {
                "dataset": "cifar10",
                "device": "cpu",
                "seed": 5,
                "batch_size": 4,
                "iterations": 1,
                "target_acc": 100.0,
                "optimizer": "SGD",
            },
            "opt_params": {"SGD": {ETA: 0.01}},
        },
    )

    assert response.status_code == 200
    status = wait_for_status(client, lambda status: not status["is_running"])
    assert status["run_id"].startswith("cifar10-sgd-cpu-seed5-")
    assert status["current_step"] == 1
    assert data_loader_factory.cifar10_calls == [(4, 5)]
    assert isinstance(captured["model"], CIFARResNet20)


def test_api_lists_and_resumes_cifar10_checkpoint(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    save_api_checkpoint(
        saver,
        run_id="cifar-resume-run",
        saved_at="2026-06-20T12:00:00+00:00",
        step=1,
        iterations=2,
        dataset="cifar10",
    )
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    client = TestClient(create_app(manager))

    checkpoints = client.get("/api/checkpoints").json()
    assert checkpoints[0]["dataset"] == "cifar10"
    assert checkpoints[0]["config"]["dataset"] == "cifar10"

    response = client.post(
        "/api/experiments/start",
        json={"checkpoint": {"run_id": "cifar-resume-run", "kind": "latest"}},
    )

    assert response.status_code == 200
    status = wait_for_status(client, lambda status: not status["is_running"])
    assert status["run_id"] == "cifar-resume-run"
    assert status["current_step"] == 2


def test_api_applies_paused_nontrajectory_controls_on_resume(tmp_path, monkeypatch) -> None:
    def build_slow_runner(optimizer_name, model, raw_params, *, device, seed, **kwargs):
        assert optimizer_name == "SGD"
        return SlowSGDRunner(model, SGDConfig(eta=float(raw_params.get(ETA, 0.01))))

    def evaluate_fixed(model, loader, device) -> float:
        return 10.0

    monkeypatch.setattr(experiment, "build_optimizer_runner", build_slow_runner)
    monkeypatch.setattr(experiment, "evaluate", evaluate_fixed)
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 3,
            "batch_size": 4,
            "iterations": 100,
            "target_acc": 100.0,
            "optimizer": "SGD",
            "checkpoint_interval": 0,
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }

    response = client.post("/api/experiments/start", json=payload)
    assert response.status_code == 200
    wait_for_status(client, lambda status: status["current_step"] >= 1)

    pause_response = client.post("/api/experiments/pause")
    assert pause_response.status_code == 200
    paused = wait_for_status(client, lambda status: status["is_paused"])
    updated_iterations = paused["current_step"] + 5

    resume_response = client.post(
        "/api/experiments/resume",
        json={
            "iterations": updated_iterations,
            "target_acc": 95.0,
            "checkpoint_interval": 1,
        },
    )

    assert resume_response.status_code == 200
    status = wait_for_status(client, lambda status: not status["is_running"])
    run_id = status["run_id"]
    assert status["current_step"] == updated_iterations
    assert status["last_checkpoint_step"] is not None

    checkpoint = manager.checkpoint_saver.load_latest(run_id)
    assert checkpoint["config"]["iterations"] == updated_iterations
    assert checkpoint["config"]["target_acc"] == 95.0
    assert checkpoint["config"]["checkpoint_interval"] == 1


def test_api_interval_checkpoint_and_checkpoint_interval_zero(tmp_path) -> None:
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 3,
            "batch_size": 4,
            "iterations": 3,
            "target_acc": 100.0,
            "optimizer": "SGD",
            "checkpoint_interval": 2,
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }

    response = client.post("/api/experiments/start", json=payload)
    assert response.status_code == 200
    status = wait_for_status(client, lambda status: not status["is_running"])

    assert status["last_checkpoint_step"] == 2
    assert (tmp_path / status["run_id"] / "latest.pt").exists()
    assert (tmp_path / status["run_id"] / "latest.json").exists()

    payload["config"]["checkpoint_interval"] = 0
    response = client.post("/api/experiments/start", json=payload)
    assert response.status_code == 200
    status = wait_for_status(client, lambda status: not status["is_running"])

    assert status["last_checkpoint_step"] is None
    assert not (tmp_path / status["run_id"] / "latest.pt").exists()


def test_api_saves_checkpoint_when_best_accuracy_is_surpassed_without_interval(
    tmp_path,
    monkeypatch,
) -> None:
    def evaluate_fixed(model, loader, device) -> float:
        return 18.5

    monkeypatch.setattr(experiment, "evaluate", evaluate_fixed)
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 3,
            "batch_size": 4,
            "iterations": 10,
            "target_acc": 100.0,
            "optimizer": "SGD",
            "checkpoint_interval": 0,
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }

    response = client.post("/api/experiments/start", json=payload)
    assert response.status_code == 200
    status = wait_for_status(client, lambda status: not status["is_running"])
    run_id = status["run_id"]

    assert status["best_acc"] == 18.5
    assert status["last_checkpoint_step"] == 10
    assert status["last_checkpoint_acc"] == 18.5
    assert status["best_checkpoint_step"] == 10
    assert status["best_checkpoint_acc"] == 18.5
    assert status["best_checkpoint_path"] == str(tmp_path / run_id / "latest.pt")
    assert (tmp_path / run_id / "latest.pt").exists()
    assert not (tmp_path / run_id / "best.pt").exists()


def test_api_regular_checkpoint_preserves_prior_best_and_reports_best_accuracy(
    tmp_path,
    monkeypatch,
) -> None:
    validation_accuracies = iter([20.0, 15.0])

    def evaluate_sequence(model, loader, device) -> float:
        return next(validation_accuracies)

    monkeypatch.setattr(experiment, "evaluate", evaluate_sequence)
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 3,
            "batch_size": 4,
            "iterations": 12,
            "target_acc": 100.0,
            "optimizer": "SGD",
            "checkpoint_interval": 12,
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }

    response = client.post("/api/experiments/start", json=payload)
    assert response.status_code == 200
    status = wait_for_status(client, lambda status: not status["is_running"])
    run_id = status["run_id"]

    assert status["best_acc"] == 20.0
    assert status["last_checkpoint_step"] == 12
    assert status["last_checkpoint_acc"] == 15.0
    assert status["best_checkpoint_step"] == 10
    assert status["best_checkpoint_acc"] == 20.0
    assert status["best_checkpoint_path"] == str(tmp_path / run_id / "best.pt")

    latest_metadata = manager.checkpoint_saver.load_latest_metadata(run_id)
    assert latest_metadata["step"] == 12
    assert latest_metadata["best_acc"] == 20.0
    assert latest_metadata["last_checkpoint_acc"] == 15.0

    best_payload = manager.checkpoint_saver.load_best(run_id)
    assert best_payload["status"]["current_step"] == 10

    checkpoints = client.get("/api/checkpoints").json()
    assert len(checkpoints) == 1
    assert checkpoints[0]["run_id"] == run_id
    assert checkpoints[0]["accuracy"] == status["best_acc"]


def test_api_best_checkpoint_updates_only_on_strict_checkpoint_accuracy_improvement(
    tmp_path,
    monkeypatch,
) -> None:
    validation_accuracies = iter([10.0, 8.0, 10.0, 12.0])

    def evaluate_sequence(model, loader, device) -> float:
        return next(validation_accuracies)

    monkeypatch.setattr(experiment, "evaluate", evaluate_sequence)
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 3,
            "batch_size": 4,
            "iterations": 8,
            "target_acc": 100.0,
            "optimizer": "SGD",
            "checkpoint_interval": 2,
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }

    response = client.post("/api/experiments/start", json=payload)
    assert response.status_code == 200
    status = wait_for_status(client, lambda status: not status["is_running"])
    run_id = status["run_id"]

    assert status["last_checkpoint_step"] == 8
    assert status["last_checkpoint_acc"] == 12.0
    assert status["best_checkpoint_step"] == 8
    assert status["best_checkpoint_acc"] == 12.0
    assert status["best_checkpoint_path"] == str(tmp_path / run_id / "latest.pt")

    latest_metadata = (tmp_path / run_id / "latest.json").read_text(encoding="utf-8")
    assert '"checkpoint": "latest.pt"' in latest_metadata
    assert '"step": 8' in latest_metadata
    assert '"last_checkpoint_acc": 12.0' in latest_metadata
    assert '"best_checkpoint_path": "' in latest_metadata
    assert not (tmp_path / run_id / "best.pt").exists()
    assert not (tmp_path / run_id / "best.json").exists()


def test_api_best_checkpoint_ignores_regressions_and_ties(tmp_path, monkeypatch) -> None:
    validation_accuracies = iter([10.0, 8.0, 10.0])

    def evaluate_sequence(model, loader, device) -> float:
        return next(validation_accuracies)

    monkeypatch.setattr(experiment, "evaluate", evaluate_sequence)
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 3,
            "batch_size": 4,
            "iterations": 6,
            "target_acc": 100.0,
            "optimizer": "SGD",
            "checkpoint_interval": 2,
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }

    response = client.post("/api/experiments/start", json=payload)
    assert response.status_code == 200
    status = wait_for_status(client, lambda status: not status["is_running"])
    run_id = status["run_id"]

    assert status["last_checkpoint_step"] == 6
    assert status["last_checkpoint_acc"] == 10.0
    assert status["best_checkpoint_step"] == 2
    assert status["best_checkpoint_acc"] == 10.0
    assert status["best_checkpoint_path"] == str(tmp_path / run_id / "best.pt")

    latest_metadata = (tmp_path / run_id / "latest.json").read_text(encoding="utf-8")
    best_metadata = (tmp_path / run_id / "best.json").read_text(encoding="utf-8")
    assert '"checkpoint": "latest.pt"' in latest_metadata
    assert '"step": 6' in latest_metadata
    assert '"last_checkpoint_acc": 10.0' in latest_metadata
    assert '"checkpoint": "best.pt"' in best_metadata
    assert '"step": 2' in best_metadata
    assert '"best_checkpoint_acc": 10.0' in best_metadata


def test_build_run_id_uses_stable_config_slug() -> None:
    config = ExperimentConfig(
        dataset="mnist",
        device="cpu",
        seed=7,
        batch_size=32,
        iterations=123,
        optimizer="SGD",
    )

    run_id = build_run_id(
        config,
        started_at=datetime(2026, 6, 20, 18, 45, 12, 123456, tzinfo=UTC),
    )

    assert run_id == "mnist-sgd-cpu-seed7-20260620T154512123456"


def test_api_pause_resume_and_stop_clears_paused_state(tmp_path, monkeypatch) -> None:
    def build_slow_runner(optimizer_name, model, raw_params, *, device, seed, **kwargs):
        assert optimizer_name == "SGD"
        return SlowSGDRunner(model, SGDConfig(eta=float(raw_params.get(ETA, 0.01))))

    monkeypatch.setattr(experiment, "build_optimizer_runner", build_slow_runner)
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    assert client.post("/api/experiments/pause").status_code == 409
    assert client.post("/api/experiments/resume").status_code == 409

    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 11,
            "batch_size": 4,
            "iterations": 100,
            "target_acc": 100.0,
            "optimizer": "SGD",
            "deterministic": True,
            "checkpoint_interval": 0,
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }

    response = client.post("/api/experiments/start", json=payload)
    assert response.status_code == 200
    wait_for_status(client, lambda status: status["current_step"] >= 1)

    pause_response = client.post("/api/experiments/pause")
    assert pause_response.status_code == 200
    assert pause_response.json()["pause_requested"] is True

    paused = wait_for_status(client, lambda status: status["is_paused"])
    paused_step = paused["current_step"]
    run_id = paused["run_id"]
    assert paused["is_running"] is False
    assert paused["last_checkpoint_step"] == paused_step
    assert paused["checkpoint_path"] == str(tmp_path / run_id / "latest.pt")
    assert (tmp_path / run_id / "latest.pt").exists()
    assert (tmp_path / run_id / "latest.json").exists()

    assert client.post("/api/experiments/start", json=payload).status_code == 409

    resume_response = client.post("/api/experiments/resume")
    assert resume_response.status_code == 200
    resumed = wait_for_status(
        client,
        lambda status: status["is_running"] and status["current_step"] > paused_step,
    )
    assert resumed["run_id"] == run_id
    assert resumed["is_paused"] is False

    second_pause_response = client.post("/api/experiments/pause")
    assert second_pause_response.status_code == 200
    paused_again = wait_for_status(client, lambda status: status["is_paused"])
    assert paused_again["is_paused"] is True

    stop_response = client.post("/api/experiments/stop")
    assert stop_response.status_code == 200
    stopped = stop_response.json()
    assert stopped["is_paused"] is False
    assert stopped["is_running"] is False


def test_api_pause_evaluates_checkpoint_accuracy_before_saving(
    tmp_path,
    monkeypatch,
) -> None:
    def build_slow_runner(optimizer_name, model, raw_params, *, device, seed, **kwargs):
        assert optimizer_name == "SGD"
        return SlowSGDRunner(model, SGDConfig(eta=float(raw_params.get(ETA, 0.01))))

    def evaluate_fixed(model, loader, device) -> float:
        return 17.5

    monkeypatch.setattr(experiment, "build_optimizer_runner", build_slow_runner)
    monkeypatch.setattr(experiment, "evaluate", evaluate_fixed)
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))
    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 11,
            "batch_size": 4,
            "iterations": 100,
            "target_acc": 100.0,
            "optimizer": "SGD",
            "checkpoint_interval": 0,
        },
        "opt_params": {"SGD": {ETA: 0.01}},
    }

    assert client.post("/api/experiments/start", json=payload).status_code == 200
    wait_for_status(client, lambda status: status["current_step"] >= 1)
    assert client.post("/api/experiments/pause").status_code == 200
    paused = wait_for_status(client, lambda status: status["is_paused"])
    run_id = paused["run_id"]

    assert paused["last_checkpoint_step"] == paused["current_step"]
    assert paused["last_checkpoint_acc"] == 17.5
    assert paused["best_checkpoint_acc"] == 17.5
    assert paused["best_checkpoint_step"] == paused["current_step"]
    assert paused["checkpoint_path"] == str(tmp_path / run_id / "latest.pt")
    assert paused["best_checkpoint_path"] == str(tmp_path / run_id / "latest.pt")
    assert not (tmp_path / run_id / "best.pt").exists()

    metadata = manager.checkpoint_saver.load_latest_metadata(run_id)
    assert metadata["last_checkpoint_acc"] == 17.5
    assert metadata["best_checkpoint_acc"] == 17.5


def test_api_reports_leea_mutation_step(tmp_path) -> None:
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    client = TestClient(create_app(manager))

    payload = {
        "config": {
            "dataset": "mnist",
            "device": "cpu",
            "seed": 5,
            "batch_size": 4,
            "iterations": 1,
            "target_acc": 100.0,
            "optimizer": "LEEA",
        },
        "opt_params": {"LEEA": {"N": 4, ETA_0: 0.2}},
    }

    response = client.post("/api/experiments/start", json=payload)

    assert response.status_code == 200

    deadline = time.monotonic() + 5
    status = client.get("/api/experiments/status").json()
    while status["current_step"] == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
        status = client.get("/api/experiments/status").json()

    assert status["optimizer"] == "LEEA"
    assert status["current_mutation_step"] == 0.2
    assert status["history"]["mutation_step"] == [{"i": 1, "value": 0.2}]


def test_sse_event_serialization() -> None:
    payload = format_sse("step", ExperimentStatus(is_running=True, current_step=1))

    assert payload.startswith("event: step\n")
    assert '"current_step": 1' in payload
    assert payload.endswith("\n\n")

    failed_payload = format_sse("failed", ExperimentStatus(error="boom"))
    assert failed_payload.startswith("event: failed\n")
    assert '"error": "boom"' in failed_payload


def test_gpu_request_errors_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(experiment.torch.cuda, "is_available", lambda: False)

    try:
        resolve_device("gpu")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("GPU request should fail when CUDA is unavailable")

    assert "CUDA was requested" in message
    assert "direnv allow" in message
    assert "repository root" in message


def test_seed_everything_configures_deterministic_torch(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed_everything(123)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_seed_everything_fast_enables_opt_in_nondeterministic_controls() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    seed_everything(123, numeric_mode="fast")

    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    seed_everything(123)
