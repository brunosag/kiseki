import os
import time
from datetime import UTC, datetime

import torch
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader, TensorDataset

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
from kiseki.models import CNN2C2DMNIST
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
) -> None:
    config = ExperimentConfig(
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
        model=CNN2C2DMNIST(),
        status=status,
        config=config,
        optimizer="SGD",
        run_id=run_id,
        optimizer_params={"SGD": {ETA: 0.02}},
        saved_at=saved_at,
        kind=kind,
    )


def test_api_lists_complete_checkpoints_sorted_newest_first(tmp_path) -> None:
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
        run_id="best-run",
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
        ("best-run", "best"),
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
    assert status["best_checkpoint_path"] == str(tmp_path / run_id / "best.pt")

    latest_metadata = (tmp_path / run_id / "latest.json").read_text(encoding="utf-8")
    best_metadata = (tmp_path / run_id / "best.json").read_text(encoding="utf-8")
    assert '"checkpoint": "latest.pt"' in latest_metadata
    assert '"step": 8' in latest_metadata
    assert '"last_checkpoint_acc": 12.0' in latest_metadata
    assert '"checkpoint": "best.pt"' in best_metadata
    assert '"step": 8' in best_metadata
    assert '"best_checkpoint_acc": 12.0' in best_metadata


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
