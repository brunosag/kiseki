import os
import time

import torch
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader, TensorDataset

from kiseki.api import create_app
from kiseki.checkpoint import CheckpointSaver
from kiseki import experiment
from kiseki.experiment import ExperimentManager, format_sse, resolve_device, seed_everything
from kiseki.schemas import ETA, ETA_0, ExperimentStatus


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
