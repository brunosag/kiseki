import time

import torch
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader, TensorDataset

from kiseki.api import create_app
from kiseki.checkpoint import CheckpointSaver
from kiseki.experiment import ExperimentManager, format_sse
from kiseki.schemas import ETA, ExperimentStatus


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

    stop_response = client.post("/api/experiments/stop")
    assert stop_response.status_code == 200


def test_sse_event_serialization() -> None:
    payload = format_sse("step", ExperimentStatus(is_running=True, current_step=1))

    assert payload.startswith("event: step\n")
    assert '"current_step": 1' in payload
    assert payload.endswith("\n\n")
