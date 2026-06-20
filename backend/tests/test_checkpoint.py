import json

import torch

from kiseki.checkpoint import CheckpointSaver, build_runtime_manifest
from kiseki.models import CNN2C2DMNIST
from kiseki.schemas import ETA, ExperimentConfig, ExperimentStatus


def test_checkpoint_saver_overwrites_run_latest_and_writes_metadata(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    model = CNN2C2DMNIST()
    config = ExperimentConfig(
        device="cpu",
        optimizer="SGD",
        deterministic=True,
        checkpoint_interval=2,
    )
    optimizer_params = {"SGD": {ETA: 0.03}}

    first_status = ExperimentStatus(
        run_id="run-1",
        optimizer="SGD",
        current_step=1,
        current_loss=2.0,
        best_acc=5.0,
        checkpoint_path=str(saver.latest_pt_path("run-1")),
    )
    second_status = first_status.model_copy(update={"current_step": 2, "current_loss": 1.5})

    first_path = saver.save(
        model=model,
        status=first_status,
        config=config,
        optimizer="SGD",
        run_id="run-1",
        optimizer_params=optimizer_params,
        saved_at="2026-06-20T00:00:00+00:00",
        runtime_manifest=build_runtime_manifest(torch.device("cpu")),
    )
    second_path = saver.save(
        model=model,
        status=second_status,
        config=config,
        optimizer="SGD",
        run_id="run-1",
        optimizer_params=optimizer_params,
        saved_at="2026-06-20T00:01:00+00:00",
        runtime_manifest=build_runtime_manifest(torch.device("cpu")),
    )

    assert first_path == second_path == tmp_path / "run-1" / "latest.pt"
    assert not list((tmp_path / "run-1").glob("step-*.pt"))

    metadata = json.loads((tmp_path / "run-1" / "latest.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "run-1"
    assert metadata["checkpoint"] == "latest.pt"
    assert metadata["step"] == 2
    assert metadata["optimizer"] == "SGD"
    assert metadata["dataset"] == "mnist"
    assert metadata["seed"] == 42
    assert metadata["deterministic"] is True
    assert metadata["checkpoint_interval"] == 2
    assert metadata["optimizer_params"] == optimizer_params
    assert metadata["best_acc"] == 5.0
    assert metadata["current_loss"] == 1.5
    assert metadata["reproducibility_mode"] == "exact"

    payload = saver.load_latest("run-1")
    assert payload["status"]["current_step"] == 2
    assert payload["config"]["checkpoint_interval"] == 2
