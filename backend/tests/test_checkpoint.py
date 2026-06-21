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
        total_elapsed_seconds=4.5,
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
    assert metadata["total_elapsed_seconds"] == 4.5
    assert metadata["reproducibility_mode"] == "exact"

    payload = saver.load_latest("run-1")
    assert payload["status"]["current_step"] == 2
    assert payload["config"]["checkpoint_interval"] == 2


def test_checkpoint_saver_writes_best_checkpoint_independently(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    model = CNN2C2DMNIST()
    config = ExperimentConfig(device="cpu", optimizer="SGD")
    optimizer_params = {"SGD": {ETA: 0.03}}
    latest_status = ExperimentStatus(
        run_id="run-1",
        optimizer="SGD",
        current_step=4,
        current_loss=1.5,
        best_acc=8.0,
        last_checkpoint_step=4,
        last_checkpoint_acc=8.0,
        checkpoint_path=str(saver.latest_pt_path("run-1")),
    )
    best_status = latest_status.model_copy(
        update={
            "current_step": 6,
            "current_loss": 1.2,
            "best_acc": 12.0,
            "last_checkpoint_step": 6,
            "last_checkpoint_acc": 12.0,
            "best_checkpoint_acc": 12.0,
            "best_checkpoint_step": 6,
            "best_checkpoint_saved_at": "2026-06-20T00:01:00+00:00",
            "best_checkpoint_path": str(saver.best_pt_path("run-1")),
        }
    )

    latest_path = saver.save(
        model=model,
        status=latest_status,
        config=config,
        optimizer="SGD",
        run_id="run-1",
        optimizer_params=optimizer_params,
        saved_at="2026-06-20T00:00:00+00:00",
    )
    best_path = saver.save(
        model=model,
        status=best_status,
        config=config,
        optimizer="SGD",
        run_id="run-1",
        optimizer_params=optimizer_params,
        saved_at="2026-06-20T00:01:00+00:00",
        kind="best",
    )

    assert latest_path == tmp_path / "run-1" / "latest.pt"
    assert best_path == tmp_path / "run-1" / "best.pt"

    latest_metadata = saver.load_latest_metadata("run-1")
    best_metadata = saver.load_best_metadata("run-1")
    assert latest_metadata["checkpoint"] == "latest.pt"
    assert latest_metadata["step"] == 4
    assert latest_metadata["last_checkpoint_acc"] == 8.0
    assert best_metadata["checkpoint"] == "best.pt"
    assert best_metadata["step"] == 6
    assert best_metadata["best_checkpoint_acc"] == 12.0
    assert best_metadata["best_checkpoint_path"] == str(tmp_path / "run-1" / "best.pt")

    best_payload = saver.load_best("run-1")
    assert best_payload["status"]["current_step"] == 6


def test_checkpoint_listing_reads_elapsed_from_legacy_payload_metadata(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    model = CNN2C2DMNIST()
    config = ExperimentConfig(device="cpu", optimizer="SGD")
    status = ExperimentStatus(
        run_id="run-1",
        optimizer="SGD",
        current_step=4,
        current_loss=1.5,
        total_elapsed_seconds=12.5,
        checkpoint_path=str(saver.latest_pt_path("run-1")),
    )
    saver.save(
        model=model,
        status=status,
        config=config,
        optimizer="SGD",
        run_id="run-1",
        saved_at="2026-06-20T00:00:00+00:00",
    )
    metadata_path = saver.latest_metadata_path("run-1")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("total_elapsed_seconds")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summaries = saver.list_summaries()

    assert summaries[0].total_elapsed_seconds == 12.5


def test_checkpoint_saver_lists_latest_only_and_promotes_latest_to_best(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    model = CNN2C2DMNIST()
    config = ExperimentConfig(device="cpu", optimizer="SGD")
    first_saved_at = "2026-06-20T00:00:00+00:00"
    status = ExperimentStatus(
        run_id="run-1",
        optimizer="SGD",
        current_step=2,
        best_acc=10.0,
        last_checkpoint_step=2,
        last_checkpoint_acc=10.0,
        last_checkpoint_saved_at=first_saved_at,
        checkpoint_path=str(saver.latest_pt_path("run-1")),
        best_checkpoint_acc=10.0,
        best_checkpoint_step=2,
        best_checkpoint_saved_at=first_saved_at,
        best_checkpoint_path=str(saver.latest_pt_path("run-1")),
    )

    saver.save(
        model=model,
        status=status,
        config=config,
        optimizer="SGD",
        run_id="run-1",
        saved_at=first_saved_at,
    )

    assert saver.promote_latest_to_best("run-1") is True
    best_metadata = saver.load_best_metadata("run-1")
    assert best_metadata["checkpoint"] == "best.pt"
    assert best_metadata["checkpoint_path"] == str(tmp_path / "run-1" / "best.pt")
    assert best_metadata["best_checkpoint_acc"] == 10.0
    assert best_metadata["best_checkpoint_step"] == 2
    assert best_metadata["best_checkpoint_saved_at"] == first_saved_at

    summaries = saver.list_summaries()

    assert [(summary.run_id, summary.kind) for summary in summaries] == [("run-1", "latest")]
