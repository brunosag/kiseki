import json
import platform
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .schemas import ExperimentConfig, ExperimentStatus

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class CheckpointSaver:
    directory: Path = Path("checkpoints")

    def run_directory(self, run_id: str) -> Path:
        return self.directory / run_id

    def latest_pt_path(self, run_id: str) -> Path:
        return self.run_directory(run_id) / "latest.pt"

    def latest_metadata_path(self, run_id: str) -> Path:
        return self.run_directory(run_id) / "latest.json"

    def save(
        self,
        *,
        model: nn.Module,
        status: ExperimentStatus,
        config: ExperimentConfig,
        optimizer: str,
        run_id: str | None = None,
        optimizer_params: dict[str, dict[str, float]] | None = None,
        optimizer_state: dict[str, Any] | None = None,
        loader_state: dict[str, Any] | None = None,
        rng_state: dict[str, Any] | None = None,
        runtime_manifest: dict[str, Any] | None = None,
        saved_at: str | None = None,
        compatibility_warnings: list[str] | None = None,
    ) -> Path:
        run_id = run_id or status.run_id or "default"
        saved_at = saved_at or utc_now_iso()
        optimizer_params = optimizer_params or {}
        runtime_manifest = runtime_manifest or build_runtime_manifest()
        compatibility_warnings = compatibility_warnings or []

        run_directory = self.run_directory(run_id)
        run_directory.mkdir(parents=True, exist_ok=True)
        pt_path = self.latest_pt_path(run_id)
        metadata_path = self.latest_metadata_path(run_id)

        metadata = checkpoint_metadata(
            run_id=run_id,
            checkpoint_path=pt_path,
            saved_at=saved_at,
            status=status,
            config=config,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            runtime_manifest=runtime_manifest,
            compatibility_warnings=compatibility_warnings,
        )
        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "metadata": metadata,
            "run_id": run_id,
            "saved_at": saved_at,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer_state or {},
            "status": status.model_dump(),
            "config": config.model_dump(),
            "optimizer": optimizer,
            "optimizer_params": optimizer_params,
            "loader_state": loader_state,
            "rng_state": rng_state or capture_rng_state(),
            "runtime_manifest": runtime_manifest,
            "compatibility_warnings": compatibility_warnings,
        }
        torch.save(payload, pt_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return pt_path

    def load_latest(self, run_id: str, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        path = self.latest_pt_path(run_id)
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def load_latest_metadata(self, run_id: str) -> dict[str, Any]:
        return json.loads(self.latest_metadata_path(run_id).read_text(encoding="utf-8"))


def checkpoint_metadata(
    *,
    run_id: str,
    checkpoint_path: Path,
    saved_at: str,
    status: ExperimentStatus,
    config: ExperimentConfig,
    optimizer: str,
    optimizer_params: dict[str, dict[str, float]],
    runtime_manifest: dict[str, Any],
    compatibility_warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "checkpoint": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "saved_at": saved_at,
        "step": status.current_step,
        "optimizer": optimizer,
        "dataset": config.dataset,
        "seed": config.seed,
        "requested_device": status.requested_device,
        "device": status.device,
        "device_name": status.device_name,
        "runtime_manifest": runtime_manifest,
        "deterministic": config.deterministic,
        "checkpoint_interval": config.checkpoint_interval,
        "config": config.model_dump(),
        "optimizer_params": optimizer_params,
        "best_acc": status.best_acc,
        "current_loss": status.current_loss,
        "reproducibility_mode": reproducibility_mode(config.deterministic, compatibility_warnings),
        "reproducibility_status": reproducibility_status(config.deterministic, compatibility_warnings),
        "compatibility_warnings": compatibility_warnings,
    }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)

    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def build_runtime_manifest(device: torch.device | None = None) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if device is not None:
        manifest["device"] = device.type
        manifest["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    return manifest


def compare_runtime_manifests(
    saved: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[str]:
    if not saved:
        return ["Checkpoint has no runtime manifest; resume is best-effort."]

    warnings = []
    for key in ("python", "torch", "numpy", "torch_cuda", "device"):
        saved_value = saved.get(key)
        current_value = current.get(key)
        if saved_value != current_value:
            warnings.append(
                f"Runtime mismatch for {key}: checkpoint={saved_value!r}, current={current_value!r}."
            )
    return warnings


def reproducibility_mode(deterministic: bool, warnings: list[str] | None = None) -> str:
    if deterministic and not warnings:
        return "exact"
    return "best_effort"


def reproducibility_status(deterministic: bool, warnings: list[str] | None = None) -> str:
    if deterministic and not warnings:
        return "exact deterministic resume state captured"
    if deterministic:
        return "deterministic state loaded with compatibility warnings"
    return "best-effort trajectory resume; deterministic mode is disabled"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
