import json
import platform
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .schemas import CheckpointKind, CheckpointSummary, ExperimentConfig, ExperimentStatus

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointNotFoundError(FileNotFoundError):
    pass


@dataclass(slots=True)
class CheckpointSaver:
    directory: Path = Path("checkpoints")

    def run_directory(self, run_id: str) -> Path:
        return self.directory / run_id

    def latest_pt_path(self, run_id: str) -> Path:
        return self.run_directory(run_id) / "latest.pt"

    def latest_metadata_path(self, run_id: str) -> Path:
        return self.run_directory(run_id) / "latest.json"

    def best_pt_path(self, run_id: str) -> Path:
        return self.run_directory(run_id) / "best.pt"

    def best_metadata_path(self, run_id: str) -> Path:
        return self.run_directory(run_id) / "best.json"

    def pt_path(self, run_id: str, kind: CheckpointKind = "latest") -> Path:
        if kind == "best":
            return self.best_pt_path(run_id)
        return self.latest_pt_path(run_id)

    def metadata_path(self, run_id: str, kind: CheckpointKind = "latest") -> Path:
        if kind == "best":
            return self.best_metadata_path(run_id)
        return self.latest_metadata_path(run_id)

    def load(
        self,
        run_id: str,
        kind: CheckpointKind = "latest",
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        path = self.pt_path(run_id, kind)
        if not path.exists():
            raise CheckpointNotFoundError(
                f"{kind.title()} checkpoint for run {run_id!r} was not found"
            )

        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def load_metadata(self, run_id: str, kind: CheckpointKind = "latest") -> dict[str, Any]:
        return json.loads(self.metadata_path(run_id, kind).read_text(encoding="utf-8"))

    def list_summaries(
        self,
        kinds: tuple[CheckpointKind, ...] = ("latest",),
    ) -> list[CheckpointSummary]:
        summaries: list[CheckpointSummary] = []
        if not self.directory.exists():
            return summaries

        for run_directory in self.directory.iterdir():
            if not run_directory.is_dir():
                continue

            run_id = run_directory.name
            for kind in kinds:
                pt_path = self.pt_path(run_id, kind)
                metadata_path = self.metadata_path(run_id, kind)
                if not pt_path.exists() or not metadata_path.exists():
                    continue

                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if "total_elapsed_seconds" not in metadata:
                        metadata = self._metadata_with_payload_elapsed(metadata, run_id, kind)
                    summaries.append(checkpoint_summary_from_metadata(metadata, run_id, kind))
                except Exception:
                    continue

        return sorted(
            summaries, key=lambda summary: saved_at_sort_key(summary.saved_at), reverse=True
        )

    def list_analysis_summaries(self) -> list[CheckpointSummary]:
        summaries: list[CheckpointSummary] = []
        if not self.directory.exists():
            return summaries

        for run_directory in self.directory.iterdir():
            if not run_directory.is_dir():
                continue

            run_id = run_directory.name
            for kind in ("best", "latest"):
                pt_path = self.pt_path(run_id, kind)
                metadata_path = self.metadata_path(run_id, kind)
                if not pt_path.exists() or not metadata_path.exists():
                    continue

                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if "total_elapsed_seconds" not in metadata:
                        metadata = self._metadata_with_payload_elapsed(metadata, run_id, kind)
                    summaries.append(checkpoint_summary_from_metadata(metadata, run_id, kind))
                    break
                except Exception:
                    continue

        return sorted(
            summaries, key=lambda summary: saved_at_sort_key(summary.saved_at), reverse=True
        )

    def has_latest(self, run_id: str) -> bool:
        return self.latest_pt_path(run_id).exists() and self.latest_metadata_path(run_id).exists()

    def has_best(self, run_id: str) -> bool:
        return self.best_pt_path(run_id).exists() and self.best_metadata_path(run_id).exists()

    def delete_best(self, run_id: str) -> None:
        self.best_pt_path(run_id).unlink(missing_ok=True)
        self.best_metadata_path(run_id).unlink(missing_ok=True)

    def promote_latest_to_best(self, run_id: str) -> bool:
        if not self.has_latest(run_id):
            return False

        payload = self.load_latest(run_id, map_location="cpu")
        metadata = self.load_latest_metadata(run_id)
        saved_at = str(metadata.get("saved_at") or payload.get("saved_at") or utc_now_iso())
        step = int(metadata["step"])
        best_pt_path = self.best_pt_path(run_id)
        best_metadata_path = self.best_metadata_path(run_id)
        best_accuracy = metadata.get("last_checkpoint_acc")
        if best_accuracy is None:
            best_accuracy = metadata.get("best_checkpoint_acc", metadata.get("best_acc"))

        metadata = {
            **metadata,
            "checkpoint": best_pt_path.name,
            "checkpoint_path": str(best_pt_path),
            "best_checkpoint_acc": best_accuracy,
            "best_checkpoint_step": step,
            "best_checkpoint_saved_at": saved_at,
            "best_checkpoint_path": str(best_pt_path),
        }

        status_payload = payload.get("status")
        if isinstance(status_payload, dict):
            status = ExperimentStatus.model_validate(status_payload)
            status.best_checkpoint_acc = best_accuracy
            status.best_checkpoint_step = step
            status.best_checkpoint_saved_at = saved_at
            status.best_checkpoint_path = str(best_pt_path)
            payload["status"] = status.model_dump()

        payload["metadata"] = metadata
        best_pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, best_pt_path)
        best_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return True

    def delete_run(self, run_id: str) -> None:
        run_directory = self._safe_run_directory(run_id)
        if not run_directory.exists() or not run_directory.is_dir():
            raise CheckpointNotFoundError(f"Checkpoint run {run_id!r} was not found")

        shutil.rmtree(run_directory)

    def _safe_run_directory(self, run_id: str) -> Path:
        run_directory = self.run_directory(run_id)
        base_directory = self.directory.resolve(strict=False)
        resolved_run_directory = run_directory.resolve(strict=False)
        if (
            resolved_run_directory == base_directory
            or not resolved_run_directory.is_relative_to(base_directory)
        ):
            raise CheckpointNotFoundError(f"Checkpoint run {run_id!r} was not found")
        return run_directory

    def _metadata_with_payload_elapsed(
        self,
        metadata: dict[str, Any],
        run_id: str,
        kind: CheckpointKind,
    ) -> dict[str, Any]:
        payload = self.load(run_id, kind, map_location="cpu")
        status = payload.get("status")
        if not isinstance(status, dict) or "total_elapsed_seconds" not in status:
            return metadata

        return {**metadata, "total_elapsed_seconds": status["total_elapsed_seconds"]}

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
        kind: CheckpointKind = "latest",
    ) -> Path:
        run_id = run_id or status.run_id or "default"
        saved_at = saved_at or utc_now_iso()
        optimizer_params = optimizer_params or {}
        runtime_manifest = runtime_manifest or build_runtime_manifest()
        compatibility_warnings = compatibility_warnings or []

        run_directory = self.run_directory(run_id)
        run_directory.mkdir(parents=True, exist_ok=True)
        pt_path = self.pt_path(run_id, kind)
        metadata_path = self.metadata_path(run_id, kind)

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

    def load_latest(
        self, run_id: str, *, map_location: str | torch.device = "cpu"
    ) -> dict[str, Any]:
        return self.load(run_id, "latest", map_location=map_location)

    def load_latest_metadata(self, run_id: str) -> dict[str, Any]:
        return self.load_metadata(run_id, "latest")

    def load_best(self, run_id: str, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        return self.load(run_id, "best", map_location=map_location)

    def load_best_metadata(self, run_id: str) -> dict[str, Any]:
        return self.load_metadata(run_id, "best")


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
        "last_checkpoint_acc": status.last_checkpoint_acc,
        "best_checkpoint_acc": status.best_checkpoint_acc,
        "best_checkpoint_step": status.best_checkpoint_step,
        "best_checkpoint_saved_at": status.best_checkpoint_saved_at,
        "best_checkpoint_path": status.best_checkpoint_path,
        "current_loss": status.current_loss,
        "total_elapsed_seconds": status.total_elapsed_seconds,
        "reproducibility_mode": reproducibility_mode(config.deterministic, compatibility_warnings),
        "reproducibility_status": reproducibility_status(
            config.deterministic, compatibility_warnings
        ),
        "compatibility_warnings": compatibility_warnings,
    }


def checkpoint_summary_from_metadata(
    metadata: dict[str, Any],
    run_id: str,
    kind: CheckpointKind,
) -> CheckpointSummary:
    config = ExperimentConfig.model_validate(metadata.get("config"))
    accuracy = metadata.get("best_acc")
    if accuracy is None:
        accuracy = (
            metadata.get("best_checkpoint_acc")
            if kind == "best"
            else metadata.get("last_checkpoint_acc")
        )

    return CheckpointSummary(
        run_id=str(metadata.get("run_id") or run_id),
        kind=kind,
        saved_at=str(metadata["saved_at"]),
        step=int(metadata["step"]),
        optimizer=metadata.get("optimizer") or config.optimizer,
        dataset=metadata.get("dataset") or config.dataset,
        seed=int(metadata.get("seed") or config.seed),
        requested_device=metadata.get("requested_device"),
        device=str(metadata.get("device") or metadata.get("requested_device") or config.device),
        device_name=metadata.get("device_name"),
        deterministic=bool(metadata.get("deterministic", config.deterministic)),
        accuracy=accuracy,
        best_acc=metadata.get("best_acc"),
        current_loss=metadata.get("current_loss"),
        total_elapsed_seconds=metadata.get("total_elapsed_seconds"),
        reproducibility_mode=metadata.get("reproducibility_mode", "best_effort"),
        reproducibility_status=metadata.get(
            "reproducibility_status",
            reproducibility_status(
                config.deterministic, metadata.get("compatibility_warnings", [])
            ),
        ),
        compatibility_warnings=metadata.get("compatibility_warnings", []),
        config=config,
        optimizer_params=metadata.get("optimizer_params", {}),
    )


def saved_at_sort_key(saved_at: str) -> datetime:
    try:
        return datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


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
        manifest["device_name"] = (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        )
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
