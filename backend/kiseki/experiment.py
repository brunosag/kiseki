import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, replace
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from .analysis import AnalysisService
from .checkpoint import (
    CheckpointSaver,
    build_runtime_manifest,
    capture_rng_state,
    compare_runtime_manifests,
    reproducibility_mode,
    restore_rng_state,
    utc_now_iso,
)
from .data import DataLoaderFactory, cycle_loader, deterministic_batch_stream, train_val_loaders
from .models import build_model
from .optimizers import build_optimizer_runner
from .schemas import (
    AccuracyPoint,
    AnalysisComparisonJobRequest,
    AnalysisComparisonJobStatus,
    AnalysisComparisonReport,
    CheckpointListMode,
    CheckpointKind,
    CheckpointSelection,
    CheckpointSummary,
    ExperimentConfig,
    ExperimentControlsUpdate,
    ExperimentStatus,
    ExperimentStatusCompactEvent,
    MutationStepPoint,
    StartExperimentRequest,
    TrainingHistoryDelta,
)

VAL_FREQ = 10
FRONTEND_STEP_UPDATE_INTERVAL_SECONDS = 0.2
NumericMode = Literal["strict", "fast"]
NUMERIC_MODES = ("strict", "fast")
NIXOS_CUDA_LIBRARY = Path("/run/opengl-driver/lib/libcuda.so.1")
NIXOS_CUDA_HINT = (
    "CUDA was requested, but torch.cuda.is_available() is false. "
    "From the repository root, run `direnv allow` and restart the API process so shell.nix "
    "exports the CUDA driver path before Python starts."
)


@dataclass(frozen=True)
class ExperimentStreamPayload:
    status_patch: dict[str, Any]
    history_delta: TrainingHistoryDelta
    replace_history: bool = False
    full_status: ExperimentStatus | None = None


class ExperimentManager:
    def __init__(
        self,
        *,
        data_loader_factory: DataLoaderFactory | None = None,
        checkpoint_saver: CheckpointSaver | None = None,
    ) -> None:
        self.data_loader_factory = data_loader_factory or DataLoaderFactory(Path("data"))
        self.checkpoint_saver = checkpoint_saver or CheckpointSaver(Path("checkpoints"))
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.lock = threading.Lock()
        self.subscribers: dict[Queue[tuple[str, ExperimentStreamPayload]], bool] = {}
        self.worker: threading.Thread | None = None
        self._status = ExperimentStatus()
        self._resume_checkpoint_kind: CheckpointKind = "latest"
        self._last_step_publish_at = 0.0
        self._pending_step_delta = TrainingHistoryDelta()
        self.analysis_service = AnalysisService(
            data_loader_factory=self.data_loader_factory,
            checkpoint_saver=self.checkpoint_saver,
            is_experiment_running=self._is_experiment_running,
        )

    def status(self) -> ExperimentStatus:
        with self.lock:
            return self._status.model_copy(deep=True)

    def checkpoints(self, mode: CheckpointListMode = "training") -> list[CheckpointSummary]:
        if mode == "analysis":
            return self.analysis_service.checkpoint_summaries()
        return self.checkpoint_saver.list_summaries()

    def create_analysis_comparison_job(
        self,
        request: AnalysisComparisonJobRequest,
    ) -> AnalysisComparisonJobStatus:
        return self.analysis_service.create_comparison_job(request)

    def get_analysis_comparison_job(self, job_id: str) -> AnalysisComparisonJobStatus:
        return self.analysis_service.get_comparison_job(job_id)

    def get_analysis_comparison_report(self, job_id: str) -> AnalysisComparisonReport:
        return self.analysis_service.get_comparison_report(job_id)

    def analysis_comparison_events(self, job_id: str) -> Iterator[str]:
        return self.analysis_service.comparison_events(job_id)

    def delete_checkpoint(self, run_id: str) -> None:
        with self.lock:
            if self._status.is_running:
                raise RuntimeError("A checkpoint cannot be deleted while an experiment is running")
            if self._status.is_paused and self._status.run_id == run_id:
                raise RuntimeError("Stop or resume the paused experiment before deleting it")

        self.checkpoint_saver.delete_run(run_id)

    def load_checkpoint(self, selection: CheckpointSelection) -> ExperimentStatus:
        with self.lock:
            if self._status.is_running:
                raise RuntimeError("An experiment is already running")

        checkpoint = self.checkpoint_saver.load(
            selection.run_id,
            selection.kind,
            map_location="cpu",
        )
        config = ExperimentConfig.model_validate(checkpoint["config"])
        saved_status = ExperimentStatus.model_validate(checkpoint["status"])

        with self.lock:
            if self._status.is_running:
                raise RuntimeError("An experiment is already running")

            restored = saved_status.model_copy(deep=True)
            restored.is_running = False
            restored.is_paused = True
            restored.pause_requested = False
            restored.error = None
            restored.run_id = selection.run_id
            restored.optimizer = config.optimizer
            restored.requested_device = config.device
            self._status = restored
            self._resume_checkpoint_kind = selection.kind
            self._pending_step_delta = TrainingHistoryDelta()
            status = self._status.model_copy(deep=True)
        self._publish("paused", status, replace_history=True)
        return status

    def start(self, request: StartExperimentRequest) -> ExperimentStatus:
        with self.lock:
            if self._status.is_running:
                raise RuntimeError("An experiment is already running")
            if self._status.is_paused:
                raise RuntimeError("Resume or stop the paused experiment before starting a new one")

        checkpoint = None
        run_id = build_run_id(request.config)
        effective_request = request
        if request.checkpoint is not None:
            run_id = request.checkpoint.run_id
            checkpoint = self.checkpoint_saver.load(
                request.checkpoint.run_id,
                request.checkpoint.kind,
                map_location="cpu",
            )
            config = ExperimentConfig.model_validate(checkpoint["config"])
            effective_request = StartExperimentRequest(
                config=config,
                opt_params=checkpoint.get("optimizer_params", {}),
                checkpoint=request.checkpoint,
            )

        with self.lock:
            if self._status.is_running:
                raise RuntimeError("An experiment is already running")
            if self._status.is_paused:
                raise RuntimeError("Resume or stop the paused experiment before starting a new one")

            self.stop_event.clear()
            self.pause_event.clear()
            self._resume_checkpoint_kind = "latest"
            self._last_step_publish_at = 0.0
            self._pending_step_delta = TrainingHistoryDelta()
            self._status = ExperimentStatus(
                is_running=True,
                optimizer=effective_request.config.optimizer,
                requested_device=effective_request.config.device,
                run_id=run_id,
                reproducibility_mode=reproducibility_mode(effective_request.config.deterministic),
            )
            self.worker = threading.Thread(
                target=self._run,
                args=(effective_request, run_id, checkpoint),
                daemon=True,
            )
            self.worker.start()
            status = self._status.model_copy(deep=True)
        self._publish("started", status, replace_history=True)
        return status

    def pause(self) -> ExperimentStatus:
        with self.lock:
            if not self._status.is_running:
                raise RuntimeError("No running experiment can be paused")
            if self._status.pause_requested:
                raise RuntimeError("Pause is already requested")
            self.pause_event.set()
            self._status.pause_requested = True
            status = self._status.model_copy(deep=True)
        self._publish("pause_requested", status)
        return status

    def resume(self, update: ExperimentControlsUpdate | None = None) -> ExperimentStatus:
        with self.lock:
            if self._status.is_running:
                raise RuntimeError("An experiment is already running")
            if not self._status.is_paused or self._status.run_id is None:
                raise RuntimeError("No paused experiment can be resumed")

            run_id = self._status.run_id
            checkpoint_kind = self._resume_checkpoint_kind
            self.stop_event.clear()
            self.pause_event.clear()
            self._last_step_publish_at = 0.0
            self._pending_step_delta = TrainingHistoryDelta()
            self._status.is_running = True
            self._status.is_paused = False
            self._status.pause_requested = False
            self._status.error = None
            self.worker = threading.Thread(
                target=self._resume_run,
                args=(run_id, checkpoint_kind, update),
                daemon=True,
            )
            self.worker.start()
            status = self._status.model_copy(deep=True)
        self._publish("resumed", status)
        return status

    def reset(self) -> ExperimentStatus:
        with self.lock:
            if self._status.is_running:
                raise RuntimeError("Pause the running experiment before starting a new experiment")

            self.pause_event.clear()
            self.stop_event.clear()
            self._resume_checkpoint_kind = "latest"
            self._pending_step_delta = TrainingHistoryDelta()
            self._status = ExperimentStatus()
            status = self._status.model_copy(deep=True)

        self._publish("stopped", status, replace_history=True)
        return status

    def stop(self) -> ExperimentStatus:
        publish_stopped = False
        with self.lock:
            if self._status.is_paused:
                self.pause_event.clear()
                self.stop_event.clear()
                self._status.is_running = False
                self._status.is_paused = False
                self._status.pause_requested = False
                self._resume_checkpoint_kind = "latest"
                status = self._status.model_copy(deep=True)
                publish_stopped = True
            elif self._status.is_running:
                self.pause_event.clear()
                self.stop_event.set()
                self._status.pause_requested = False
                status = self._status.model_copy(deep=True)
            else:
                status = self._status.model_copy(deep=True)

        if publish_stopped:
            self._publish("stopped", status)
        return status

    def events(self, *, compact: bool = False) -> Iterator[str]:
        queue: Queue[tuple[str, ExperimentStreamPayload]] = Queue()
        with self.lock:
            self.subscribers[queue] = compact
            initial_status = self._status.model_copy(deep=True)
        yield format_sse("status", initial_status)
        try:
            while True:
                try:
                    event_type, payload = queue.get(timeout=15)
                except Empty:
                    yield ": heartbeat\n\n"
                    continue
                yield format_sse(event_type, payload, compact=compact)
        finally:
            with self.lock:
                self.subscribers.pop(queue, None)

    def _resume_run(
        self,
        run_id: str,
        kind: CheckpointKind = "latest",
        update: ExperimentControlsUpdate | None = None,
    ) -> None:
        try:
            checkpoint = self.checkpoint_saver.load(run_id, kind, map_location="cpu")
            config = ExperimentConfig.model_validate(checkpoint["config"])
            if update is not None:
                config = config.model_copy(
                    update=update.model_dump(exclude_none=True, exclude_unset=True)
                )
            request = StartExperimentRequest(
                config=config,
                opt_params=checkpoint.get("optimizer_params", {}),
            )
        except Exception as exc:  # pragma: no cover - surfaced through API status.
            status = self._set_finished(error=str(exc))
            self._publish("failed", status)
            return

        self._run(request, run_id, checkpoint)

    def _run(
        self,
        request: StartExperimentRequest,
        run_id: str,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        final_event = "completed"
        started_at = time.perf_counter()
        elapsed_offset = 0.0
        try:
            config = request.config
            seed_everything(config.seed, numeric_mode="strict" if config.deterministic else "fast")
            device, resume_warnings = resolve_run_device(
                config.device, allow_cpu_fallback=checkpoint is not None
            )
            current_manifest = build_runtime_manifest(device)
            compatibility_warnings = list(resume_warnings)
            if checkpoint is not None:
                compatibility_warnings.extend(
                    compare_runtime_manifests(
                        checkpoint.get("runtime_manifest"),
                        current_manifest,
                    )
                )

            if checkpoint is None:
                status = self._update_runtime(
                    config.device,
                    device,
                    reproducibility_mode=reproducibility_mode(config.deterministic),
                    checkpoint_warnings=[],
                )
                self._publish("runtime", status, replace_history=True)

            if hasattr(self.data_loader_factory, "pin_memory"):
                self.data_loader_factory.pin_memory = device.type == "cuda"
            train_loader, val_loader = train_val_loaders(
                self.data_loader_factory,
                config.dataset,
                batch_size=config.batch_size,
                seed=config.seed,
            )
            model = build_model(config.dataset).to(device)
            runner = build_optimizer_runner(
                config.optimizer,
                model,
                request.opt_params.get(config.optimizer, {}),
                device=device,
                seed=config.seed,
            )
            if config.deterministic:
                train_batches = deterministic_batch_stream(
                    train_loader,
                    val_loader,
                    batch_size=config.batch_size,
                    seed=config.seed,
                )
            else:
                train_batches = cycle_loader(train_loader)

            first_step = 1
            interval_losses: list[float] = []
            interval_iteration_seconds: list[float] = []
            if checkpoint is not None:
                saved_status = ExperimentStatus.model_validate(checkpoint["status"])
                elapsed_offset = saved_status.total_elapsed_seconds
                model.load_state_dict(checkpoint["model_state"])
                load_runner_state(runner, checkpoint.get("optimizer_state", {}))
                if config.deterministic and checkpoint.get("loader_state") is not None:
                    train_batches.load_state_dict(checkpoint["loader_state"])
                restore_rng_state(checkpoint.get("rng_state"))
                status = self._restore_status_for_resume(
                    saved_status,
                    config=config,
                    device=device,
                    compatibility_warnings=compatibility_warnings,
                )
                self._publish("runtime", status, replace_history=True)
                first_step = saved_status.current_step + 1

            for step in range(first_step, config.iterations + 1):
                if self.stop_event.is_set():
                    final_event = "stopped"
                    break

                iteration_started_at = time.perf_counter()
                inputs, targets = next(train_batches)
                inputs, targets = move_batch(inputs, targets, device)
                loss = runner.step(inputs, targets)
                train_accuracy = batch_accuracy(model, inputs, targets)
                iteration_seconds = time.perf_counter() - iteration_started_at
                interval_losses.append(loss)
                interval_iteration_seconds.append(iteration_seconds)
                loss_mean, loss_stdev, mean_iteration_seconds = interval_training_stats(
                    interval_losses,
                    interval_iteration_seconds,
                )
                elapsed_seconds = elapsed_offset + time.perf_counter() - started_at
                step_payload = self._update_step(
                    step,
                    loss,
                    elapsed_seconds=elapsed_seconds,
                    last_iteration_seconds=iteration_seconds,
                    loss_mean_since_validation=loss_mean,
                    loss_stdev_since_validation=loss_stdev,
                    mean_iteration_seconds_since_validation=mean_iteration_seconds,
                    mutation_step=current_mutation_step(runner),
                    train_accuracy=train_accuracy,
                    peak_memory_mb=peak_memory_mb(device),
                )
                self._publish_step(step_payload)

                checkpoint_accuracy = None
                best_accuracy_surpassed = False
                target_reached = False
                if step % VAL_FREQ == 0:
                    accuracy = evaluate(model, val_loader, device)
                    validation_loss = evaluate_mean_loss(model, val_loader, device)
                    best_accuracy_surpassed = accuracy > float(
                        step_payload.status_patch["best_acc"]
                    )
                    update_scheduler = getattr(runner, "update_scheduler", None)
                    if callable(update_scheduler):
                        update_scheduler(best_accuracy_surpassed)
                    validation_payload = self._update_accuracy(
                        step,
                        accuracy,
                        validation_loss=validation_loss,
                        elapsed_seconds=elapsed_offset + time.perf_counter() - started_at,
                        mutation_step=current_mutation_step(runner),
                        peak_memory_mb=peak_memory_mb(device),
                    )
                    self._publish("validation", validation_payload)
                    checkpoint_accuracy = accuracy
                    interval_losses = []
                    interval_iteration_seconds = []
                    if accuracy >= config.target_acc:
                        target_reached = True

                checkpoint_interval_passed = (
                    config.checkpoint_interval > 0
                    and step % config.checkpoint_interval == 0
                )
                if checkpoint_interval_passed or best_accuracy_surpassed:
                    if checkpoint_accuracy is None:
                        checkpoint_accuracy = evaluate(model, val_loader, device)
                        validation_payload = self._update_accuracy(
                            step,
                            checkpoint_accuracy,
                            validation_loss=evaluate_mean_loss(model, val_loader, device),
                            elapsed_seconds=elapsed_offset + time.perf_counter() - started_at,
                            mutation_step=current_mutation_step(runner),
                            peak_memory_mb=peak_memory_mb(device),
                        )
                        self._publish("validation", validation_payload)

                    status = self._save_checkpoint(
                        model=model,
                        runner=runner,
                        train_batches=train_batches,
                        run_id=run_id,
                        config=config,
                        opt_params=request.opt_params,
                        device=device,
                        checkpoint_accuracy=checkpoint_accuracy,
                    )
                    self._publish("checkpoint", status)
                    if checkpoint_accuracy >= config.target_acc:
                        target_reached = True

                if target_reached:
                    final_event = "completed"
                    break

                if self.stop_event.is_set():
                    final_event = "stopped"
                    break

                if self.pause_event.is_set():
                    pause_accuracy = checkpoint_accuracy
                    if pause_accuracy is None:
                        pause_accuracy = current_step_accuracy(self.status(), step)

                    if pause_accuracy is None:
                        pause_accuracy = evaluate(model, val_loader, device)
                        validation_payload = self._update_accuracy(
                            step,
                            pause_accuracy,
                            validation_loss=evaluate_mean_loss(model, val_loader, device),
                            elapsed_seconds=elapsed_offset + time.perf_counter() - started_at,
                            mutation_step=current_mutation_step(runner),
                            peak_memory_mb=peak_memory_mb(device),
                        )
                        self._publish("validation", validation_payload)

                    status = self._save_checkpoint(
                        model=model,
                        runner=runner,
                        train_batches=train_batches,
                        run_id=run_id,
                        config=config,
                        opt_params=request.opt_params,
                        device=device,
                        checkpoint_accuracy=pause_accuracy,
                    )
                    self._publish("checkpoint", status)
                    status = self._set_paused(
                        elapsed_seconds=elapsed_offset + time.perf_counter() - started_at,
                    )
                    self._publish("paused", status)
                    return

        except Exception as exc:  # pragma: no cover - surfaced through API status.
            final_event = "failed"
            status = self._set_finished(
                error=str(exc),
                elapsed_seconds=elapsed_offset + time.perf_counter() - started_at,
            )
            self._publish(final_event, status)
            return

        status = self._set_finished(
            elapsed_seconds=elapsed_offset + time.perf_counter() - started_at,
        )
        self._publish(final_event, status)

    def _save_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        runner: Any,
        train_batches: Any,
        run_id: str,
        config: ExperimentConfig,
        opt_params: dict[str, dict[str, float | bool]],
        device: torch.device,
        checkpoint_accuracy: float | None,
    ) -> ExperimentStatus:
        saved_at = utc_now_iso()
        checkpoint_path = self.checkpoint_saver.latest_pt_path(run_id)
        previous_status = self.status()
        best_checkpoint_acc = previous_status.best_checkpoint_acc
        best_checkpoint_step = previous_status.best_checkpoint_step
        best_checkpoint_saved_at = previous_status.best_checkpoint_saved_at
        best_checkpoint_path = (
            Path(previous_status.best_checkpoint_path)
            if previous_status.best_checkpoint_path
            else None
        )
        delete_hidden_best = False

        if checkpoint_accuracy is not None:
            if is_best_checkpoint(previous_status, checkpoint_accuracy):
                best_checkpoint_acc = checkpoint_accuracy
                best_checkpoint_step = previous_status.current_step
                best_checkpoint_saved_at = saved_at
                best_checkpoint_path = checkpoint_path
                delete_hidden_best = True
            else:
                if not self.checkpoint_saver.has_best(run_id):
                    self.checkpoint_saver.promote_latest_to_best(run_id)
                if self.checkpoint_saver.has_best(run_id):
                    best_checkpoint_path = self.checkpoint_saver.best_pt_path(run_id)

        status = self._record_checkpoint(
            saved_at=saved_at,
            checkpoint_path=checkpoint_path,
            checkpoint_accuracy=checkpoint_accuracy,
            best_checkpoint_acc=best_checkpoint_acc,
            best_checkpoint_step=best_checkpoint_step,
            best_checkpoint_saved_at=best_checkpoint_saved_at,
            best_checkpoint_path=best_checkpoint_path,
        )
        optimizer_state = runner_state_dict(runner)
        loader_state = train_batches.state_dict() if config.deterministic else None
        compatibility_warnings = status.checkpoint_warnings
        rng_state = capture_rng_state()
        runtime_manifest = build_runtime_manifest(device)
        self.checkpoint_saver.save(
            model=model,
            status=status,
            config=config,
            optimizer=config.optimizer,
            run_id=run_id,
            optimizer_params=opt_params,
            optimizer_state=optimizer_state,
            loader_state=loader_state,
            rng_state=rng_state,
            runtime_manifest=runtime_manifest,
            saved_at=saved_at,
            compatibility_warnings=compatibility_warnings,
        )
        if delete_hidden_best:
            self.checkpoint_saver.delete_best(run_id)
        return status

    def _update_step(
        self,
        step: int,
        loss: float,
        *,
        elapsed_seconds: float,
        last_iteration_seconds: float,
        loss_mean_since_validation: float,
        loss_stdev_since_validation: float,
        mean_iteration_seconds_since_validation: float,
        mutation_step: float | None,
        train_accuracy: float | None,
        peak_memory_mb: float | None,
    ) -> ExperimentStreamPayload:
        with self.lock:
            self._status.current_step = step
            self._status.current_loss = loss
            self._status.total_elapsed_seconds = max(elapsed_seconds, 0.0)
            self._status.last_iteration_seconds = max(last_iteration_seconds, 0.0)
            self._status.loss_mean_since_validation = loss_mean_since_validation
            self._status.loss_stdev_since_validation = loss_stdev_since_validation
            self._status.mean_iteration_seconds_since_validation = (
                mean_iteration_seconds_since_validation
            )
            self._status.history.loss.append(loss)
            delta = TrainingHistoryDelta(loss=[AccuracyPoint(i=step, value=loss)])
            if train_accuracy is not None:
                train_accuracy_point = AccuracyPoint(i=step, value=train_accuracy)
                self._status.history.train_acc.append(train_accuracy_point)
                delta.train_acc.append(train_accuracy_point)
            if peak_memory_mb is not None:
                memory_point = AccuracyPoint(i=step, value=peak_memory_mb)
                self._status.history.memory_mb.append(memory_point)
                delta.memory_mb.append(memory_point)
            mutation_point = self._record_mutation_step(step, mutation_step)
            if mutation_point is not None:
                delta.mutation_step.append(mutation_point)
            return ExperimentStreamPayload(
                status_patch=self._status.model_dump(exclude={"history"}),
                history_delta=delta,
            )

    def _update_accuracy(
        self,
        step: int,
        accuracy: float,
        *,
        validation_loss: float | None,
        elapsed_seconds: float,
        mutation_step: float | None,
        peak_memory_mb: float | None,
    ) -> ExperimentStreamPayload:
        with self.lock:
            self._status.best_acc = max(self._status.best_acc, accuracy)
            self._status.total_elapsed_seconds = max(elapsed_seconds, 0.0)
            accuracy_point = AccuracyPoint(i=step, value=accuracy)
            self._status.history.acc.append(accuracy_point)
            delta = TrainingHistoryDelta(acc=[accuracy_point])
            if validation_loss is not None:
                validation_loss_point = AccuracyPoint(i=step, value=validation_loss)
                self._status.history.val_loss.append(validation_loss_point)
                delta.val_loss.append(validation_loss_point)
            if peak_memory_mb is not None:
                memory_point = AccuracyPoint(i=step, value=peak_memory_mb)
                self._status.history.memory_mb.append(memory_point)
                delta.memory_mb.append(memory_point)
            mutation_point = self._record_mutation_step(step, mutation_step)
            if mutation_point is not None:
                delta.mutation_step.append(mutation_point)
            return ExperimentStreamPayload(
                status_patch=self._status.model_dump(exclude={"history"}),
                history_delta=delta,
            )

    def _record_mutation_step(
        self,
        step: int,
        mutation_step: float | None,
    ) -> MutationStepPoint | None:
        self._status.current_mutation_step = mutation_step
        if mutation_step is None:
            return None

        point = MutationStepPoint(i=step, value=mutation_step)
        if self._status.history.mutation_step and self._status.history.mutation_step[-1].i == step:
            self._status.history.mutation_step[-1] = point
            return point

        self._status.history.mutation_step.append(point)
        return point

    def _update_runtime(
        self,
        requested_device: str,
        device: torch.device,
        *,
        reproducibility_mode: str,
        checkpoint_warnings: list[str],
    ) -> ExperimentStatus:
        with self.lock:
            self._status.requested_device = requested_device
            self._status.device = device.type
            self._status.device_name = device_name(device)
            self._status.reproducibility_mode = reproducibility_mode
            self._status.checkpoint_warnings = checkpoint_warnings
            return self._status.model_copy(deep=True)

    def _restore_status_for_resume(
        self,
        saved_status: ExperimentStatus,
        *,
        config: ExperimentConfig,
        device: torch.device,
        compatibility_warnings: list[str],
    ) -> ExperimentStatus:
        with self.lock:
            restored = saved_status.model_copy(deep=True)
            restored.is_running = True
            restored.is_paused = False
            restored.pause_requested = False
            restored.error = None
            restored.requested_device = config.device
            restored.device = device.type
            restored.device_name = device_name(device)
            restored.reproducibility_mode = reproducibility_mode(
                config.deterministic,
                compatibility_warnings,
            )
            restored.checkpoint_warnings = compatibility_warnings
            self._status = restored
            return self._status.model_copy(deep=True)

    def _record_checkpoint(
        self,
        *,
        saved_at: str,
        checkpoint_path: Path,
        checkpoint_accuracy: float | None,
        best_checkpoint_acc: float | None,
        best_checkpoint_step: int | None,
        best_checkpoint_saved_at: str | None,
        best_checkpoint_path: Path | None,
    ) -> ExperimentStatus:
        with self.lock:
            self._status.last_checkpoint_step = self._status.current_step
            self._status.last_checkpoint_acc = checkpoint_accuracy
            self._status.last_checkpoint_saved_at = saved_at
            self._status.checkpoint_path = str(checkpoint_path)
            self._status.best_checkpoint_acc = best_checkpoint_acc
            self._status.best_checkpoint_step = best_checkpoint_step
            self._status.best_checkpoint_saved_at = best_checkpoint_saved_at
            self._status.best_checkpoint_path = (
                str(best_checkpoint_path) if best_checkpoint_path is not None else None
            )
            return self._status.model_copy(deep=True)

    def _set_paused(self, *, elapsed_seconds: float | None = None) -> ExperimentStatus:
        with self.lock:
            self._status.is_running = False
            self._status.is_paused = True
            self._status.pause_requested = False
            self._status.error = None
            self._resume_checkpoint_kind = "latest"
            if elapsed_seconds is not None:
                self._status.total_elapsed_seconds = max(elapsed_seconds, 0.0)
            return self._status.model_copy(deep=True)

    def _set_finished(
        self,
        error: str | None = None,
        *,
        elapsed_seconds: float | None = None,
    ) -> ExperimentStatus:
        with self.lock:
            self._status.is_running = False
            self._status.is_paused = False
            self._status.pause_requested = False
            self._status.error = error
            self._resume_checkpoint_kind = "latest"
            if elapsed_seconds is not None:
                self._status.total_elapsed_seconds = max(elapsed_seconds, 0.0)
            return self._status.model_copy(deep=True)

    def _publish(
        self,
        event_type: str,
        payload: ExperimentStatus | ExperimentStreamPayload,
        *,
        replace_history: bool = False,
    ) -> None:
        stream_payload = self._stream_payload(
            payload,
            include_pending_step_delta=event_type != "step",
            replace_history=replace_history,
        )
        with self.lock:
            subscribers = tuple(self.subscribers.items())
        if stream_payload.full_status is None and any(
            not compact for _, compact in subscribers
        ):
            stream_payload = replace(stream_payload, full_status=self.status())
        for queue, _ in subscribers:
            queue.put((event_type, stream_payload))

    def _publish_step(self, payload: ExperimentStreamPayload) -> None:
        now = time.monotonic()
        with self.lock:
            if (
                now - self._last_step_publish_at
                < FRONTEND_STEP_UPDATE_INTERVAL_SECONDS
            ):
                self._pending_step_delta = merge_training_history_deltas(
                    self._pending_step_delta,
                    payload.history_delta,
                )
                return
            history_delta = merge_training_history_deltas(
                self._pending_step_delta,
                payload.history_delta,
            )
            self._pending_step_delta = TrainingHistoryDelta()
            self._last_step_publish_at = now
        self._publish("step", replace(payload, history_delta=history_delta))

    def _stream_payload(
        self,
        payload: ExperimentStatus | ExperimentStreamPayload,
        *,
        include_pending_step_delta: bool,
        replace_history: bool,
    ) -> ExperimentStreamPayload:
        with self.lock:
            pending_delta = (
                self._pending_step_delta
                if include_pending_step_delta
                else TrainingHistoryDelta()
            )
            if include_pending_step_delta:
                self._pending_step_delta = TrainingHistoryDelta()
            needs_full_status = any(not compact for compact in self.subscribers.values())

        if isinstance(payload, ExperimentStatus):
            history_delta = (
                training_history_delta_from_status(payload)
                if replace_history
                else TrainingHistoryDelta()
            )
            history_delta = merge_training_history_deltas(pending_delta, history_delta)
            return stream_payload_from_status(
                payload,
                history_delta=history_delta,
                replace_history=replace_history,
            )

        history_delta = merge_training_history_deltas(
            pending_delta,
            payload.history_delta,
        )
        full_status = payload.full_status
        if needs_full_status and full_status is None:
            full_status = self.status()
        return replace(payload, history_delta=history_delta, full_status=full_status)

    def _is_experiment_running(self) -> bool:
        with self.lock:
            return self._status.is_running


def seed_everything(seed: int, numeric_mode: NumericMode = "strict") -> None:
    if numeric_mode not in NUMERIC_MODES:
        choices = ", ".join(NUMERIC_MODES)
        raise ValueError(f"Unsupported numeric mode {numeric_mode!r}; choose one of {choices}")

    if numeric_mode == "strict":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(numeric_mode == "strict")
    torch.backends.cudnn.benchmark = numeric_mode == "fast"
    torch.backends.cudnn.deterministic = numeric_mode == "strict"
    torch.backends.cuda.matmul.allow_tf32 = numeric_mode == "fast"
    torch.backends.cudnn.allow_tf32 = numeric_mode == "fast"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_id(config: ExperimentConfig, *, started_at: datetime | None = None) -> str:
    started_at = started_at.astimezone() if started_at is not None else datetime.now().astimezone()
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%f")
    return "-".join(
        (
            config.dataset,
            config.optimizer.lower(),
            config.device,
            f"seed{config.seed}",
            timestamp,
        )
    )


def is_best_checkpoint(status: ExperimentStatus, checkpoint_accuracy: float | None) -> bool:
    if checkpoint_accuracy is None:
        return False
    return status.best_checkpoint_acc is None or checkpoint_accuracy > status.best_checkpoint_acc


def current_step_accuracy(status: ExperimentStatus, step: int) -> float | None:
    for point in reversed(status.history.acc):
        if point.i == step:
            return point.value
    return None


def stream_payload_from_status(
    status: ExperimentStatus,
    *,
    history_delta: TrainingHistoryDelta | None = None,
    replace_history: bool = False,
) -> ExperimentStreamPayload:
    return ExperimentStreamPayload(
        status_patch=status.model_dump(exclude={"history"}),
        history_delta=history_delta or TrainingHistoryDelta(),
        replace_history=replace_history,
        full_status=status,
    )


def training_history_delta_from_status(status: ExperimentStatus) -> TrainingHistoryDelta:
    return TrainingHistoryDelta(
        loss=[
            AccuracyPoint(i=index + 1, value=value)
            for index, value in enumerate(status.history.loss)
        ],
        acc=list(status.history.acc),
        train_acc=list(status.history.train_acc),
        val_loss=list(status.history.val_loss),
        memory_mb=list(status.history.memory_mb),
        mutation_step=list(status.history.mutation_step),
    )


def merge_training_history_deltas(
    *deltas: TrainingHistoryDelta,
) -> TrainingHistoryDelta:
    if not deltas:
        return TrainingHistoryDelta()

    return TrainingHistoryDelta(
        loss=merge_accuracy_point_series(*(delta.loss for delta in deltas)),
        acc=merge_accuracy_point_series(*(delta.acc for delta in deltas)),
        train_acc=merge_accuracy_point_series(*(delta.train_acc for delta in deltas)),
        val_loss=merge_accuracy_point_series(*(delta.val_loss for delta in deltas)),
        memory_mb=merge_accuracy_point_series(*(delta.memory_mb for delta in deltas)),
        mutation_step=merge_mutation_step_series(
            *(delta.mutation_step for delta in deltas)
        ),
    )


def merge_accuracy_point_series(
    *series_items: Sequence[AccuracyPoint],
) -> list[AccuracyPoint]:
    points_by_step: dict[int, AccuracyPoint] = {}
    for series in series_items:
        for point in series:
            points_by_step[point.i] = point
    return [points_by_step[step] for step in sorted(points_by_step)]


def merge_mutation_step_series(
    *series_items: Sequence[MutationStepPoint],
) -> list[MutationStepPoint]:
    points_by_step: dict[int, MutationStepPoint] = {}
    for series in series_items:
        for point in series:
            points_by_step[point.i] = point
    return [points_by_step[step] for step in sorted(points_by_step)]


def resolve_device(requested: str) -> torch.device:
    if requested == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "gpu":
        message = NIXOS_CUDA_HINT
        if NIXOS_CUDA_LIBRARY.exists():
            message = f"{message} Found {NIXOS_CUDA_LIBRARY}."
        raise RuntimeError(message)
    return torch.device("cpu")


def resolve_run_device(
    requested: str,
    *,
    allow_cpu_fallback: bool,
) -> tuple[torch.device, list[str]]:
    try:
        return resolve_device(requested), []
    except RuntimeError:
        if requested == "gpu" and allow_cpu_fallback:
            return torch.device("cpu"), ["CUDA is unavailable; resumed checkpoint on CPU."]
        raise


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def current_mutation_step(runner: Any) -> float | None:
    mutation_step = getattr(runner, "mutation_step", None)
    if mutation_step is None:
        return None
    return float(mutation_step)


def interval_training_stats(
    losses: Sequence[float],
    iteration_seconds: Sequence[float],
) -> tuple[float, float, float]:
    if not losses:
        return 0.0, 0.0, 0.0

    loss_mean = sum(losses) / len(losses)
    loss_variance = sum((loss - loss_mean) ** 2 for loss in losses) / len(losses)
    iteration_mean = (
        sum(iteration_seconds) / len(iteration_seconds) if iteration_seconds else 0.0
    )
    return loss_mean, math.sqrt(loss_variance), max(iteration_mean, 0.0)


def runner_state_dict(runner: Any) -> dict[str, Any]:
    state_dict = getattr(runner, "state_dict", None)
    if callable(state_dict):
        return state_dict()
    return {}


def load_runner_state(runner: Any, state: dict[str, Any]) -> None:
    load_state_dict = getattr(runner, "load_state_dict", None)
    if callable(load_state_dict):
        load_state_dict(state)


def move_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    inputs = inputs.to(device, non_blocking=non_blocking)
    targets = targets.to(device, non_blocking=non_blocking)
    return inputs, targets


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = move_batch(inputs, targets, device)
            predictions = model(inputs).argmax(dim=1)
            correct += int((predictions == targets).sum().cpu())
            total += targets.numel()
    return 100.0 * correct / max(total, 1)


def evaluate_mean_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = move_batch(inputs, targets, device)
            logits = model(inputs)
            loss_sum += float(
                torch.nn.functional.cross_entropy(logits, targets, reduction="sum").cpu()
            )
            total += targets.numel()
    model.train(was_training)
    return loss_sum / max(total, 1)


def batch_accuracy(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1)
        correct = int((predictions == targets).sum().cpu())
        total = int(targets.numel())
    model.train(was_training)
    return 100.0 * correct / max(total, 1)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def format_sse(
    event_type: str,
    payload: ExperimentStatus | ExperimentStreamPayload | dict[str, Any],
    *,
    compact: bool = False,
) -> str:
    if isinstance(payload, ExperimentStreamPayload):
        if compact:
            data = ExperimentStatusCompactEvent(
                status_patch=payload.status_patch,
                history_delta=payload.history_delta,
                replace_history=payload.replace_history,
            ).model_dump()
        elif payload.full_status is not None:
            data = payload.full_status.model_dump()
        else:
            raise ValueError("Full status is required for non-compact SSE payloads")
    elif isinstance(payload, ExperimentStatus):
        data = payload.model_dump()
    else:
        data = payload
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
