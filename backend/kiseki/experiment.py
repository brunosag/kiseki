import json
import os
import random
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import (
    CheckpointSaver,
    build_runtime_manifest,
    capture_rng_state,
    compare_runtime_manifests,
    reproducibility_mode,
    restore_rng_state,
    utc_now_iso,
)
from .data import DataLoaderFactory, cycle_loader, deterministic_batch_stream
from .models import CNN2C2DMNIST
from .optimizers import build_optimizer_runner
from .schemas import (
    AccuracyPoint,
    ExperimentConfig,
    ExperimentStatus,
    MutationStepPoint,
    StartExperimentRequest,
)

VAL_FREQ = 10
NumericMode = Literal["strict", "fast"]
NUMERIC_MODES = ("strict", "fast")
NIXOS_CUDA_LIBRARY = Path("/run/opengl-driver/lib/libcuda.so.1")
NIXOS_CUDA_HINT = (
    "CUDA was requested, but torch.cuda.is_available() is false. "
    "From the repository root, run `direnv allow` and restart the API process so shell.nix "
    "exports the CUDA driver path before Python starts."
)


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
        self.subscribers: set[Queue[tuple[str, ExperimentStatus]]] = set()
        self.worker: threading.Thread | None = None
        self._status = ExperimentStatus()

    def status(self) -> ExperimentStatus:
        with self.lock:
            return self._status.model_copy(deep=True)

    def start(self, request: StartExperimentRequest) -> ExperimentStatus:
        run_id = uuid.uuid4().hex
        with self.lock:
            if self._status.is_running:
                raise RuntimeError("An experiment is already running")
            if self._status.is_paused:
                raise RuntimeError("Resume or stop the paused experiment before starting a new one")

            self.stop_event.clear()
            self.pause_event.clear()
            self._status = ExperimentStatus(
                is_running=True,
                optimizer=request.config.optimizer,
                requested_device=request.config.device,
                run_id=run_id,
                reproducibility_mode=reproducibility_mode(request.config.deterministic),
            )
            self.worker = threading.Thread(
                target=self._run,
                args=(request, run_id, None),
                daemon=True,
            )
            self.worker.start()
            status = self._status.model_copy(deep=True)
        self._publish("started", status)
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

    def resume(self) -> ExperimentStatus:
        with self.lock:
            if self._status.is_running:
                raise RuntimeError("An experiment is already running")
            if not self._status.is_paused or self._status.run_id is None:
                raise RuntimeError("No paused experiment can be resumed")

            run_id = self._status.run_id
            self.stop_event.clear()
            self.pause_event.clear()
            self._status.is_running = True
            self._status.is_paused = False
            self._status.pause_requested = False
            self._status.error = None
            self.worker = threading.Thread(target=self._resume_run, args=(run_id,), daemon=True)
            self.worker.start()
            status = self._status.model_copy(deep=True)
        self._publish("resumed", status)
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

    def events(self) -> Iterator[str]:
        queue: Queue[tuple[str, ExperimentStatus]] = Queue()
        with self.lock:
            self.subscribers.add(queue)
            initial_status = self._status.model_copy(deep=True)
        yield format_sse("status", initial_status)
        try:
            while True:
                try:
                    event_type, status = queue.get(timeout=15)
                except Empty:
                    yield ": heartbeat\n\n"
                    continue
                yield format_sse(event_type, status)
        finally:
            with self.lock:
                self.subscribers.discard(queue)

    def _resume_run(self, run_id: str) -> None:
        try:
            checkpoint = self.checkpoint_saver.load_latest(run_id, map_location="cpu")
            config = ExperimentConfig.model_validate(checkpoint["config"])
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
            device, resume_warnings = resolve_run_device(config.device, allow_cpu_fallback=checkpoint is not None)
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
                self._publish("runtime", status)

            if hasattr(self.data_loader_factory, "pin_memory"):
                self.data_loader_factory.pin_memory = device.type == "cuda"
            train_loader, val_loader = self.data_loader_factory.mnist(
                batch_size=config.batch_size,
                seed=config.seed,
            )
            model = CNN2C2DMNIST().to(device)
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
                self._publish("runtime", status)
                first_step = saved_status.current_step + 1

            for step in range(first_step, config.iterations + 1):
                if self.stop_event.is_set():
                    final_event = "stopped"
                    break

                iteration_started_at = time.perf_counter()
                inputs, targets = next(train_batches)
                inputs, targets = move_batch(inputs, targets, device)
                loss = runner.step(inputs, targets)
                iteration_seconds = time.perf_counter() - iteration_started_at
                elapsed_seconds = elapsed_offset + time.perf_counter() - started_at
                status = self._update_step(
                    step,
                    loss,
                    elapsed_seconds=elapsed_seconds,
                    last_iteration_seconds=iteration_seconds,
                    mutation_step=current_mutation_step(runner),
                )
                self._publish("step", status)

                if step % VAL_FREQ == 0:
                    accuracy = evaluate(model, val_loader, device)
                    is_best = accuracy > status.best_acc
                    update_scheduler = getattr(runner, "update_scheduler", None)
                    if callable(update_scheduler):
                        update_scheduler(is_best)
                    status = self._update_accuracy(
                        step,
                        accuracy,
                        elapsed_seconds=elapsed_offset + time.perf_counter() - started_at,
                        mutation_step=current_mutation_step(runner),
                    )
                    self._publish("validation", status)
                    if accuracy >= config.target_acc:
                        final_event = "completed"
                        break

                if config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0:
                    status = self._save_checkpoint(
                        model=model,
                        runner=runner,
                        train_batches=train_batches,
                        run_id=run_id,
                        config=config,
                        opt_params=request.opt_params,
                        device=device,
                    )
                    self._publish("checkpoint", status)

                if self.stop_event.is_set():
                    final_event = "stopped"
                    break

                if self.pause_event.is_set():
                    self._save_checkpoint(
                        model=model,
                        runner=runner,
                        train_batches=train_batches,
                        run_id=run_id,
                        config=config,
                        opt_params=request.opt_params,
                        device=device,
                    )
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
        opt_params: dict[str, dict[str, float]],
        device: torch.device,
    ) -> ExperimentStatus:
        saved_at = utc_now_iso()
        checkpoint_path = self.checkpoint_saver.latest_pt_path(run_id)
        status = self._record_checkpoint(saved_at=saved_at, checkpoint_path=checkpoint_path)
        optimizer_state = runner_state_dict(runner)
        loader_state = train_batches.state_dict() if config.deterministic else None
        compatibility_warnings = status.checkpoint_warnings
        self.checkpoint_saver.save(
            model=model,
            status=status,
            config=config,
            optimizer=config.optimizer,
            run_id=run_id,
            optimizer_params=opt_params,
            optimizer_state=optimizer_state,
            loader_state=loader_state,
            rng_state=capture_rng_state(),
            runtime_manifest=build_runtime_manifest(device),
            saved_at=saved_at,
            compatibility_warnings=compatibility_warnings,
        )
        return status

    def _update_step(
        self,
        step: int,
        loss: float,
        *,
        elapsed_seconds: float,
        last_iteration_seconds: float,
        mutation_step: float | None,
    ) -> ExperimentStatus:
        with self.lock:
            self._status.current_step = step
            self._status.current_loss = loss
            self._status.total_elapsed_seconds = max(elapsed_seconds, 0.0)
            self._status.last_iteration_seconds = max(last_iteration_seconds, 0.0)
            self._status.history.loss.append(loss)
            self._record_mutation_step(step, mutation_step)
            return self._status.model_copy(deep=True)

    def _update_accuracy(
        self,
        step: int,
        accuracy: float,
        *,
        elapsed_seconds: float,
        mutation_step: float | None,
    ) -> ExperimentStatus:
        with self.lock:
            self._status.best_acc = max(self._status.best_acc, accuracy)
            self._status.total_elapsed_seconds = max(elapsed_seconds, 0.0)
            self._status.history.acc.append(AccuracyPoint(i=step, value=accuracy))
            self._record_mutation_step(step, mutation_step)
            return self._status.model_copy(deep=True)

    def _record_mutation_step(self, step: int, mutation_step: float | None) -> None:
        self._status.current_mutation_step = mutation_step
        if mutation_step is None:
            return

        point = MutationStepPoint(i=step, value=mutation_step)
        if self._status.history.mutation_step and self._status.history.mutation_step[-1].i == step:
            self._status.history.mutation_step[-1] = point
            return

        self._status.history.mutation_step.append(point)

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

    def _record_checkpoint(self, *, saved_at: str, checkpoint_path: Path) -> ExperimentStatus:
        with self.lock:
            self._status.last_checkpoint_step = self._status.current_step
            self._status.last_checkpoint_saved_at = saved_at
            self._status.checkpoint_path = str(checkpoint_path)
            return self._status.model_copy(deep=True)

    def _set_paused(self, *, elapsed_seconds: float | None = None) -> ExperimentStatus:
        with self.lock:
            self._status.is_running = False
            self._status.is_paused = True
            self._status.pause_requested = False
            self._status.error = None
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
            if elapsed_seconds is not None:
                self._status.total_elapsed_seconds = max(elapsed_seconds, 0.0)
            return self._status.model_copy(deep=True)

    def _publish(self, event_type: str, status: ExperimentStatus) -> None:
        with self.lock:
            subscribers = tuple(self.subscribers)
        for queue in subscribers:
            queue.put((event_type, status))


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


def format_sse(event_type: str, payload: ExperimentStatus | dict[str, Any]) -> str:
    if isinstance(payload, ExperimentStatus):
        data = payload.model_dump()
    else:
        data = payload
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
