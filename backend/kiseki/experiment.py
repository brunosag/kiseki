import json
import os
import random
import threading
from collections.abc import Iterator
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import CheckpointSaver
from .data import DataLoaderFactory, cycle_loader
from .models import CNN2C2DMNIST
from .optimizers import build_optimizer_runner
from .schemas import AccuracyPoint, ExperimentStatus, StartExperimentRequest

VAL_FREQ = 10
SAVE_FREQ = 50
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
        self.lock = threading.Lock()
        self.subscribers: set[Queue[tuple[str, ExperimentStatus]]] = set()
        self.worker: threading.Thread | None = None
        self._status = ExperimentStatus()

    def status(self) -> ExperimentStatus:
        with self.lock:
            return self._status.model_copy(deep=True)

    def start(self, request: StartExperimentRequest) -> ExperimentStatus:
        with self.lock:
            if self._status.is_running:
                raise RuntimeError("An experiment is already running")
            self.stop_event.clear()
            self._status = ExperimentStatus(
                is_running=True,
                requested_device=request.config.device,
            )
            self.worker = threading.Thread(target=self._run, args=(request,), daemon=True)
            self.worker.start()
            status = self._status.model_copy(deep=True)
        self._publish("started", status)
        return status

    def stop(self) -> ExperimentStatus:
        self.stop_event.set()
        return self.status()

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

    def _run(self, request: StartExperimentRequest) -> None:
        final_event = "completed"
        try:
            config = request.config
            seed_everything(config.seed)
            device = resolve_device(config.device)
            status = self._update_runtime(config.device, device)
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
            train_batches = cycle_loader(train_loader)

            for step in range(1, config.iterations + 1):
                if self.stop_event.is_set():
                    final_event = "stopped"
                    break

                inputs, targets = next(train_batches)
                inputs, targets = move_batch(inputs, targets, device)
                loss = runner.step(inputs, targets)
                status = self._update_step(step, loss)
                self._publish("step", status)

                if step % VAL_FREQ == 0:
                    accuracy = evaluate(model, val_loader, device)
                    status = self._update_accuracy(step, accuracy)
                    self._publish("validation", status)
                    if accuracy >= config.target_acc:
                        final_event = "completed"
                        break

                if step % SAVE_FREQ == 0:
                    self.checkpoint_saver.save(
                        model=model,
                        status=self.status(),
                        config=config,
                        optimizer=config.optimizer,
                    )

        except Exception as exc:  # pragma: no cover - surfaced through API status.
            final_event = "failed"
            status = self._set_finished(error=str(exc))
            self._publish(final_event, status)
            return

        status = self._set_finished()
        self._publish(final_event, status)

    def _update_step(self, step: int, loss: float) -> ExperimentStatus:
        with self.lock:
            self._status.current_step = step
            self._status.current_loss = loss
            self._status.history.loss.append(loss)
            return self._status.model_copy(deep=True)

    def _update_accuracy(self, step: int, accuracy: float) -> ExperimentStatus:
        with self.lock:
            self._status.best_acc = max(self._status.best_acc, accuracy)
            self._status.history.acc.append(AccuracyPoint(i=step, value=accuracy))
            return self._status.model_copy(deep=True)

    def _update_runtime(
        self,
        requested_device: str,
        device: torch.device,
    ) -> ExperimentStatus:
        with self.lock:
            self._status.requested_device = requested_device
            self._status.device = device.type
            self._status.device_name = device_name(device)
            return self._status.model_copy(deep=True)

    def _set_finished(self, error: str | None = None) -> ExperimentStatus:
        with self.lock:
            self._status.is_running = False
            self._status.error = error
            return self._status.model_copy(deep=True)

    def _publish(self, event_type: str, status: ExperimentStatus) -> None:
        with self.lock:
            subscribers = tuple(self.subscribers)
        for queue in subscribers:
            queue.put((event_type, status))


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


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
