from __future__ import annotations

import signal
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TextIO

from pydantic import ValidationError

from .checkpoint import CheckpointNotFoundError
from .dataset_types import DatasetName
from .experiment import ExperimentManager
from .schemas import (
    ETA,
    ETA_0,
    GAMMA,
    LAMBDA,
    OPTIMIZERS_SCHEMA,
    P_M,
    RHO,
    RHO_X,
    TAU_PAT,
    CheckpointSelection,
    ExperimentConfig,
    ExperimentControlsUpdate,
    ExperimentStatus,
    StartExperimentRequest,
)

DEFAULT_OPTIMIZER_PARAMS = {
    optimizer: {field.key: field.default for field in fields}
    for optimizer, fields in OPTIMIZERS_SCHEMA.items()
}

RESUME_ONLY_FIELDS = ("iterations", "target_acc", "checkpoint_interval")
RESUME_BLOCKED_FIELDS = (
    "dataset",
    "device",
    "seed",
    "batch_size",
    "optimizer",
    "deterministic",
    "learning_rate",
    "population_size",
    "mutation_probability",
    "mutation_step",
    "mutation_decay",
    "retention_fraction",
    "crossover_fraction",
    "fitness_decay",
    "validation_patience",
)


class TrainError(RuntimeError):
    pass


@dataclass(slots=True)
class TrainOptions:
    dataset: DatasetName | None = None
    device: str | None = None
    seed: int | None = None
    batch_size: int | None = None
    iterations: int | None = None
    target_acc: float | None = None
    optimizer: str | None = None
    deterministic: bool | None = None
    checkpoint_interval: int | None = None
    log_every: int = 10
    learning_rate: float | None = None
    population_size: int | None = None
    mutation_probability: float | None = None
    mutation_step: float | None = None
    mutation_decay: float | None = None
    retention_fraction: float | None = None
    crossover_fraction: float | None = None
    fitness_decay: float | None = None
    validation_patience: int | None = None
    resume: str | None = None


class TrainingSignalHandler:
    def __init__(self, manager: ExperimentManager, *, stream: TextIO) -> None:
        self.manager = manager
        self.stream = stream
        self.signal_count = 0
        self.interrupted = False

    def __call__(self, signum: int, frame: Any) -> None:
        del frame
        self.signal_count += 1
        self.interrupted = True
        signal_name = signal.Signals(signum).name
        if self.signal_count == 1:
            print(
                f"\n{signal_name} received; pausing after the current step to save a checkpoint.",
                file=self.stream,
                flush=True,
            )
            try:
                self.manager.pause()
            except RuntimeError as exc:
                print(f"Pause request was not accepted: {exc}", file=self.stream, flush=True)
            return

        print(
            f"\n{signal_name} received again; stopping without an extra checkpoint.",
            file=self.stream,
            flush=True,
        )
        try:
            self.manager.stop()
        except RuntimeError as exc:
            print(f"Stop request was not accepted: {exc}", file=self.stream, flush=True)


def train_options_from_args(args: Any) -> TrainOptions:
    return TrainOptions(
        dataset=args.dataset,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        iterations=args.iterations,
        target_acc=args.target_acc,
        optimizer=args.optimizer,
        deterministic=args.deterministic,
        checkpoint_interval=args.checkpoint_interval,
        log_every=args.log_every,
        learning_rate=args.learning_rate,
        population_size=args.population_size,
        mutation_probability=args.mutation_probability,
        mutation_step=args.mutation_step,
        mutation_decay=args.mutation_decay,
        retention_fraction=args.retention_fraction,
        crossover_fraction=args.crossover_fraction,
        fitness_decay=args.fitness_decay,
        validation_patience=args.validation_patience,
        resume=args.resume,
    )


def run_training(
    options: TrainOptions,
    *,
    manager: ExperimentManager | None = None,
    stream: TextIO | None = None,
    error_stream: TextIO | None = None,
    poll_interval: float = 1.0,
    enable_signal_handlers: bool = True,
) -> int:
    stream = stream or sys.stdout
    error_stream = error_stream or sys.stderr
    manager = manager or ExperimentManager()
    signal_handler = TrainingSignalHandler(manager, stream=error_stream)
    previous_handlers: dict[signal.Signals, Any] = {}

    try:
        if enable_signal_handlers:
            previous_handlers = install_signal_handlers(signal_handler)

        status = start_or_resume(manager, options)
        print_start(status, options, stream=stream)
        status = wait_for_completion(
            manager,
            initial_status=status,
            log_every=options.log_every,
            poll_interval=poll_interval,
            stream=stream,
        )
    except (CheckpointNotFoundError, RuntimeError, ValidationError, TrainError, ValueError) as exc:
        print(str(exc), file=error_stream)
        return 1
    finally:
        if previous_handlers:
            restore_signal_handlers(previous_handlers)

    print_final(status, stream=stream)
    if status.error:
        return 1
    if signal_handler.interrupted:
        return 130
    return 0


def start_or_resume(manager: ExperimentManager, options: TrainOptions) -> ExperimentStatus:
    if options.resume is not None:
        update = build_resume_update(options)
        manager.load_checkpoint(CheckpointSelection(run_id=options.resume, kind="latest"))
        return manager.resume(update)

    request = build_start_request(options)
    return manager.start(request)


def build_start_request(options: TrainOptions) -> StartExperimentRequest:
    config_values: dict[str, Any] = {}
    for field in (
        "dataset",
        "device",
        "seed",
        "batch_size",
        "iterations",
        "target_acc",
        "optimizer",
        "deterministic",
        "checkpoint_interval",
    ):
        value = getattr(options, field)
        if value is not None:
            config_values[field] = value

    config = ExperimentConfig(**config_values)
    opt_params = {config.optimizer: optimizer_params_for(config.optimizer, options)}
    return StartExperimentRequest(config=config, opt_params=opt_params)


def build_resume_update(options: TrainOptions) -> ExperimentControlsUpdate:
    blocked = [field for field in RESUME_BLOCKED_FIELDS if getattr(options, field) is not None]
    if blocked:
        blocked_text = ", ".join(f"--{field.replace('_', '-')}" for field in blocked)
        allowed_text = ", ".join(f"--{field.replace('_', '-')}" for field in RESUME_ONLY_FIELDS)
        raise TrainError(
            f"Cannot use {blocked_text} with --resume. Resume overrides are limited to "
            f"{allowed_text}."
        )

    return ExperimentControlsUpdate(
        iterations=options.iterations,
        target_acc=options.target_acc,
        checkpoint_interval=options.checkpoint_interval,
    )


def optimizer_params_for(optimizer: str, options: TrainOptions) -> dict[str, float]:
    if optimizer == "SGD":
        leea_fields = (
            "population_size",
            "mutation_probability",
            "mutation_step",
            "mutation_decay",
            "retention_fraction",
            "crossover_fraction",
            "fitness_decay",
            "validation_patience",
        )
        reject_optimizer_fields(leea_fields, options, optimizer)
        return {
            ETA: float(value_or_default(options.learning_rate, DEFAULT_OPTIMIZER_PARAMS["SGD"][ETA]))
        }

    if optimizer == "LEEA":
        reject_optimizer_fields(("learning_rate",), options, optimizer)
        defaults = DEFAULT_OPTIMIZER_PARAMS["LEEA"]
        return {
            "N": float(value_or_default(options.population_size, defaults["N"])),
            P_M: float(value_or_default(options.mutation_probability, defaults[P_M])),
            ETA_0: float(value_or_default(options.mutation_step, defaults[ETA_0])),
            GAMMA: float(value_or_default(options.mutation_decay, defaults[GAMMA])),
            RHO: float(value_or_default(options.retention_fraction, defaults[RHO])),
            RHO_X: float(value_or_default(options.crossover_fraction, defaults[RHO_X])),
            LAMBDA: float(value_or_default(options.fitness_decay, defaults[LAMBDA])),
            TAU_PAT: float(value_or_default(options.validation_patience, defaults[TAU_PAT])),
        }

    raise TrainError(f"Unsupported optimizer: {optimizer}")


def reject_optimizer_fields(fields: Sequence[str], options: TrainOptions, optimizer: str) -> None:
    rejected = [field for field in fields if getattr(options, field) is not None]
    if not rejected:
        return
    rejected_text = ", ".join(f"--{field.replace('_', '-')}" for field in rejected)
    raise TrainError(f"{rejected_text} cannot be used with --optimizer {optimizer}.")


def value_or_default(value: int | float | None, default: int | float) -> int | float:
    return default if value is None else value


def wait_for_completion(
    manager: ExperimentManager,
    *,
    initial_status: ExperimentStatus,
    log_every: int,
    poll_interval: float,
    stream: TextIO,
) -> ExperimentStatus:
    last_step = -1
    last_checkpoint_path: str | None = None
    status = initial_status

    while status.is_running:
        if should_print_status(status, last_step, last_checkpoint_path, log_every):
            print(format_status(status), file=stream, flush=True)
            last_step = status.current_step
            last_checkpoint_path = status.checkpoint_path
        time.sleep(max(poll_interval, 0.0))
        status = manager.status()

    if should_print_status(status, last_step, last_checkpoint_path, log_every) or status.current_step:
        print(format_status(status), file=stream, flush=True)
    return status


def should_print_status(
    status: ExperimentStatus,
    last_step: int,
    last_checkpoint_path: str | None,
    log_every: int,
) -> bool:
    if status.checkpoint_path and status.checkpoint_path != last_checkpoint_path:
        return True
    if status.current_step == last_step:
        return False
    if status.current_step == 0:
        return True
    return status.current_step % log_every == 0


def print_start(status: ExperimentStatus, options: TrainOptions, *, stream: TextIO) -> None:
    action = "Resumed" if options.resume else "Started"
    print(
        (
            f"{action} run_id={status.run_id} optimizer={status.optimizer} "
            f"requested_device={status.requested_device}"
        ),
        file=stream,
        flush=True,
    )


def print_final(status: ExperimentStatus, *, stream: TextIO) -> None:
    if status.error:
        print(f"Failed run_id={status.run_id}: {status.error}", file=stream, flush=True)
        return
    if status.is_paused:
        print(
            f"Paused run_id={status.run_id} checkpoint={status.checkpoint_path}",
            file=stream,
            flush=True,
        )
        return
    print(
        (
            f"Finished run_id={status.run_id} step={status.current_step} "
            f"best_acc={status.best_acc:.2f}% checkpoint={status.checkpoint_path}"
        ),
        file=stream,
        flush=True,
    )


def format_status(status: ExperimentStatus) -> str:
    parts = [
        f"step={status.current_step}",
        f"loss={status.current_loss:.6f}",
        f"best_acc={status.best_acc:.2f}%",
        f"elapsed={status.total_elapsed_seconds:.1f}s",
    ]
    if status.current_mutation_step is not None:
        parts.append(f"mutation_step={status.current_mutation_step:.6f}")
    if status.last_checkpoint_step is not None:
        parts.append(f"last_checkpoint_step={status.last_checkpoint_step}")
    if status.checkpoint_path:
        parts.append(f"checkpoint={status.checkpoint_path}")
    return " ".join(parts)


def install_signal_handlers(handler: TrainingSignalHandler) -> dict[signal.Signals, Any]:
    previous_handlers = {}
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, handler)
    return previous_handlers


def restore_signal_handlers(previous_handlers: dict[signal.Signals, Any]) -> None:
    for signal_number, handler in previous_handlers.items():
        signal.signal(signal_number, handler)
