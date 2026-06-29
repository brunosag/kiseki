from __future__ import annotations

import math
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
    COSYNE_P_M,
    CONFIG_SCHEMA,
    ETA,
    ETA_0,
    ETA_SBX,
    GAMMA,
    LAMBDA,
    NUM_CHILDREN,
    OPTIMIZERS_SCHEMA,
    PERMUTE_ALL,
    P_M,
    RHO,
    RHO_E,
    RHO_X,
    SIGMA_M,
    TAU_PAT,
    TOURNAMENT_SIZE,
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

METRIC_SEPARATOR = "    "
UNICODE_OPTIMIZER_LABELS = {
    r"\eta": "η",
    r"\eta_0": "η₀",
    r"p_{\mathrm{m}}": "pₘ",
    r"\gamma": "γ",
    r"\rho": "ρ",
    r"\rho_{\mathrm{x}}": "ρₓ",
    r"\lambda": "λ",
    r"\tau_{\mathrm{pat}}": "τₚₐₜ",
    r"\sigma_{\mathrm{m}}": "σₘ",
    r"\rho_{\mathrm{e}}": "ρₑ",
    r"\eta_{\mathrm{SBX}}": "η_SBX",
    r"\lambda_{\mathrm{c}}": "λ_c",
    r"\pi_{\mathrm{all}}": "π_all",
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
    "mutation_stdev",
    "mutation_decay",
    "retention_fraction",
    "crossover_fraction",
    "fitness_decay",
    "validation_patience",
    "tournament_size",
    "permute_all",
    "elitism_ratio",
    "sbx_eta",
    "num_children",
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
    mutation_stdev: float | None = None
    mutation_decay: float | None = None
    retention_fraction: float | None = None
    crossover_fraction: float | None = None
    fitness_decay: float | None = None
    validation_patience: int | None = None
    tournament_size: int | None = None
    permute_all: bool | None = None
    elitism_ratio: float | None = None
    sbx_eta: float | None = None
    num_children: int | None = None
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
        mutation_stdev=args.mutation_stdev,
        mutation_decay=args.mutation_decay,
        retention_fraction=args.retention_fraction,
        crossover_fraction=args.crossover_fraction,
        fitness_decay=args.fitness_decay,
        validation_patience=args.validation_patience,
        tournament_size=args.tournament_size,
        permute_all=args.permute_all,
        elitism_ratio=args.elitism_ratio,
        sbx_eta=args.sbx_eta,
        num_children=args.num_children,
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


def optimizer_params_for(optimizer: str, options: TrainOptions) -> dict[str, float | bool]:
    if optimizer == "SGD":
        evolutionary_fields = (
            "population_size",
            "mutation_probability",
            "mutation_step",
            "mutation_stdev",
            "mutation_decay",
            "retention_fraction",
            "crossover_fraction",
            "fitness_decay",
            "validation_patience",
            "tournament_size",
            "permute_all",
            "elitism_ratio",
            "sbx_eta",
            "num_children",
        )
        reject_optimizer_fields(evolutionary_fields, options, optimizer)
        return {
            ETA: float(value_or_default(options.learning_rate, DEFAULT_OPTIMIZER_PARAMS["SGD"][ETA]))
        }

    if optimizer == "LEEA":
        reject_optimizer_fields(
            (
                "learning_rate",
                "mutation_stdev",
                "tournament_size",
                "permute_all",
                "elitism_ratio",
                "sbx_eta",
                "num_children",
            ),
            options,
            optimizer,
        )
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

    if optimizer == "CoSyNE":
        reject_optimizer_fields(
            (
                "learning_rate",
                "mutation_step",
                "mutation_decay",
                "retention_fraction",
                "crossover_fraction",
                "fitness_decay",
                "validation_patience",
            ),
            options,
            optimizer,
        )
        defaults = DEFAULT_OPTIMIZER_PARAMS["CoSyNE"]
        return {
            "N": float(value_or_default(options.population_size, defaults["N"])),
            TOURNAMENT_SIZE: float(
                value_or_default(options.tournament_size, defaults[TOURNAMENT_SIZE])
            ),
            SIGMA_M: float(value_or_default(options.mutation_stdev, defaults[SIGMA_M])),
            COSYNE_P_M: float(
                value_or_default(options.mutation_probability, defaults[COSYNE_P_M])
            ),
            PERMUTE_ALL: bool(value_or_default(options.permute_all, defaults[PERMUTE_ALL])),
            RHO_E: float(value_or_default(options.elitism_ratio, defaults[RHO_E])),
            ETA_SBX: float(value_or_default(options.sbx_eta, defaults[ETA_SBX])),
            NUM_CHILDREN: float(value_or_default(options.num_children, defaults[NUM_CHILDREN])),
        }

    raise TrainError(f"Unsupported optimizer: {optimizer}")


def reject_optimizer_fields(fields: Sequence[str], options: TrainOptions, optimizer: str) -> None:
    rejected = [field for field in fields if getattr(options, field) is not None]
    if not rejected:
        return
    rejected_text = ", ".join(f"--{field.replace('_', '-')}" for field in rejected)
    raise TrainError(f"{rejected_text} cannot be used with --optimizer {optimizer}.")


def value_or_default(
    value: int | float | bool | None,
    default: int | float | bool,
) -> int | float | bool:
    return default if value is None else value


def wait_for_completion(
    manager: ExperimentManager,
    *,
    initial_status: ExperimentStatus,
    log_every: int,
    poll_interval: float,
    stream: TextIO,
) -> ExperimentStatus:
    status = initial_status
    last_step = 0 if status.current_step == 0 else -1

    while status.is_running:
        if should_print_status(status, last_step, log_every):
            print(format_status(status), file=stream, flush=True)
            last_step = status.current_step
        time.sleep(max(poll_interval, 0.0))
        status = manager.status()

    if should_print_status(status, last_step, log_every) or status.current_step:
        print(format_status(status), file=stream, flush=True)
    return status


def should_print_status(
    status: ExperimentStatus,
    last_step: int,
    log_every: int,
) -> bool:
    if status.current_step == last_step:
        return False
    if status.current_step == 0:
        return True
    return status.current_step % log_every == 0


def print_start(status: ExperimentStatus, options: TrainOptions, *, stream: TextIO) -> None:
    print(format_start_summary(status, options), file=stream, flush=True)


def format_start_summary(status: ExperimentStatus, options: TrainOptions) -> str:
    if options.resume:
        return format_resume_summary(status, options)

    request = build_start_request(options)
    config = request.config
    optimizer = config.optimizer
    optimizer_params = request.opt_params.get(optimizer, {})
    lines = [
        "Started training",
        *format_aligned_rows([("Run ID", status.run_id or "—")], indent="  "),
        "",
        "Configuration",
        *format_aligned_rows(
            [
                (CONFIG_SCHEMA["dataset"].label, config.dataset),
                (CONFIG_SCHEMA["optimizer"].label, config.optimizer),
                ("Requested device", config.device),
                (CONFIG_SCHEMA["seed"].label, config.seed),
                (CONFIG_SCHEMA["batch_size"].label, config.batch_size),
                (CONFIG_SCHEMA["iterations"].label, config.iterations),
                (CONFIG_SCHEMA["target_acc"].label, format_percent(config.target_acc)),
                (CONFIG_SCHEMA["deterministic"].label, config.deterministic),
                (
                    CONFIG_SCHEMA["checkpoint_interval"].label,
                    config.checkpoint_interval,
                ),
            ],
            indent="  ",
        ),
        "",
        f"{optimizer} parameters",
        *format_optimizer_rows(optimizer, optimizer_params),
    ]
    return "\n".join(lines)


def format_resume_summary(status: ExperimentStatus, options: TrainOptions) -> str:
    rows: list[tuple[str, Any]] = [
        ("Run ID", status.run_id or options.resume or "—"),
    ]
    if status.optimizer is not None:
        rows.append(("Optimizer", status.optimizer))
    rows.append(("Requested device", status.requested_device))

    override_rows = [
        (CONFIG_SCHEMA["iterations"].label, options.iterations),
        (CONFIG_SCHEMA["target_acc"].label, format_percent(options.target_acc))
        if options.target_acc is not None
        else (CONFIG_SCHEMA["target_acc"].label, None),
        (CONFIG_SCHEMA["checkpoint_interval"].label, options.checkpoint_interval),
    ]
    override_rows = [(label, value) for label, value in override_rows if value is not None]

    lines = [
        "Resumed training",
        *format_aligned_rows(rows, indent="  "),
    ]
    if override_rows:
        lines.extend(["", "Resume controls", *format_aligned_rows(override_rows, indent="  ")])
    return "\n".join(lines)


def format_optimizer_rows(
    optimizer: str,
    optimizer_params: dict[str, float | bool],
) -> list[str]:
    fields = OPTIMIZERS_SCHEMA.get(optimizer, [])
    rows = [
        (
            unicode_optimizer_label(field.label),
            format_optimizer_value(optimizer_params.get(field.key, field.default), field.step),
            field.desc,
        )
        for field in fields
    ]
    if not rows:
        return ["  none"]

    label_width = max(len(label) for label, _, _ in rows)
    value_width = max(len(value) for _, value, _ in rows)
    return [
        f"  {label:<{label_width}}  {value:<{value_width}}  {desc}"
        for label, value, desc in rows
    ]


def unicode_optimizer_label(label: str) -> str:
    return UNICODE_OPTIMIZER_LABELS.get(label, label)


def format_aligned_rows(rows: Sequence[tuple[str, Any]], *, indent: str) -> list[str]:
    label_width = max(len(label) for label, _ in rows)
    return [
        f"{indent}{label:<{label_width}}  {format_display_value(value)}"
        for label, value in rows
    ]


def format_display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return format_integer(value)
    if isinstance(value, float):
        return format_number(value)
    return str(value)


def format_optimizer_value(value: float | bool, step: float | None) -> str:
    if isinstance(value, bool):
        return format_display_value(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        if step is None or step >= 1:
            return format_integer(int(value))
    return format_number(value)


def print_final(status: ExperimentStatus, *, stream: TextIO) -> None:
    if status.error:
        print(f"Failed run_id={status.run_id}: {status.error}", file=stream, flush=True)
        return
    if status.is_paused:
        metrics = METRIC_SEPARATOR.join(
            [
                f"i={format_integer(status.current_step)}",
                f"t={format_duration(status.total_elapsed_seconds)}",
            ]
        )
        print(
            f"Paused run_id={status.run_id} {metrics}",
            file=stream,
            flush=True,
        )
        return
    metrics = METRIC_SEPARATOR.join(
        [
            f"i={format_integer(status.current_step)}",
            *format_best_accuracy_metrics(status),
            f"t={format_duration(status.total_elapsed_seconds)}",
        ]
    )
    print(
        f"Finished run_id={status.run_id} {metrics}",
        file=stream,
        flush=True,
    )


def format_status(status: ExperimentStatus) -> str:
    parts = [
        f"i={format_integer(status.current_step)}",
        (
            f"ℓ={format_loss(status.loss_mean_since_validation)}"
            f" ± {format_loss(status.loss_stdev_since_validation)}"
        ),
        *format_best_accuracy_metrics(status),
        f"t={format_duration(status.total_elapsed_seconds)}",
        f"Δt̄={format_seconds(status.mean_iteration_seconds_since_validation)}",
    ]
    if status.current_mutation_step is not None:
        parts.append(f"η={format_loss(status.current_mutation_step)}")
    return METRIC_SEPARATOR.join(parts)


def format_best_accuracy_metrics(status: ExperimentStatus) -> list[str]:
    metric = f"a*={format_percent(status.best_acc)}"
    step = best_accuracy_step(status)
    if step is not None:
        metric = f"{metric} ({format_integer(step)})"
    return [metric]


def best_accuracy_step(status: ExperimentStatus) -> int | None:
    if math.isfinite(status.best_acc):
        for point in status.history.acc:
            if math.isclose(point.value, status.best_acc, rel_tol=1e-12, abs_tol=1e-12):
                return point.i
    return status.best_checkpoint_step


def format_integer(value: int) -> str:
    return f"{value:,.0f}"


def format_percent(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:.2f}%"


def format_loss(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:.4f}"


def format_number(value: int | float) -> str:
    if not math.isfinite(value):
        return "—"
    if isinstance(value, int) or value.is_integer():
        return format_integer(int(value))
    return f"{value:.6g}"


def format_seconds(seconds: float) -> str:
    safe_seconds = seconds if math.isfinite(seconds) and seconds > 0 else 0.0
    if safe_seconds < 1:
        milliseconds = math.floor(safe_seconds * 1000 + 0.5)
        return f"{milliseconds}ms"
    return f"{safe_seconds:.3g}s"


def format_duration(seconds: float) -> str:
    safe_seconds = seconds if math.isfinite(seconds) and seconds > 0 else 0.0
    total_seconds = math.floor(safe_seconds + 0.5)

    if total_seconds < 60:
        return f"{total_seconds:02d}s"

    hours = total_seconds // 3600
    if hours > 0:
        minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60
        return f"{hours}h {minutes:02d}m {remaining_seconds:02d}s"

    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60
    return f"{minutes:02d}m {remaining_seconds:02d}s"


def install_signal_handlers(handler: TrainingSignalHandler) -> dict[signal.Signals, Any]:
    previous_handlers = {}
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, handler)
    return previous_handlers


def restore_signal_handlers(previous_handlers: dict[signal.Signals, Any]) -> None:
    for signal_number, handler in previous_handlers.items():
        signal.signal(signal_number, handler)
