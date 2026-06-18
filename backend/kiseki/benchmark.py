import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import cycle_loader, load_mnist
from .experiment import NumericMode, move_batch, seed_everything
from .models import CNN2C2DMNIST
from .optimizers import build_optimizer_runner
from .schemas import ETA, ETA_0, OPTIMIZERS_SCHEMA, P_M, ExperimentConfig

BenchmarkName = Literal["synthetic", "mnist"]
OptimizerName = Literal["LEEA", "SGD"]
RequestedDevice = Literal["auto", "both", "cpu", "gpu"]

FRONTEND_DEFAULT_CONFIG = ExperimentConfig()
FRONTEND_DEFAULT_OPTIMIZER_PARAMS = {
    optimizer: {field.key: field.default for field in fields}
    for optimizer, fields in OPTIMIZERS_SCHEMA.items()
}

NIXOS_CUDA_LIBRARY = Path("/run/opengl-driver/lib/libcuda.so.1")
NIXOS_CUDA_PREFIX = (
    "direnv allow\n"
    "uv run kiseki benchmark --device gpu"
)


class BenchmarkError(RuntimeError):
    pass


@dataclass(slots=True)
class BenchmarkOptions:
    device: RequestedDevice = "both"
    optimizer: Literal["LEEA", "SGD", "both"] = "both"
    benchmark: Literal["synthetic", "mnist", "both"] = FRONTEND_DEFAULT_CONFIG.dataset
    iterations: int = 10
    batch_size: int = FRONTEND_DEFAULT_CONFIG.batch_size
    seed: int = FRONTEND_DEFAULT_CONFIG.seed
    num_workers: int = 0
    population_size: int = int(FRONTEND_DEFAULT_OPTIMIZER_PARAMS["LEEA"]["N"])
    mutation_probability: float = FRONTEND_DEFAULT_OPTIMIZER_PARAMS["LEEA"][P_M]
    mutation_step: float = FRONTEND_DEFAULT_OPTIMIZER_PARAMS["LEEA"][ETA_0]
    leea_chunk_size: int | None = None
    leea_profile: bool = False
    numeric_mode: NumericMode = "strict"
    output: Path | None = None


def run_benchmarks(options: BenchmarkOptions) -> list[dict[str, Any]]:
    devices = resolve_benchmark_devices(options.device)
    optimizers = expand_optimizer(options.optimizer)
    benchmarks = expand_benchmark(options.benchmark)

    results = []
    for device in devices:
        for benchmark_name in benchmarks:
            for optimizer_name in optimizers:
                results.append(run_single_benchmark(benchmark_name, optimizer_name, device, options))
    return results


def run_single_benchmark(
    benchmark_name: BenchmarkName,
    optimizer_name: OptimizerName,
    device: torch.device,
    options: BenchmarkOptions,
) -> dict[str, Any]:
    seed_everything(options.seed, numeric_mode=options.numeric_mode)
    pin_memory = device.type == "cuda"
    train_loader = build_benchmark_loader(benchmark_name, options, pin_memory=pin_memory)
    model = CNN2C2DMNIST().to(device)
    should_profile_leea = options.leea_profile and optimizer_name == "LEEA"
    runner = build_optimizer_runner(
        optimizer_name,
        model,
        raw_optimizer_params(optimizer_name, options),
        device=device,
        seed=options.seed,
        leea_evaluation_chunk_size=options.leea_chunk_size,
        leea_profile=should_profile_leea,
    )
    train_batches = cycle_loader(train_loader)

    if should_profile_leea and device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    final_loss = float("nan")
    iteration_seconds = []
    leea_profile_steps: list[dict[str, float | int]] = []
    for iteration in range(1, options.iterations + 1):
        iteration_start = time.perf_counter()
        inputs, targets = next(train_batches)
        inputs, targets = move_batch(inputs, targets, device)
        final_loss = runner.step(inputs, targets)
        if should_profile_leea and device.type == "cuda":
            torch.cuda.synchronize(device)
        iteration_seconds.append(max(time.perf_counter() - iteration_start, 0.0))
        profile = getattr(runner, "last_profile", None)
        if should_profile_leea and profile is not None:
            leea_profile_steps.append({"iteration": iteration, **profile})
    if should_profile_leea and device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-12)
    first_iteration_seconds = iteration_seconds[0] if iteration_seconds else float("nan")
    steady_state_seconds = iteration_seconds[1:] or iteration_seconds
    steady_state_median_seconds = statistics.median(steady_state_seconds)

    result = {
        "benchmark": benchmark_name,
        "optimizer": optimizer_name,
        "status": "ok",
        "requested_device": options.device,
        "device": device.type,
        "device_name": device_name(device),
        "iterations": options.iterations,
        "batch_size": options.batch_size,
        "num_workers": options.num_workers,
        "seed": options.seed,
        "numeric_mode": options.numeric_mode,
        "numeric_mode_trajectory_changing": options.numeric_mode != "strict",
        "duration_seconds": elapsed,
        "iterations_per_second": options.iterations / elapsed,
        "average_iteration_seconds": elapsed / options.iterations,
        "median_iteration_seconds": statistics.median(iteration_seconds),
        "first_iteration_seconds": first_iteration_seconds,
        "steady_state_median_seconds": steady_state_median_seconds,
        "final_loss": final_loss,
    }
    if optimizer_name == "LEEA":
        result.update(
            {
                "population_size": options.population_size,
                "mutation_probability": options.mutation_probability,
                "mutation_step": options.mutation_step,
                "leea_chunk_size": getattr(runner, "evaluation_chunk_size", options.leea_chunk_size),
                "leea_profile": should_profile_leea,
            }
        )
        if should_profile_leea:
            result["leea_profile_steps"] = leea_profile_steps
            result["leea_profile_summary"] = summarize_leea_profile(leea_profile_steps)
    return result


def build_benchmark_loader(
    benchmark_name: BenchmarkName,
    options: BenchmarkOptions,
    *,
    pin_memory: bool,
) -> DataLoader:
    if benchmark_name == "synthetic":
        return synthetic_loader(options, pin_memory=pin_memory)
    if benchmark_name == "mnist":
        try:
            train_loader, _ = load_mnist(
                Path("data"),
                batch_size=options.batch_size,
                seed=options.seed,
                download=True,
                num_workers=options.num_workers,
                pin_memory=pin_memory,
            )
        except RuntimeError as exc:
            raise BenchmarkError(mnist_unavailable_message(exc)) from exc
        return train_loader
    raise BenchmarkError(f"Unsupported benchmark: {benchmark_name}")


def synthetic_loader(options: BenchmarkOptions, *, pin_memory: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(options.seed)
    batch_count = max(1, min(options.iterations, 16))
    sample_count = options.batch_size * batch_count
    inputs = torch.randn(sample_count, 1, 28, 28, generator=generator)
    targets = torch.randint(0, 10, (sample_count,), generator=generator)
    dataset = TensorDataset(inputs, targets)
    shuffle_generator = torch.Generator().manual_seed(options.seed + 1)
    return DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=options.num_workers,
        pin_memory=pin_memory,
    )


def resolve_benchmark_devices(requested: RequestedDevice | str) -> tuple[torch.device, ...]:
    if requested == "auto":
        return (torch.device("cuda" if torch.cuda.is_available() else "cpu"),)
    if requested == "both":
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return tuple(devices)
    if requested == "cpu":
        return (torch.device("cpu"),)

    if torch.cuda.is_available():
        return (torch.device("cuda"),)

    message = "CUDA was requested for benchmarking, but torch.cuda.is_available() is false."
    if NIXOS_CUDA_LIBRARY.exists():
        message = (
            f"{message}\n"
            f"NixOS CUDA driver library found at {NIXOS_CUDA_LIBRARY}.\n"
            "From the repository root, allow the direnv shell first:\n"
            f"{NIXOS_CUDA_PREFIX}"
        )
    raise BenchmarkError(message)


def mnist_unavailable_message(exc: RuntimeError) -> str:
    return (
        "MNIST is not available in data/ and torchvision could not download it. "
        "If you are using direnv, run `direnv allow` at the repository root so shell.nix exports "
        "the system CA bundle before Python starts.\n"
        f"Original error: {exc}"
    )


def raw_optimizer_params(
    optimizer_name: OptimizerName,
    options: BenchmarkOptions,
) -> dict[str, float]:
    if optimizer_name == "SGD":
        return {ETA: FRONTEND_DEFAULT_OPTIMIZER_PARAMS["SGD"][ETA]}
    return {
        "N": float(options.population_size),
        P_M: options.mutation_probability,
        ETA_0: options.mutation_step,
    }


def expand_optimizer(choice: str) -> tuple[OptimizerName, ...]:
    if choice == "both":
        return ("LEEA", "SGD")
    if choice in {"LEEA", "SGD"}:
        return (choice,)
    raise BenchmarkError(f"Unsupported optimizer: {choice}")


def expand_benchmark(choice: str) -> tuple[BenchmarkName, ...]:
    if choice == "both":
        return ("synthetic", "mnist")
    if choice in {"synthetic", "mnist"}:
        return (choice,)
    raise BenchmarkError(f"Unsupported benchmark: {choice}")


def write_jsonl(results: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(result, sort_keys=True) for result in results]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_leea_profile(profile_steps: list[dict[str, float | int]]) -> dict[str, float]:
    summary = {}
    phase_keys = (
        "evaluation_seconds",
        "fitness_selection_seconds",
        "reproduction_seconds",
        "scheduler_model_copy_seconds",
        "total_step_seconds",
    )
    for key in phase_keys:
        values = [float(step[key]) for step in profile_steps if key in step]
        if not values:
            continue
        summary[f"{key}_mean"] = statistics.mean(values)
        summary[f"{key}_median"] = statistics.median(values)
    return summary


def print_summary(results: list[dict[str, Any]], stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    if not results:
        print("No benchmark results.", file=stream)
        return

    print("Benchmark results", file=stream)
    for result in results:
        population = result.get("population_size")
        population_text = f", population={population}" if population is not None else ""
        chunk_size = result.get("leea_chunk_size")
        chunk_text = f", chunk={chunk_size}" if chunk_size is not None else ""
        numeric_mode = result.get("numeric_mode", "strict")
        if result.get("numeric_mode_trajectory_changing"):
            numeric_text = f", numeric={numeric_mode} (trajectory-changing)"
        else:
            numeric_text = "" if numeric_mode == "strict" else f", numeric={numeric_mode}"
        profile_summary = result.get("leea_profile_summary") or {}
        profile_eval = profile_summary.get("evaluation_seconds_mean")
        profile_total = profile_summary.get("total_step_seconds_mean")
        profile_text = ""
        if profile_eval is not None and profile_total is not None:
            profile_text = (
                f", profile_eval_avg={profile_eval * 1000.0:.3f}ms"
                f", profile_total_avg={profile_total * 1000.0:.3f}ms"
            )
        first_ms = result["first_iteration_seconds"] * 1000.0
        steady_ms = result["steady_state_median_seconds"] * 1000.0
        print(
            (
                f"- {result['benchmark']} / {result['optimizer']} on {result['device']}"
                f": {result['iterations_per_second']:.2f} iter/s, "
                f"avg={result['average_iteration_seconds'] * 1000.0:.3f}ms/iter, "
                f"first={first_ms:.3f}ms/iter, "
                f"median={result['median_iteration_seconds'] * 1000.0:.3f}ms/iter, "
                f"steady_median={steady_ms:.3f}ms/iter, "
                f"{result['duration_seconds']:.3f}s total, final_loss={result['final_loss']:.4f}"
                f"{population_text}{chunk_text}{numeric_text}{profile_text}"
            ),
            file=stream,
        )


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"
