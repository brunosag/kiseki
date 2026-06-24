import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .benchmark import (
    BenchmarkError,
    BenchmarkOptions,
    print_summary,
    run_benchmarks,
    write_jsonl,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        options = BenchmarkOptions(
            device=args.device,
            optimizer=args.optimizer,
            benchmark=args.benchmark,
            iterations=args.iterations,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            population_size=args.population_size,
            mutation_probability=args.mutation_probability,
            mutation_step=args.mutation_step,
            leea_chunk_size=args.leea_chunk_size,
            leea_profile=args.leea_profile,
            numeric_mode=args.numeric_mode,
            output=args.output,
        )
        try:
            results = run_benchmarks(options)
        except BenchmarkError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print_summary(results)
        if options.output is not None:
            write_jsonl(results, options.output)
        return 0

    parser.print_help(sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kiseki")
    subparsers = parser.add_subparsers(dest="command", required=True)
    defaults = BenchmarkOptions()

    benchmark = subparsers.add_parser("benchmark", help="Run backend training benchmarks")
    benchmark.add_argument(
        "--device",
        choices=("auto", "both", "cpu", "gpu"),
        default=defaults.device,
    )
    benchmark.add_argument(
        "--optimizer",
        choices=("LEEA", "SGD", "both"),
        default=defaults.optimizer,
    )
    benchmark.add_argument(
        "--benchmark",
        choices=("synthetic", "mnist", "cifar10", "both"),
        default=defaults.benchmark,
    )
    benchmark.add_argument("--iterations", type=positive_int, default=defaults.iterations)
    benchmark.add_argument("--batch-size", type=positive_int, default=defaults.batch_size)
    benchmark.add_argument("--seed", type=int, default=defaults.seed)
    benchmark.add_argument("--num-workers", type=non_negative_int, default=0)
    benchmark.add_argument("--population-size", type=positive_int, default=defaults.population_size)
    benchmark.add_argument("--leea-chunk-size", type=positive_int)
    benchmark.add_argument("--leea-profile", action="store_true", default=defaults.leea_profile)
    benchmark.add_argument(
        "--numeric-mode",
        choices=("strict", "fast"),
        default=defaults.numeric_mode,
    )
    benchmark.add_argument(
        "--mutation-probability",
        type=probability,
        default=defaults.mutation_probability,
    )
    benchmark.add_argument("--mutation-step", type=non_negative_float, default=defaults.mutation_step)
    benchmark.add_argument("--output", type=Path)
    return parser


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
