import argparse
import sys
from collections.abc import Sequence

from .dataset_types import REAL_DATASETS
from .train import run_training, train_options_from_args


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return run_training(train_options_from_args(args))

    parser.print_help(sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kiseki")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Run headless training")
    train.add_argument("--dataset", choices=REAL_DATASETS)
    train.add_argument("--device", choices=("cpu", "gpu"))
    train.add_argument("--optimizer", choices=("LEEA", "SGD", "CoSyNE"))
    train.add_argument("--iterations", type=positive_int)
    train.add_argument("--target-acc", type=accuracy_percentage)
    train.add_argument("--batch-size", type=positive_int)
    train.add_argument("--seed", type=int)
    train.add_argument("--deterministic", action="store_true", default=None)
    train.add_argument("--checkpoint-interval", type=non_negative_int)
    train.add_argument("--log-every", type=positive_int, default=10)
    train.add_argument("--learning-rate", type=positive_float)
    train.add_argument("--population-size", type=positive_int)
    train.add_argument("--mutation-probability", type=probability)
    train.add_argument("--mutation-step", type=non_negative_float)
    train.add_argument("--mutation-stdev", type=non_negative_float)
    train.add_argument("--mutation-decay", type=non_negative_float)
    train.add_argument("--retention-fraction", type=probability)
    train.add_argument("--crossover-fraction", type=probability)
    train.add_argument("--fitness-decay", type=non_negative_float)
    train.add_argument("--validation-patience", type=positive_int)
    train.add_argument("--tournament-size", type=positive_int)
    train.add_argument("--permute-all", action="store_true", default=None)
    train.add_argument("--elitism-ratio", type=probability)
    train.add_argument("--sbx-eta", type=non_negative_float)
    train.add_argument("--num-children", type=non_negative_int)
    train.add_argument("--resume")
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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
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


def accuracy_percentage(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
