import json
from pathlib import Path

import pytest

from kiseki import benchmark
from kiseki.cli import build_parser, main
from kiseki.schemas import ETA_0, OPTIMIZERS_SCHEMA, P_M, ExperimentConfig


def test_benchmark_defaults_match_frontend_schema() -> None:
    parser = build_parser()
    args = parser.parse_args(["benchmark"])
    config = ExperimentConfig()
    leea_defaults = {field.key: field.default for field in OPTIMIZERS_SCHEMA["LEEA"]}

    assert args.device == "auto"
    assert args.optimizer == config.optimizer
    assert args.benchmark == config.dataset
    assert args.speed_mode == config.speed_mode
    assert args.iterations == config.iterations
    assert args.batch_size == config.batch_size
    assert args.seed == config.seed
    assert args.population_size == leea_defaults["N"]
    assert args.mutation_probability == leea_defaults[P_M]
    assert args.mutation_step == leea_defaults[ETA_0]


def test_benchmark_argument_parsing() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "--device",
            "cpu",
            "--optimizer",
            "LEEA",
            "--benchmark",
            "synthetic",
            "--speed-mode",
            "fast",
            "--iterations",
            "2",
            "--batch-size",
            "4",
            "--population-size",
            "6",
            "--mutation-probability",
            "0.2",
            "--mutation-step",
            "0.01",
        ]
    )

    assert args.command == "benchmark"
    assert args.optimizer == "LEEA"
    assert args.benchmark == "synthetic"
    assert args.speed_mode == "fast"
    assert args.population_size == 6
    assert args.mutation_probability == 0.2


def test_benchmark_cli_writes_synthetic_jsonl(tmp_path, capsys) -> None:
    output = tmp_path / "results.jsonl"

    exit_code = main(
        [
            "benchmark",
            "--device",
            "cpu",
            "--benchmark",
            "synthetic",
            "--optimizer",
            "SGD",
            "--iterations",
            "1",
            "--batch-size",
            "4",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(output.read_text(encoding="utf-8").strip())

    assert exit_code == 0
    assert "Benchmark results" in captured.out
    assert result["benchmark"] == "synthetic"
    assert result["optimizer"] == "SGD"
    assert result["device"] == "cpu"
    assert result["iterations"] == 1
    assert result["batch_size"] == 4
    assert result["duration_seconds"] > 0
    assert result["iterations_per_second"] > 0
    assert result["average_iteration_seconds"] > 0
    assert result["median_iteration_seconds"] > 0
    assert isinstance(result["final_loss"], float)
    assert result["status"] == "ok"
    assert "avg=" in captured.out
    assert "median=" in captured.out


def test_benchmark_cli_prints_nixos_cuda_hint(monkeypatch, tmp_path, capsys) -> None:
    cuda_library = tmp_path / "libcuda.so.1"
    cuda_library.touch()
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(benchmark, "NIXOS_CUDA_LIBRARY", cuda_library)

    exit_code = main(
        [
            "benchmark",
            "--device",
            "gpu",
            "--benchmark",
            "synthetic",
            "--optimizer",
            "SGD",
            "--iterations",
            "1",
            "--batch-size",
            "4",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "NixOS CUDA driver library found" in captured.err
    assert "direnv allow" in captured.err
    assert "uv run kiseki benchmark --device gpu --speed-mode fast" in captured.err


def test_explicit_mnist_failure_returns_error(monkeypatch, capsys) -> None:
    def fail_mnist(*args, **kwargs):
        raise RuntimeError("certificate verify failed")

    monkeypatch.setattr(benchmark, "load_mnist", fail_mnist)

    exit_code = main(
        [
            "benchmark",
            "--device",
            "cpu",
            "--benchmark",
            "mnist",
            "--optimizer",
            "SGD",
            "--iterations",
            "1",
            "--batch-size",
            "4",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "MNIST is not available" in captured.err
    assert "certificate verify failed" in captured.err


@pytest.mark.parametrize("removed_option", ["--data-dir", "--no-download"])
def test_benchmark_parser_rejects_removed_mnist_options(removed_option, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["benchmark", removed_option, "mnist-cache"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert f"unrecognized arguments: {removed_option}" in captured.err


def test_direnv_shell_sets_nixos_environment() -> None:
    backend_dir = Path(__file__).parents[1]
    envrc = backend_dir / ".envrc"
    shell_nix = backend_dir / "shell.nix"
    text = shell_nix.read_text(encoding="utf-8")

    assert envrc.read_text(encoding="utf-8").strip() == "use nix"
    assert "LD_LIBRARY_PATH" in text
    assert "TRITON_LIBCUDA_PATH" in text
    assert "SSL_CERT_FILE" in text


def test_auto_device_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)

    assert benchmark.resolve_benchmark_device("auto") == benchmark.torch.device("cuda")


def test_auto_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)

    assert benchmark.resolve_benchmark_device("auto") == benchmark.torch.device("cpu")
