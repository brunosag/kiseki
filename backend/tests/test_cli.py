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

    assert args.device == "both"
    assert args.optimizer == "both"
    assert args.benchmark == config.dataset
    assert args.iterations == 10
    assert args.batch_size == config.batch_size
    assert args.seed == config.seed
    assert args.population_size == leea_defaults["N"]
    assert args.mutation_probability == leea_defaults[P_M]
    assert args.mutation_step == leea_defaults[ETA_0]
    assert args.leea_chunk_size is None
    assert args.leea_profile is False
    assert args.numeric_mode == "strict"


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
            "--leea-chunk-size",
            "3",
            "--leea-profile",
            "--numeric-mode",
            "fast",
        ]
    )

    assert args.command == "benchmark"
    assert args.optimizer == "LEEA"
    assert args.benchmark == "synthetic"
    assert args.population_size == 6
    assert args.mutation_probability == 0.2
    assert args.leea_chunk_size == 3
    assert args.leea_profile is True
    assert args.numeric_mode == "fast"


def test_benchmark_accepts_cifar10_and_expands_both() -> None:
    parser = build_parser()
    args = parser.parse_args(["benchmark", "--benchmark", "cifar10"])

    assert args.benchmark == "cifar10"
    assert benchmark.expand_benchmark("both") == ("synthetic", "mnist", "cifar10")


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
    assert "speed_mode" not in result
    assert result["numeric_mode"] == "strict"
    assert result["numeric_mode_trajectory_changing"] is False
    assert result["duration_seconds"] > 0
    assert result["iterations_per_second"] > 0
    assert result["average_iteration_seconds"] > 0
    assert result["median_iteration_seconds"] > 0
    assert isinstance(result["final_loss"], float)
    assert result["status"] == "ok"
    assert "avg=" in captured.out
    assert "median=" in captured.out
    assert "steady_median=" in captured.out


def test_benchmark_cli_writes_leea_profile_jsonl(tmp_path, capsys) -> None:
    output = tmp_path / "profile.jsonl"

    exit_code = main(
        [
            "benchmark",
            "--device",
            "cpu",
            "--benchmark",
            "synthetic",
            "--optimizer",
            "LEEA",
            "--iterations",
            "1",
            "--batch-size",
            "4",
            "--population-size",
            "4",
            "--leea-chunk-size",
            "2",
            "--leea-profile",
            "--numeric-mode",
            "fast",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(output.read_text(encoding="utf-8").strip())

    assert exit_code == 0
    assert result["leea_profile"] is True
    assert result["leea_profile_steps"][0]["iteration"] == 1
    assert result["leea_profile_summary"]["evaluation_seconds_mean"] >= 0.0
    assert result["numeric_mode"] == "fast"
    assert result["numeric_mode_trajectory_changing"] is True
    assert "profile_eval_avg=" in captured.out
    assert "trajectory-changing" in captured.out


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
    assert "repository root" in captured.err
    assert "direnv allow" in captured.err
    assert "uv run kiseki benchmark --device gpu" in captured.err


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


def test_explicit_cifar10_failure_returns_error(monkeypatch, capsys) -> None:
    def fail_cifar10(*args, **kwargs):
        raise RuntimeError("certificate verify failed")

    monkeypatch.setattr(benchmark, "load_cifar10", fail_cifar10)

    exit_code = main(
        [
            "benchmark",
            "--device",
            "cpu",
            "--benchmark",
            "cifar10",
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
    assert "CIFAR-10 is not available" in captured.err
    assert "certificate verify failed" in captured.err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--data-dir", "mnist-cache"], "unrecognized arguments: --data-dir"),
        (["--no-download"], "unrecognized arguments: --no-download"),
        (["--speed-mode", "fast"], "unrecognized arguments: --speed-mode"),
        (["--leea-evaluator", "on"], "unrecognized arguments: --leea-evaluator"),
        (["--leea-eval-backend", "generic"], "unrecognized arguments: --leea-eval-backend"),
    ],
)
def test_benchmark_parser_rejects_removed_options(argv, message, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["benchmark", *argv])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert message in captured.err


def test_direnv_shell_sets_nixos_environment() -> None:
    repo_root = Path(__file__).parents[2]
    envrc = repo_root / ".envrc"
    shell_nix = repo_root / "shell.nix"
    text = shell_nix.read_text(encoding="utf-8")

    assert envrc.read_text(encoding="utf-8").strip() == "use nix"
    assert "LD_LIBRARY_PATH" in text
    assert "TRITON_LIBCUDA_PATH" in text
    assert "CUBLAS_WORKSPACE_CONFIG" in text
    assert "SSL_CERT_FILE" in text


def test_auto_device_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)

    assert benchmark.resolve_benchmark_devices("auto") == (benchmark.torch.device("cuda"),)


def test_auto_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)

    assert benchmark.resolve_benchmark_devices("auto") == (benchmark.torch.device("cpu"),)


def test_both_device_includes_cpu_and_available_cuda(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)

    assert benchmark.resolve_benchmark_devices("both") == (
        benchmark.torch.device("cpu"),
        benchmark.torch.device("cuda"),
    )


def test_both_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)

    assert benchmark.resolve_benchmark_devices("both") == (benchmark.torch.device("cpu"),)
