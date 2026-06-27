import io
import json
import os
import signal
import subprocess
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import kiseki.cli as cli
from kiseki.checkpoint import CheckpointSaver
from kiseki.experiment import ExperimentManager
from kiseki.schemas import (
    AccuracyPoint,
    ETA,
    ETA_0,
    GAMMA,
    LAMBDA,
    OPTIMIZERS_SCHEMA,
    P_M,
    RHO,
    RHO_X,
    TAU_PAT,
    ExperimentConfig,
    ExperimentStatus,
)
from kiseki.train import (
    TrainError,
    TrainOptions,
    TrainingSignalHandler,
    build_resume_update,
    build_start_request,
    format_status,
    run_training,
    train_options_from_args,
)


class SyntheticLoaderFactory:
    def mnist(self, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
        generator = torch.Generator().manual_seed(seed)
        inputs = torch.randn(24, 1, 28, 28, generator=generator)
        targets = torch.randint(0, 10, (24,), generator=generator)
        dataset = TensorDataset(inputs, targets)
        return (
            DataLoader(dataset, batch_size=batch_size, shuffle=True),
            DataLoader(dataset, batch_size=batch_size, shuffle=False),
        )


def test_train_defaults_match_frontend_schema() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["train"])
    request = build_start_request(train_options_from_args(args))
    config = ExperimentConfig()
    leea_defaults = {field.key: field.default for field in OPTIMIZERS_SCHEMA["LEEA"]}

    assert request.config == config
    assert request.opt_params["LEEA"]["N"] == leea_defaults["N"]
    assert request.opt_params["LEEA"][P_M] == leea_defaults[P_M]
    assert request.opt_params["LEEA"][ETA_0] == leea_defaults[ETA_0]
    assert request.opt_params["LEEA"][GAMMA] == leea_defaults[GAMMA]
    assert request.opt_params["LEEA"][RHO] == leea_defaults[RHO]
    assert request.opt_params["LEEA"][RHO_X] == leea_defaults[RHO_X]
    assert request.opt_params["LEEA"][LAMBDA] == leea_defaults[LAMBDA]
    assert request.opt_params["LEEA"][TAU_PAT] == leea_defaults[TAU_PAT]


def test_train_argument_parsing_builds_start_request() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset",
            "cifar10",
            "--device",
            "cpu",
            "--optimizer",
            "LEEA",
            "--iterations",
            "20",
            "--target-acc",
            "95.5",
            "--batch-size",
            "8",
            "--seed",
            "7",
            "--deterministic",
            "--checkpoint-interval",
            "2",
            "--log-every",
            "3",
            "--population-size",
            "12",
            "--mutation-probability",
            "0.2",
            "--mutation-step",
            "0.01",
            "--mutation-decay",
            "0.9",
            "--retention-fraction",
            "0.3",
            "--crossover-fraction",
            "0.4",
            "--fitness-decay",
            "0.1",
            "--validation-patience",
            "7",
        ]
    )
    options = train_options_from_args(args)
    request = build_start_request(options)

    assert options.log_every == 3
    assert request.config.dataset == "cifar10"
    assert request.config.device == "cpu"
    assert request.config.optimizer == "LEEA"
    assert request.config.iterations == 20
    assert request.config.target_acc == 95.5
    assert request.config.batch_size == 8
    assert request.config.seed == 7
    assert request.config.deterministic is True
    assert request.config.checkpoint_interval == 2
    assert request.opt_params["LEEA"] == {
        "N": 12.0,
        P_M: 0.2,
        ETA_0: 0.01,
        GAMMA: 0.9,
        RHO: 0.3,
        RHO_X: 0.4,
        LAMBDA: 0.1,
        TAU_PAT: 7.0,
    }


def test_train_sgd_params_and_rejects_leea_only_flags() -> None:
    request = build_start_request(
        TrainOptions(
            device="cpu",
            optimizer="SGD",
            iterations=1,
            learning_rate=0.05,
        )
    )

    assert request.opt_params == {"SGD": {ETA: 0.05}}

    with pytest.raises(TrainError, match="--population-size"):
        build_start_request(TrainOptions(optimizer="SGD", population_size=4))


def test_train_leea_rejects_sgd_only_flags() -> None:
    with pytest.raises(TrainError, match="--learning-rate"):
        build_start_request(TrainOptions(optimizer="LEEA", learning_rate=0.05))


def test_run_training_saves_checkpoint(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    output = io.StringIO()
    errors = io.StringIO()

    exit_code = run_training(
        TrainOptions(
            device="cpu",
            optimizer="SGD",
            iterations=2,
            batch_size=4,
            checkpoint_interval=1,
            learning_rate=0.01,
            log_every=1,
        ),
        manager=manager,
        stream=output,
        error_stream=errors,
        poll_interval=0.001,
        enable_signal_handlers=False,
    )

    status = manager.status()
    checkpoint = saver.load_latest(status.run_id)

    assert exit_code == 0
    assert errors.getvalue() == ""
    assert "Started run_id=" in output.getvalue()
    assert "Finished run_id=" in output.getvalue()
    assert status.current_step == 2
    assert status.checkpoint_path == str(tmp_path / status.run_id / "latest.pt")
    assert checkpoint["config"]["iterations"] == 2
    assert checkpoint["optimizer_params"] == {"SGD": {ETA: 0.01}}


def test_run_training_resumes_checkpoint_with_allowed_overrides(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    first_manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )

    assert (
        run_training(
            TrainOptions(
                device="cpu",
                optimizer="SGD",
                iterations=1,
                batch_size=4,
                checkpoint_interval=1,
                learning_rate=0.01,
            ),
            manager=first_manager,
            stream=io.StringIO(),
            error_stream=io.StringIO(),
            poll_interval=0.001,
            enable_signal_handlers=False,
        )
        == 0
    )
    run_id = first_manager.status().run_id

    second_manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    output = io.StringIO()
    exit_code = run_training(
        TrainOptions(
            resume=run_id,
            iterations=3,
            checkpoint_interval=1,
            log_every=1,
        ),
        manager=second_manager,
        stream=output,
        error_stream=io.StringIO(),
        poll_interval=0.001,
        enable_signal_handlers=False,
    )

    status = second_manager.status()
    checkpoint = saver.load_latest(run_id)

    assert exit_code == 0
    assert "Resumed run_id=" in output.getvalue()
    assert status.current_step == 3
    assert checkpoint["config"]["iterations"] == 3
    assert checkpoint["config"]["checkpoint_interval"] == 1


def test_format_status_uses_interval_stats_and_omits_checkpoint_fields() -> None:
    status = ExperimentStatus(
        current_step=1234,
        best_acc=98.123,
        total_elapsed_seconds=65.4,
        loss_mean_since_validation=0.123456,
        loss_stdev_since_validation=0.001234,
        mean_iteration_seconds_since_validation=0.01234,
        current_mutation_step=0.03,
        last_checkpoint_step=100,
        checkpoint_path="/tmp/run/latest.pt",
    )
    status.history.acc = [
        AccuracyPoint(i=1000, value=97.0),
        AccuracyPoint(i=1230, value=98.123),
    ]

    line = format_status(status)

    assert line == (
        "i=1,234    ℓ=0.1235 ± 0.0012    a*=98.12% (1,230)    "
        "t=1m 05s    Δt̄=0.012s    η=0.0300"
    )
    assert "loss=" not in line
    assert "last_checkpoint_step" not in line
    assert "checkpoint" not in line


def test_resume_rejects_trajectory_changing_overrides() -> None:
    with pytest.raises(TrainError, match="--optimizer"):
        build_resume_update(TrainOptions(resume="run-1", optimizer="SGD", iterations=10))


def test_signal_handler_pauses_then_stops() -> None:
    class FakeManager:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def pause(self) -> None:
            self.calls.append("pause")

        def stop(self) -> None:
            self.calls.append("stop")

    manager = FakeManager()
    stream = io.StringIO()
    handler = TrainingSignalHandler(manager, stream=stream)  # type: ignore[arg-type]

    handler(signal.SIGTERM, None)
    handler(signal.SIGTERM, None)

    assert manager.calls == ["pause", "stop"]
    assert handler.interrupted is True
    assert "pausing" in stream.getvalue()
    assert "stopping" in stream.getvalue()


def test_main_dispatches_train(monkeypatch) -> None:
    captured: dict[str, TrainOptions] = {}

    def fake_run_training(options: TrainOptions) -> int:
        captured["options"] = options
        return 7

    monkeypatch.setattr(cli, "run_training", fake_run_training)

    assert cli.main(["train", "--device", "cpu", "--optimizer", "SGD"]) == 7
    assert captured["options"].device == "cpu"
    assert captured["options"].optimizer == "SGD"


def test_removed_command_is_rejected(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["benchmark"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "invalid choice" in captured.err


def test_train_tmux_script_and_npm_alias_are_configured() -> None:
    repo_root = Path(__file__).parents[2]
    script = repo_root / "scripts" / "train-tmux.sh"
    package_json = json.loads((repo_root / "package.json").read_text(encoding="utf-8"))
    script_text = script.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script)], check=True)

    assert os.access(script, os.X_OK)
    assert package_json["scripts"]["train"] == "bash scripts/train-tmux.sh"
    assert "uv run kiseki train" in script_text
    assert "tmux new-session" in script_text
    assert "tmux attach" in script_text


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
