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
from kiseki.models import build_model
from kiseki.schemas import (
    AccuracyPoint,
    COSYNE_P_M,
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
    ExperimentConfig,
    ExperimentStatus,
)
from kiseki.train import (
    RESUME_LATEST_RUN_ID,
    TrainError,
    TrainOptions,
    TrainingSignalHandler,
    build_resume_update,
    build_start_request,
    format_duration,
    format_seconds,
    format_start_summary,
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


def save_cli_resume_checkpoint(
    saver: CheckpointSaver,
    *,
    run_id: str,
    saved_at: str,
    step: int = 1,
) -> None:
    config = ExperimentConfig(
        device="cpu",
        optimizer="SGD",
        iterations=step,
        batch_size=4,
        checkpoint_interval=0,
    )
    status = ExperimentStatus(
        run_id=run_id,
        optimizer="SGD",
        current_step=step,
        last_checkpoint_step=step,
        last_checkpoint_saved_at=saved_at,
        checkpoint_path=str(saver.latest_pt_path(run_id)),
    )
    saver.save(
        model=build_model("mnist"),
        status=status,
        config=config,
        optimizer="SGD",
        run_id=run_id,
        optimizer_params={"SGD": {ETA: 0.01}},
        saved_at=saved_at,
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

    cosyne_request = build_start_request(TrainOptions(optimizer="CoSyNE"))
    cosyne_defaults = {field.key: field.default for field in OPTIMIZERS_SCHEMA["CoSyNE"]}
    assert cosyne_request.opt_params["CoSyNE"]["N"] == cosyne_defaults["N"]
    assert cosyne_request.opt_params["CoSyNE"][TOURNAMENT_SIZE] == cosyne_defaults[TOURNAMENT_SIZE]
    assert cosyne_request.opt_params["CoSyNE"][SIGMA_M] == cosyne_defaults[SIGMA_M]
    assert cosyne_request.opt_params["CoSyNE"][COSYNE_P_M] == cosyne_defaults[COSYNE_P_M]
    assert cosyne_request.opt_params["CoSyNE"][PERMUTE_ALL] == cosyne_defaults[PERMUTE_ALL]
    assert cosyne_request.opt_params["CoSyNE"][RHO_E] == 0.01
    assert cosyne_request.opt_params["CoSyNE"][ETA_SBX] == cosyne_defaults[ETA_SBX]
    assert cosyne_request.opt_params["CoSyNE"][NUM_CHILDREN] == cosyne_defaults[NUM_CHILDREN]


def test_train_resume_flag_accepts_missing_checkpoint_id() -> None:
    parser = cli.build_parser()

    resume_latest_args = parser.parse_args(["train", "--resume"])
    resume_named_args = parser.parse_args(["train", "--resume", "run-1"])

    assert train_options_from_args(resume_latest_args).resume == RESUME_LATEST_RUN_ID
    assert train_options_from_args(resume_named_args).resume == "run-1"


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


def test_train_argument_parsing_accepts_fashion_mnist() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["train", "--dataset", "fashion_mnist"])
    request = build_start_request(train_options_from_args(args))

    assert request.config.dataset == "fashion_mnist"


def test_train_cosyne_argument_parsing_builds_start_request() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--device",
            "cpu",
            "--optimizer",
            "CoSyNE",
            "--iterations",
            "20",
            "--population-size",
            "12",
            "--mutation-probability",
            "0.7",
            "--tournament-size",
            "5",
            "--mutation-stdev",
            "0.04",
            "--permute-all",
            "--elitism-ratio",
            "0.2",
            "--sbx-eta",
            "3",
            "--num-children",
            "6",
        ]
    )
    request = build_start_request(train_options_from_args(args))

    assert request.config.optimizer == "CoSyNE"
    assert request.opt_params["CoSyNE"] == {
        "N": 12.0,
        TOURNAMENT_SIZE: 5.0,
        SIGMA_M: 0.04,
        COSYNE_P_M: 0.7,
        PERMUTE_ALL: True,
        RHO_E: 0.2,
        ETA_SBX: 3.0,
        NUM_CHILDREN: 6.0,
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

    with pytest.raises(TrainError, match="--sbx-eta"):
        build_start_request(TrainOptions(optimizer="SGD", sbx_eta=2.0))


def test_train_leea_rejects_sgd_only_flags() -> None:
    with pytest.raises(TrainError, match="--learning-rate"):
        build_start_request(TrainOptions(optimizer="LEEA", learning_rate=0.05))

    with pytest.raises(TrainError, match="--mutation-stdev"):
        build_start_request(TrainOptions(optimizer="LEEA", mutation_stdev=0.05))


def test_train_cosyne_rejects_incompatible_flags() -> None:
    with pytest.raises(TrainError, match="--learning-rate"):
        build_start_request(TrainOptions(optimizer="CoSyNE", learning_rate=0.05))

    with pytest.raises(TrainError, match="--mutation-step"):
        build_start_request(TrainOptions(optimizer="CoSyNE", mutation_step=0.05))


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
    assert "Started training" in output.getvalue()
    assert "Configuration" in output.getvalue()
    assert "SGD parameters" in output.getvalue()
    assert "η" in output.getvalue()
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
    assert "Resumed training" in output.getvalue()
    assert status.current_step == 3
    assert checkpoint["config"]["iterations"] == 3
    assert checkpoint["config"]["checkpoint_interval"] == 1


def test_run_training_resumes_newest_checkpoint_when_resume_id_omitted(tmp_path) -> None:
    saver = CheckpointSaver(tmp_path)
    save_cli_resume_checkpoint(
        saver,
        run_id="older-run",
        saved_at="2026-06-20T12:00:00+00:00",
    )
    save_cli_resume_checkpoint(
        saver,
        run_id="newer-run",
        saved_at="2026-06-21T12:00:00+00:00",
    )
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=saver,
    )
    output = io.StringIO()
    errors = io.StringIO()

    exit_code = run_training(
        TrainOptions(
            resume=RESUME_LATEST_RUN_ID,
            iterations=2,
            log_every=1,
        ),
        manager=manager,
        stream=output,
        error_stream=errors,
        poll_interval=0.001,
        enable_signal_handlers=False,
    )

    status = manager.status()

    assert exit_code == 0
    assert errors.getvalue() == ""
    assert status.run_id == "newer-run"
    assert status.current_step == 2
    assert "Resumed training" in output.getvalue()
    assert "newer-run" in output.getvalue()


def test_run_training_resume_without_id_rejects_empty_checkpoint_directory(tmp_path) -> None:
    manager = ExperimentManager(
        data_loader_factory=SyntheticLoaderFactory(),
        checkpoint_saver=CheckpointSaver(tmp_path),
    )
    errors = io.StringIO()

    exit_code = run_training(
        TrainOptions(resume=RESUME_LATEST_RUN_ID),
        manager=manager,
        stream=io.StringIO(),
        error_stream=errors,
        poll_interval=0.001,
        enable_signal_handlers=False,
    )

    assert exit_code == 1
    assert "No checkpoints were found to resume" in errors.getvalue()


def test_format_start_summary_shows_config_and_unicode_optimizer_params() -> None:
    status = ExperimentStatus(run_id="run-1", optimizer="CoSyNE", requested_device="gpu")
    summary = format_start_summary(
        status,
        TrainOptions(
            dataset="cifar10",
            device="gpu",
            optimizer="CoSyNE",
            iterations=20,
            population_size=12,
            mutation_probability=0.7,
            tournament_size=5,
            mutation_stdev=0.04,
            permute_all=True,
            elitism_ratio=0.2,
            sbx_eta=3,
            num_children=6,
        ),
    )

    assert summary == (
        "Started training\n"
        "  Run ID  run-1\n"
        "\n"
        "Configuration\n"
        "  Dataset              cifar10\n"
        "  Optimizer            CoSyNE\n"
        "  Requested device     gpu\n"
        "  Seed                 42\n"
        "  Batch size           512\n"
        "  Iterations           20\n"
        "  Target accuracy      100.00%\n"
        "  Deterministic        no\n"
        "  Checkpoint interval  50\n"
        "\n"
        "CoSyNE parameters\n"
        "  N      12    Population size\n"
        "  k      5     Tournament size\n"
        "  σₘ     0.04  Mutation standard deviation\n"
        "  pₘ     0.7   Mutation probability\n"
        "  ρₑ     0.2   Elitism fraction\n"
        "  η_SBX  3     SBX distribution index\n"
        "  λ_c    6     Children count\n"
        "  π_all  yes   Permute all columns"
    )


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
        "t=01m 05s    Δt̄=12ms    η=0.0300"
    )
    assert "loss=" not in line
    assert "last_checkpoint_step" not in line
    assert "checkpoint" not in line


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0ms"),
        (0.01234, "12ms"),
        (0.999, "999ms"),
        (1.271, "1.27s"),
        (14.24, "14.2s"),
        (102.4, "102s"),
    ],
)
def test_format_seconds_uses_milliseconds_below_one_second(
    seconds: float, expected: str
) -> None:
    assert format_seconds(seconds) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (5.4, "05s"),
        (65.4, "01m 05s"),
        (3600.0, "1h 00m 00s"),
        (3605.4, "1h 00m 05s"),
        (3723.4, "1h 02m 03s"),
    ],
)
def test_format_duration_pads_minutes_and_seconds(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_resume_rejects_trajectory_changing_overrides() -> None:
    with pytest.raises(TrainError, match="--optimizer"):
        build_resume_update(TrainOptions(resume="run-1", optimizer="SGD", iterations=10))

    with pytest.raises(TrainError, match="--sbx-eta"):
        build_resume_update(TrainOptions(resume="run-1", sbx_eta=2.0, iterations=10))


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
