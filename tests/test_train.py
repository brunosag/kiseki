"""Training runs end to end on the CPU (MNIST, tiny settings)."""

import msgspec
import numpy as np
import pytest

from kiseki.spec import RunSpec, from_overrides, new_run_id, normalize
from kiseki.store import Run
from kiseki.worker import checkpoint, train


def request_after_first_block(monkeypatch, action: str) -> None:
    """The worker reads requests at start and after each block; ask once, after the first."""
    calls = []
    original = Run.pending_request

    def pending(self):
        calls.append(None)
        return action if len(calls) == 2 else original(self)

    monkeypatch.setattr(Run, "pending_request", pending)


def make(optimizer: str, **overrides) -> Run:
    settings = {
        "dataset": "mnist",
        "optimizer": optimizer,
        "iterations": 20,
        "batch_size": 128,
        "validation_interval": 5,
        "checkpoint_interval": 10,
    } | overrides
    spec = normalize(from_overrides(RunSpec(), settings))
    return Run.create(new_run_id(spec), spec, machine="local")


@pytest.mark.parametrize("optimizer", ["sgd", "openai_es", "leea", "cosyne"])
def test_run_completes_with_curves_and_models(home, mnist, optimizer):
    run = make(optimizer, **({} if optimizer == "sgd" else {"population_size": 16}))
    assert train.main(run) == 0
    status = run.status()
    assert (status.state, status.step) == ("completed", 20)
    assert status.test_accuracy is not None
    assert {"best", "final", "step-10", "step-20"} <= set(run.model_names())
    assert [r["step"] for r in run.iter_rows("val")] == [5, 10, 15, 20]
    assert [r["step"] for r in run.iter_rows("train")] == list(range(1, 21))


def test_pause_and_resume_reproduce_the_uninterrupted_run(home, mnist, monkeypatch):
    straight = make("sgd")
    assert train.main(straight) == 0

    paused = make("sgd")
    request_after_first_block(monkeypatch, "pause")
    assert train.main(paused) == 0
    assert paused.status().state == "paused"
    assert 0 < paused.status().step < 20
    checkpointed = paused.status().elapsed_seconds
    paused.update_status(elapsed_seconds=1000.0)  # as a crash after the checkpoint leaves it
    assert train.main(paused) == 0
    assert paused.status().state == "completed"
    assert checkpointed < paused.status().elapsed_seconds < 1000.0

    def final(run):
        return checkpoint.load_network(run.models_dir / "final.safetensors")

    assert np.array_equal(final(straight), final(paused))
    losses = [[r["loss"] for r in run.iter_rows("train")] for run in (straight, paused)]
    assert losses[0] == losses[1]


def test_a_run_stopped_before_validating_keeps_its_final_network_as_best(home, mnist, monkeypatch):
    run = make("sgd", iterations=40, validation_interval=30, checkpoint_interval=40)
    request_after_first_block(monkeypatch, "stop")
    assert train.main(run) == 0
    assert (run.status().state, run.status().best_step) == ("stopped", None)
    summary = msgspec.json.decode((run.path / "summary.json").read_bytes())
    assert {k: v for k, v in summary["best"].items() if k != "step"} == summary["final"]
    best, final = (
        checkpoint.load_network(run.models_dir / f"{name}.safetensors")
        for name in ("best", "final")
    )
    assert np.array_equal(best, final)
