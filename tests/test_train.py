"""Training runs end to end on the CPU (MNIST, tiny settings)."""

import numpy as np
import pytest

from kiseki.spec import RunSpec, from_overrides, new_run_id, normalize
from kiseki.store import Run
from kiseki.worker import checkpoint, train


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
    asked = []
    original = Run.pending_request

    def pause_once(self):
        if not asked:
            asked.append(True)
            return "pause"
        return original(self)

    monkeypatch.setattr(Run, "pending_request", pause_once)
    assert train.main(paused) == 0
    assert paused.status().state == "paused"
    assert 0 < paused.status().step < 20
    assert train.main(paused) == 0
    assert paused.status().state == "completed"

    def final(run):
        return checkpoint.load_network(run.models_dir / "final.safetensors")

    assert np.array_equal(final(straight), final(paused))
    losses = [[r["loss"] for r in run.iter_rows("train")] for run in (straight, paused)]
    assert losses[0] == losses[1]
