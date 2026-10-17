"""The queue runner: which runs it owns, the notice before a run, and starting a run once."""

from kiseki import runner
from kiseki.machines import Contact
from kiseki.spec import RunSpec
from kiseki.store import Run
from kiseki.worker import train

CONTACT = Contact("Ada Lovelace", "ada@example.org")


class FakeWorker:
    pid = 4242

    def __init__(self, args, **kwargs):
        self.args = args


def test_mirrors_of_remote_runs_are_neither_started_nor_failed(home):
    mine = Run.create("20261017-100000-mnist-sgd-aaaa", RunSpec(), machine="local")
    queued_there = Run.create("20261017-100000-mnist-sgd-bbbb", RunSpec(), machine="lugia")
    running_there = Run.create("20261017-100000-mnist-sgd-cccc", RunSpec(), machine="lugia")
    running_there.update_status(state="running", pid=None)

    assert [run.id for run in runner.queued(home)] == [mine.id]
    assert runner.reconcile(home) == []
    assert queued_there.status().state == "queued"
    assert running_there.status().state == "running"


def test_runs_submitted_in_the_same_second_start_in_submission_order(home):
    ids = [f"20261017-100000-mnist-{name}" for name in ("sgd-zzzz", "leea-aaaa", "cosyne-mmmm")]
    for run_id in ids:
        Run.create(run_id, RunSpec(), machine="local")

    assert [run.id for run in runner.queued(home)] == ids


def test_a_run_does_not_start_when_the_notice_cannot_be_written(home, tmp_path, monkeypatch):
    missing = tmp_path / "etc" / "motd.tail"
    runner.write_profile(home, runner.Profile(name="lugia", notice=str(missing), contact=CONTACT))
    run = Run.create("20261017-100000-mnist-sgd-aaaa", RunSpec(), machine="lugia")
    started = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: started.append(a))

    runner.Runner(home).start(run)

    assert started == []
    assert run.status().state == "failed"
    assert "notice" in (run.status().error or "")


def test_starting_writes_the_notice_first_and_owns_the_run_at_once(home, tmp_path, monkeypatch):
    motd = tmp_path / "motd.tail"
    motd.write_text("Welcome.\n")
    runner.write_profile(home, runner.Profile(name="lugia", notice=str(motd), contact=CONTACT))
    run = Run.create("20261017-100000-mnist-sgd-aaaa", RunSpec(), machine="lugia")
    seen = {}

    def launch(args, **kwargs):
        seen["notice"] = motd.read_text()
        return FakeWorker(args)

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(
        runner.Runner,
        "wait",
        lambda self, run, process: seen.update(state=run.status().state, pid=run.status().pid),
    )

    runner.Runner(home).start(run)

    assert "Ada Lovelace" in seen["notice"]
    assert (seen["state"], seen["pid"]) == ("running", FakeWorker.pid)


def test_a_worker_honours_a_pause_requested_before_it_started(home):
    run = Run.create("20261017-100000-mnist-sgd-aaaa", RunSpec(), machine="local")
    run.update_status(state="running", pid=4242)  # as the runner leaves it
    run.request("pause")

    assert train.main(run) == 0

    status = run.status()
    assert (status.state, status.step, status.pid) == ("paused", 0, None)
    assert run.pending_request() is None
    assert list(run.iter_rows("train")) == []
