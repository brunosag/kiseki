"""Mirrors of remote runs: what a refresh pulls from the machine a run lives on."""

from kiseki import actions
from kiseki.spec import RunSpec
from kiseki.store import Run


class FakeRemote:
    """The remote side of rsync: the first pull brings the status it has there."""

    def __init__(self, state_there: dict[str, str]):
        self.state_there = state_there
        self.pulls: list[tuple[list[str], bool]] = []

    def pull(self, run_ids, *, models=False):
        self.pulls.append((list(run_ids), models))
        for run_id in run_ids:
            Run.find(run_id).update_status(state=self.state_there[run_id])


def test_models_come_once_with_the_pull_that_sees_a_run_finish(home, monkeypatch):
    done = Run.create("20261017-100000-mnist-sgd-aaaa", RunSpec(), machine="vps")
    going = Run.create("20261017-100000-mnist-sgd-bbbb", RunSpec(), machine="vps")
    for run in (done, going):
        run.update_status(state="running")
    vps = FakeRemote({done.id: "completed", going.id: "running"})
    monkeypatch.setattr(actions, "remote_for", lambda name: vps)

    assert actions.refresh([done, going]) == []
    assert vps.pulls == [([done.id, going.id], False), ([done.id], True)]

    actions.refresh([done, going])
    assert vps.pulls[2:] == [([going.id], False)]
