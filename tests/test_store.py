import pytest

from kiseki import actions, runner
from kiseki.spec import RunSpec
from kiseki.store import Run, read_ndjson


def test_ndjson_skips_a_torn_last_line_and_resumes_at_offset(tmp_path):
    path = tmp_path / "val.ndjson"
    path.write_bytes(b'{"step":1}\n{"step":2}\n{"st')
    rows, offset = read_ndjson(path)
    assert [r["step"] for r in rows] == [1, 2]
    assert offset == len(b'{"step":1}\n{"step":2}\n')
    with path.open("ab") as handle:
        handle.write(b'ep":3}\n')
    rows, _ = read_ndjson(path, offset)
    assert rows == [{"step": 3}]


def test_find_by_unique_part(home):
    a = Run.create("20261017-100000-mnist-sgd-aaaa", RunSpec(), machine="local")
    Run.create("20261017-100000-mnist-sgd-bbbb", RunSpec(), machine="local")
    assert Run.find("aaaa").id == a.id
    with pytest.raises(KeyError):
        Run.find("20261017")


def test_control_of_queued_and_paused_runs(home, monkeypatch):
    started = []
    monkeypatch.setattr(runner, "ensure_runner", started.append)
    run = Run.create("20261017-100000-mnist-sgd-cccc", RunSpec(), machine="local")
    assert actions.control_here(run, "pause")["result"] == "paused"
    assert actions.control_here(run, "pause")["result"] == "already paused"
    assert actions.control_here(run, "resume")["result"] == "queued"
    assert started
    run.update_status(state="running")
    assert actions.control_here(run, "stop")["result"] == "stop requested"
    assert run.pending_request() == "stop"
