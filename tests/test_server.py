import asyncio
import threading
import time

import pytest
from starlette.testclient import TestClient

from kiseki import actions, remote, runner, server
from kiseki.report import Report, ReportSpec
from kiseki.server import create_app
from kiseki.spec import RunSpec
from kiseki.store import Run

TOKEN = {"X-Kiseki-Token": "secret"}


@pytest.fixture
def client(home):
    with TestClient(create_app("secret"), base_url="http://127.0.0.1") as client:
        yield client


def test_api_needs_the_token_and_a_local_host(client):
    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/runs", headers=TOKEN).json() == []
    assert client.get("/api/runs", headers=TOKEN | {"Host": "evil.example"}).status_code == 403
    assert client.get("/api/runs?token=secret").status_code == 200


def test_rows_come_column_wise_from_an_offset(client):
    run = Run.create("20261017-100000-mnist-sgd-aaaa", RunSpec(), machine="local")
    run.append("val", [{"step": 5, "val_accuracy": 0.5}, {"step": 10, "val_accuracy": 0.7}])
    first = client.get(f"/api/runs/{run.id}/rows/val", headers=TOKEN).json()
    assert first["columns"] == {"step": [5, 10], "val_accuracy": [0.5, 0.7]}
    run.append("val", [{"step": 15, "val_accuracy": 0.8}])
    more = client.get(f"/api/runs/aaaa/rows/val?offset={first['offset']}", headers=TOKEN).json()
    assert more["columns"] == {"step": [15], "val_accuracy": [0.8]}


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope", headers=TOKEN).status_code == 404


def test_queue_runs(client, monkeypatch):
    monkeypatch.setattr(runner, "ensure_runner", lambda home: False)
    response = client.post(
        "/api/runs",
        headers=TOKEN,
        json={
            "overrides": {"dataset": "mnist"},
            "grid": {"optimizer": ["sgd", "leea"]},
            "preset": "smoke",
        },
    )
    assert response.status_code == 201
    assert [r["status"]["state"] for r in response.json()] == ["queued", "queued"]
    assert len(client.get("/api/runs", headers=TOKEN).json()) == 2


def test_a_failed_deploy_says_why(client, home, monkeypatch):
    config = home.parent / "config"
    config.mkdir()
    (config / "machines.toml").write_text('[machines.box]\nssh = "box"\n')

    def deploy(self):
        raise remote.RemoteError("box: `uv sync` exited 2: no wheels for Python 3.15")

    monkeypatch.setattr(remote.Remote, "deploy", deploy)
    response = client.post("/api/runs", headers=TOKEN, json={"preset": "smoke", "machine": "box"})
    assert response.status_code == 502
    assert response.json() == {"error": "box: `uv sync` exited 2: no wheels for Python 3.15"}


def test_report_files_stay_inside_the_report(client, home):
    report = Report.create(ReportSpec(models=["x:best"]))
    (report.path / "figures").mkdir()
    (report.path / "figures" / "a.png").write_bytes(b"png")
    base = f"/api/reports/{report.id}/files"
    assert client.get(f"{base}/figures/a.png", headers=TOKEN).content == b"png"
    assert client.get(f"{base}/..%2F..%2Fconfig", headers=TOKEN).status_code == 404


def test_run_files_stay_inside_the_run(client):
    run = Run.create("20261017-100000-mnist-sgd-aaaa", RunSpec(), machine="local")
    (run.path / "worker.log").write_text("compiled in 1.0s\n")
    base = f"/api/runs/{run.id}/files"
    response = client.get(f"{base}/worker.log", headers=TOKEN)
    assert response.text == "compiled in 1.0s\n"
    assert response.headers["content-type"].startswith("text/plain")
    assert client.get(f"{base}/..%2F..%2F..%2Fconfig", headers=TOKEN).status_code == 404


def test_a_stale_machine_cache_answers_at_once_while_a_probe_runs(home, monkeypatch):
    probes, release = [], threading.Event()

    def machine_records():
        probes.append(None)
        if len(probes) > 1:  # the first probe, at startup, returns at once
            release.wait(10)
        return [{"name": "local", "probe": len(probes)}]

    monkeypatch.setattr(actions, "machine_records", machine_records)
    with TestClient(create_app("secret"), base_url="http://127.0.0.1") as client:
        assert client.get("/api/machines", headers=TOKEN).json() == [{"name": "local", "probe": 1}]
        monkeypatch.setattr(server, "MACHINES_TTL", 0.0)  # the cache is stale from now on
        started = time.monotonic()
        stale = client.get("/api/machines", headers=TOKEN).json()
        assert time.monotonic() - started < 1
        assert stale == [{"name": "local", "probe": 1}]
        release.set()
        assert client.get("/api/machines?fresh=1", headers=TOKEN).json()[0]["probe"] >= 2


def test_a_refresh_announces_finished_remote_runs_once_their_models_are_here(home, monkeypatch):
    done = Run.create("20261017-100000-mnist-sgd-aaaa", RunSpec(), machine="vps")
    going = Run.create("20261017-100000-mnist-sgd-bbbb", RunSpec(), machine="vps")
    for run in (done, going):
        run.update_status(state="running")

    def refresh(runs):  # the pull: one finished there, with its models
        done.update_status(state="completed")
        return []

    monkeypatch.setattr(actions, "refresh", refresh)
    state = server.State("secret")
    events = asyncio.Queue()
    state.subscribers.add(events)

    asyncio.run(server.refresh_remote_runs(state))
    message = events.get_nowait()
    assert message.startswith("event: run\n") and done.id in message and "completed" in message
    assert events.empty()
