"""A machine's run queue: one runner process starts queued runs one at a time.

Submitting (`kiseki run`, locally or over SSH) creates the run directory in state
"queued" and calls `ensure_runner`, which starts a detached runner
(`python -m kiseki.runner <home>`) unless one already holds <home>/queue.lock. The runner
launches `python -m kiseki.worker train <run>` for the oldest queued run, waits for it and
repeats, then exits when nothing is queued. Runs outlive the terminal, the app and the SSH
session that started them.

The runner reads the machine's settings from <home>/machine.json (written by the
submitter), so a remote needs no config of its own: the environment for workers, and
where to keep the shared-server notice (`kiseki.notice`) while runs are active.
"""

import fcntl
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import msgspec

from kiseki import notice
from kiseki.machines import Contact
from kiseki.spec import RunSpec
from kiseki.store import Run, write_atomic

NOTICE_SECONDS = 600  # how often the runner refreshes the notice during a run
POLL_SECONDS = 5


class Profile(msgspec.Struct, kw_only=True):
    """What a runner needs to know about its machine; <home>/machine.json."""

    name: str = "local"
    env: dict[str, str] = {}
    notice: str | None = None
    contact: Contact | None = None


def read_profile(home: Path) -> Profile:
    path = home / "machine.json"
    return msgspec.json.decode(path.read_bytes(), type=Profile) if path.exists() else Profile()


def write_profile(home: Path, profile: Profile) -> None:
    home.mkdir(parents=True, exist_ok=True)
    write_atomic(home / "machine.json", msgspec.json.encode(profile))


def worker_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"kiseki.worker" in cmdline


def runs(home: Path) -> list[Run]:
    return Run.all(home / "runs")


def queued(home: Path) -> list[Run]:
    found = [(run.status(), run) for run in runs(home)]
    waiting = [(status.created_at, run.id, run) for status, run in found if status.state == "queued"]
    return [run for *_, run in sorted(waiting)]


def reconcile(home: Path) -> list[Run]:
    """Mark runs whose worker died without saying so as failed; return live running ones."""
    alive = []
    for run in runs(home):
        status = run.status()
        if status.state != "running":
            continue
        if worker_alive(status.pid):
            alive.append(run)
            continue
        error = "worker exited without recording an outcome (killed, crashed or rebooted)"
        run.update_status(state="failed", pid=None, error=error)
        run.event("failed", step=status.step, error=error)
    return alive


def runner_running(home: Path) -> bool:
    home.mkdir(parents=True, exist_ok=True)
    with (home / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def ensure_runner(home: Path) -> bool:
    """Start a detached runner unless one is running. Returns whether it started one."""
    if runner_running(home):
        return False
    log = (home / "queue.log").open("ab")
    env = dict(os.environ, KISEKI_HOME=str(home))
    subprocess.Popen(
        [sys.executable, "-m", "kiseki.runner", str(home)],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=env, close_fds=True,
    )
    return True


def submit(home: Path, run_id: str, spec: RunSpec, machine: str) -> Run:
    """Queue a run on this machine and make sure a runner will start it."""
    run = Run.create(run_id, spec, machine=machine, root=home / "runs")
    ensure_runner(home)
    return run


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


class Runner:
    def __init__(self, home: Path):
        self.home = home
        self.profile = read_profile(home)
        self.last_notice = 0.0

    def serve(self) -> int:
        lock = (self.home / "queue.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        write_atomic(self.home / "queue.pid", str(os.getpid()).encode())
        log(f"runner {os.getpid()} on {self.profile.name}")
        while True:
            for run in reconcile(self.home):  # a worker left by an earlier runner
                self.wait(run, None)
            pending = queued(self.home)
            if pending:
                self.start(pending[0])
                continue
            self.refresh_notice(force=True)
            # Release, then look once more: a run queued while we were deciding to stop
            # would otherwise wait for the next submission.
            fcntl.flock(lock, fcntl.LOCK_UN)
            if not queued(self.home):
                log("queue empty; exiting")
                return 0
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0

    def start(self, run: Run) -> None:
        self.profile = read_profile(self.home)
        env = dict(os.environ, KISEKI_HOME=str(self.home), **self.profile.env)
        log(f"starting {run.id}")
        with (run.path / "worker.log").open("ab") as out:
            process = subprocess.Popen(
                [sys.executable, "-m", "kiseki.worker", "train", str(run.path)],
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env,
            )
        self.wait(run, process)

    def wait(self, run: Run, process: subprocess.Popen | None) -> None:
        self.refresh_notice(force=True)
        while True:
            if process is not None:
                try:
                    code = process.wait(timeout=POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    pass
            else:
                time.sleep(POLL_SECONDS)
                if not worker_alive(run.status().pid):
                    code = None
                    break
            self.refresh_notice()
        status = run.status()
        if status.state in {"running", "queued"}:
            error = f"worker exited with code {code} without recording an outcome; see worker.log"
            run.update_status(state="failed", pid=None, error=error)
            run.event("failed", step=status.step, error=error)
        log(f"{run.id} {run.status().state}")

    def refresh_notice(self, force: bool = False) -> None:
        profile = self.profile
        if not profile.notice or not profile.contact:
            return
        if not force and time.monotonic() - self.last_notice < NOTICE_SECONDS:
            return
        self.last_notice = time.monotonic()
        try:
            notice.write(Path(profile.notice), self.notice_block(profile.contact))
        except OSError as error:
            log(f"could not update {profile.notice}: {error}")

    def notice_block(self, contact: Contact) -> str | None:
        now = datetime.now(UTC)
        statuses = [run.status() for run in runs(self.home)]
        running = [s for s in statuses if s.state == "running"]
        waiting = [s for s in statuses if s.state == "queued"]
        if not running and not waiting:
            return None
        remaining = 0.0
        for status in running + waiting:
            if status.eta_seconds is not None:
                remaining += status.eta_seconds
            else:  # not started yet: guess from the running run's pace
                rate = next((s.steps_per_second for s in running if s.steps_per_second), None)
                remaining += (status.iterations - status.step) / rate if rate else 3600.0
        starts = [datetime.fromisoformat(s.started_at) for s in running if s.started_at]
        start = min(starts) if starts else now
        end = notice.forecast_end(now, remaining)
        return notice.render(contact, start, end, len(running), len(waiting), now)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m kiseki.runner <kiseki home>", file=sys.stderr)
        return 2
    return Runner(Path(sys.argv[1])).serve()


if __name__ == "__main__":
    sys.exit(main())
