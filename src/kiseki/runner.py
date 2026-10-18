"""A home's run queue: one runner process starts queued runs one at a time.

Submitting (`kiseki run`, locally or over SSH) creates the run directory in state
"queued" and calls `ensure_runner`, which starts a detached runner
(`python -m kiseki.runner <home>`) unless one already holds <home>/queue.lock. The runner
launches `python -m kiseki.worker train <run>` for the oldest queued run, waits for it and
repeats, then exits when nothing is queued. Runs outlive the terminal, the app and the SSH
session that started them.

A machine runs one run at a time even with several homes on it (its own, the one remotes
use, scratch homes): the runner takes the machine lock (`paths.machine_lock_file`) before
starting a worker, and the worker inherits it, so it stays taken while the worker lives.
A runner that finds it taken waits, its runs still queued.

The runner reads the machine's settings from <home>/machine.json (written by the
submitter), so a remote needs no config of its own: the environment for workers, and
where to keep the shared-server notice (`kiseki.notice`) while runs are active. The notice
is written before each run starts, and a run fails instead of starting when it can't be.
A started run is marked running with its worker's pid at once, so the queue never starts
it twice.

While a worker runs, the runner keeps the machine awake (systemd-inhibit, where the system
has it): a laptop that sleeps mid-run can lose its CUDA context and leave the worker hung
on wake. A worker whose status file stops changing for STALL_SECONDS of awake time is
killed and its run requeued, to resume from its last checkpoint (failed after
MAX_STALLS).
"""

import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

import msgspec

from kiseki import notice, paths
from kiseki.machines import Contact
from kiseki.spec import RunSpec
from kiseki.store import Run, RunStatus, write_atomic

NOTICE_SECONDS = 600  # how often the runner refreshes the notice during a run
POLL_SECONDS = 5
STALL_SECONDS = 900.0  # awake time a live worker may leave status.json unchanged
MAX_STALLS = 3  # stalls after which a run fails instead of being requeued


class NoticeError(Exception):
    """The machine's notice could not be written; runs there must not start."""


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


def owned(home: Path) -> list[tuple[RunStatus, Run]]:
    """This machine's runs and their status. Mirrors of other machines' runs are theirs to
    start and reconcile."""
    name = read_profile(home).name
    found = [(run.status(), run) for run in Run.all(home / "runs")]
    return [(status, run) for status, run in found if status.machine == name]


def queued(home: Path) -> list[Run]:
    waiting = [(s.created_at, run.id, run) for s, run in owned(home) if s.state == "queued"]
    return [run for *_, run in sorted(waiting)]


def reconcile(home: Path) -> list[Run]:
    """Mark runs whose worker died without saying so as failed; return live running ones."""
    alive = []
    for status, run in owned(home):
        if status.state != "running":
            continue
        if worker_alive(status.pid):
            alive.append(run)
            continue
        run.fail("worker exited without recording an outcome (killed, crashed or rebooted)")
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
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        close_fds=True,
    )
    return True


def submit(home: Path, run_id: str, spec: RunSpec, machine: str) -> Run:
    """Queue a run on this machine and make sure a runner will start it."""
    run = Run.create(run_id, spec, machine=machine, root=home / "runs")
    ensure_runner(home)
    return run


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


@contextmanager
def machine_lock() -> Iterator[IO[str]]:
    """This machine's run lock, waiting while another home's worker holds it."""
    path = paths.machine_lock_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("waiting for another kiseki run on this machine")
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield lock


@contextmanager
def inhibit_sleep(why: str) -> Iterator[None]:
    """Keep the machine from sleeping or idling into suspend, where systemd-inhibit exists.
    The inhibitor follows the runner's pid, so it ends with the runner even if killed."""
    tool = shutil.which("systemd-inhibit")
    process = None
    if tool:
        command = [tool, "--what=sleep:idle", "--who=kiseki", f"--why={why}", "--mode=block"]
        command += ["tail", f"--pid={os.getpid()}", "-f", "/dev/null"]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            log(f"could not inhibit sleep: {error}")
    try:
        yield
    finally:
        if process is not None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            process.wait()


class StallWatch:
    """Awake seconds since a file last changed. Monotonic time stops during suspend, so a
    machine that slept is not taken for a stalled worker."""

    def __init__(self, path: Path):
        self.path = path
        self.mtime = self.read()
        self.changed = time.monotonic()

    def read(self) -> int:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return 0

    def idle(self) -> float:
        mtime = self.read()
        if mtime != self.mtime:
            self.mtime, self.changed = mtime, time.monotonic()
        return time.monotonic() - self.changed


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
            if queued(self.home):
                with machine_lock() as held:
                    pending = queued(self.home)  # a run may have been stopped while waiting
                    if pending:
                        self.start(pending[0], held)
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

    def start(self, run: Run, lock: IO[str]) -> None:
        """Start a worker for a queued run, passing it the machine `lock` to hold."""
        self.profile = read_profile(self.home)
        try:
            self.write_notice()  # the server's rule: announce before using it
        except NoticeError as error:
            log(f"not starting {run.id}: {error}")
            run.fail(str(error))
            return
        env = dict(os.environ, KISEKI_HOME=str(self.home), **self.profile.env)
        log(f"starting {run.id}")
        with (run.path / "worker.log").open("ab") as out:
            process = subprocess.Popen(
                [sys.executable, "-m", "kiseki.worker", "train", str(run.path)],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                pass_fds=(lock.fileno(),),
            )
        # Owned from here, while the worker still loads data: stop and pause become
        # requests it honours, and a later runner waits for it instead of starting another.
        run.update_status(state="running", pid=process.pid)
        self.wait(run, process)

    def wait(self, run: Run, process: subprocess.Popen | None) -> None:
        self.refresh_notice(force=True)
        watch = StallWatch(run.path / "status.json")
        code = None
        with inhibit_sleep(f"kiseki run {run.id}"):
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
                        break
                if watch.idle() > STALL_SECONDS:
                    self.stalled(run, process)
                    return
                self.refresh_notice()
        if run.status().state in {"running", "queued"}:
            run.fail(f"worker exited with code {code} without recording an outcome; see worker.log")
        log(f"{run.id} {run.status().state}")

    def stalled(self, run: Run, process: subprocess.Popen | None) -> None:
        """Kill a hung worker; requeue its run to resume from its checkpoint."""
        status = run.status()
        if status.pid:
            with suppress(ProcessLookupError):
                os.kill(status.pid, signal.SIGKILL)
        if process is not None:
            process.wait()
        else:
            while worker_alive(status.pid):
                time.sleep(0.1)
        stalls = 1 + sum(1 for row in run.iter_rows("events") if row.get("kind") == "stalled")
        message = f"worker stalled {STALL_SECONDS:.0f} s at step {status.step}"
        if stalls >= MAX_STALLS:
            run.fail(f"{message}, {stalls} times")
        else:
            run.update_status(state="queued", pid=None)
            run.event("stalled", step=status.step, action="requeued")
        log(f"{run.id} {message}; {run.status().state}")

    def refresh_notice(self, force: bool = False) -> None:
        """Keep the notice current. A failed refresh is only logged: the notice written
        before the run started still stands."""
        if not force and time.monotonic() - self.last_notice < NOTICE_SECONDS:
            return
        try:
            self.write_notice()
        except NoticeError as error:
            log(str(error))

    def write_notice(self) -> None:
        profile = self.profile
        if not profile.notice:
            return
        if not profile.contact:
            raise NoticeError(f"{profile.name} needs a contact for {profile.notice}")
        self.last_notice = time.monotonic()
        try:
            notice.write(Path(profile.notice), self.notice_block(profile.contact))
        except OSError as error:
            raise NoticeError(f"could not write the notice {profile.notice}: {error}") from error

    def notice_block(self, contact: Contact) -> str | None:
        now = datetime.now(UTC)
        statuses = [status for status, _ in owned(self.home)]
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
