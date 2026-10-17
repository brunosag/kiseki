"""Run directories: the single source of truth about a run, on whichever machine has it.

<runs>/<run_id>/
  spec.json      frozen RunSpec (normalized)
  status.json    current RunStatus, replaced atomically by the worker or launcher
  events.ndjson  lifecycle events, append-only
  train.ndjson   one row per step: step, loss, accuracy, optimizer extras
  val.ndjson     one row per validation: step, loss, accuracy, elapsed, extras
  models/        best / final / step-<n> networks (.safetensors with a .json sidecar)
  state/         resume checkpoints; state/manifest.json names the committed one
  control        a pending request for the worker: "stop" or "pause"
  worker.log     the worker's stdout and stderr

Readers tolerate a truncated last line in .ndjson files (a worker killed mid-write).
"""

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import msgspec

from kiseki import paths
from kiseki.spec import RunSpec, encode

State = Literal["queued", "running", "paused", "stopped", "completed", "failed"]
ACTIVE: tuple[State, ...] = ("queued", "running")


class RunStatus(msgspec.Struct, kw_only=True):
    state: State
    iterations: int
    step: int = 0
    best_accuracy: float | None = None
    best_step: int | None = None
    test_accuracy: float | None = None
    last_accuracy: float | None = None
    last_loss: float | None = None
    steps_per_second: float | None = None
    eta_seconds: float | None = None
    elapsed_seconds: float = 0.0
    machine: str = "local"
    pid: int | None = None
    device: str | None = None
    chunk_size: int | None = None
    error: str | None = None
    created_at: str = ""
    started_at: str | None = None
    updated_at: str = ""
    finished_at: str | None = None


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


_decoder = msgspec.json.Decoder()
_encoder = msgspec.json.Encoder()


def read_ndjson(path: Path, start: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Rows from byte offset `start` to the last complete line, and the next offset."""
    if not path.exists():
        return [], start
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read()
    end = data.rfind(b"\n") + 1
    rows = [_decoder.decode(line) for line in data[:end].splitlines() if line.strip()]
    return rows, start + end


class Run:
    def __init__(self, path: Path):
        self.path = path
        self.id = path.name

    # -- creation and lookup -------------------------------------------------------------

    @classmethod
    def create(cls, run_id: str, spec: RunSpec, *, machine: str, root: Path | None = None) -> "Run":
        path = (root or paths.runs_dir()) / run_id
        path.mkdir(parents=True)
        run = cls(path)
        write_atomic(path / "spec.json", encode(spec))
        stamp = now()
        run.write_status(RunStatus(
            state="queued", iterations=spec.iterations, machine=machine,
            created_at=stamp, updated_at=stamp,
        ))
        run.event("created", machine=machine)
        return run

    @classmethod
    def find(cls, run_id: str, root: Path | None = None) -> "Run":
        base = root or paths.runs_dir()
        path = base / run_id
        if path.is_dir():
            return cls(path)
        matches = sorted(p for p in base.glob(f"*{run_id}*") if p.is_dir()) if base.exists() else []
        if len(matches) == 1:
            return cls(matches[0])
        if not matches:
            raise KeyError(f"no run matches {run_id!r}")
        raise KeyError(f"{run_id!r} matches {len(matches)} runs: {', '.join(m.name for m in matches[:5])}")

    @classmethod
    def all(cls, root: Path | None = None) -> list["Run"]:
        base = root or paths.runs_dir()
        if not base.exists():
            return []
        return [cls(p) for p in sorted(base.iterdir()) if (p / "spec.json").exists()]

    # -- spec and status -----------------------------------------------------------------

    def spec(self) -> RunSpec:
        return msgspec.json.decode((self.path / "spec.json").read_bytes(), type=RunSpec)

    def status(self) -> RunStatus:
        return msgspec.json.decode((self.path / "status.json").read_bytes(), type=RunStatus)

    def write_status(self, status: RunStatus) -> None:
        status.updated_at = now()
        write_atomic(self.path / "status.json", encode(status))

    def update_status(self, **changes: Any) -> RunStatus:
        status = msgspec.structs.replace(self.status(), **changes)
        self.write_status(status)
        return status

    # -- append-only logs ----------------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> None:
        self.append("events", [{"time": now(), "kind": kind, **fields}])

    def append(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        payload = b"".join(_encoder.encode(row) + b"\n" for row in rows)
        with (self.path / f"{name}.ndjson").open("ab") as handle:
            handle.write(payload)

    def rows(self, name: str, start: int = 0) -> tuple[list[dict[str, Any]], int]:
        return read_ndjson(self.path / f"{name}.ndjson", start)

    def iter_rows(self, name: str) -> Iterator[dict[str, Any]]:
        yield from self.rows(name)[0]

    # -- control requests ----------------------------------------------------------------

    def request(self, action: Literal["stop", "pause"]) -> None:
        write_atomic(self.path / "control", action.encode())

    def pending_request(self) -> str | None:
        path = self.path / "control"
        return path.read_text().strip() if path.exists() else None

    def clear_request(self) -> None:
        (self.path / "control").unlink(missing_ok=True)

    # -- artifacts -----------------------------------------------------------------------

    @property
    def models_dir(self) -> Path:
        return self.path / "models"

    @property
    def state_dir(self) -> Path:
        return self.path / "state"

    def model_names(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        return sorted(p.stem for p in self.models_dir.glob("*.safetensors"))
