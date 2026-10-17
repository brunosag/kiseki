"""Running on another machine over SSH. A remote mirrors this machine's layout under its
own kiseki home: code/ (this checkout's package, installed with uv into code/.venv),
runs/, a queue runner and machine.json; datasets go to its ~/.cache/kiseki/datasets.

Every operation is a short SSH command over one shared connection (ControlMaster), so
the laptop can sleep or go offline while remote runs continue. Remote runs are mirrored
into this machine's runs/ by `pull`, which copies small files by default (spec, status,
metrics, events, logs) and models when asked.
"""

import hashlib
import shlex
import subprocess
from pathlib import Path

import msgspec

from kiseki import paths
from kiseki.machines import Machine
from kiseki.runner import Profile
from kiseki.spec import RunSpec

SMALL_FILES = ("spec.json", "status.json", "summary.json", "*.ndjson", "worker.log")


class RemoteError(RuntimeError):
    pass


def _remote_path(path: str) -> str:
    """A remote path for shell commands: ~/x becomes "$HOME"/x, quoted."""
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def _rsync_path(path: str) -> str:
    """A remote path for rsync, relative to the remote home when it starts with ~/."""
    return path[2:] if path.startswith("~/") else path


class Remote:
    def __init__(self, machine: Machine):
        if machine.ssh is None:
            raise ValueError(f"{machine.name} is local")
        self.machine = machine
        self.host = machine.ssh
        control = paths.cache_dir() / "ssh"
        control.mkdir(parents=True, exist_ok=True)
        self.ssh_options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={control}/%C",
            "-o",
            "ControlPersist=15m",
            "-o",
            "ServerAliveInterval=30",
        ]
        self.home = machine.home.rstrip("/")
        self.code = f"{self.home}/code"
        self.python = f"{self.code}/.venv/bin/python"

    # -- primitives ------------------------------------------------------------------------

    def run(
        self, command: str, *, input: bytes | None = None, check: bool = True, timeout: float = 600
    ) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(
            ["ssh", *self.ssh_options, self.host, command],
            input=input,
            capture_output=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            raise RemoteError(
                f"{self.machine.name}: `{command}` exited {result.returncode}: "
                f"{result.stderr.decode(errors='replace').strip()[-2000:]}"
            )
        return result

    def rsync(self, *args: str) -> str:
        command = ["rsync", "-e", shlex.join(["ssh", *self.ssh_options]), *args]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RemoteError(f"{self.machine.name}: rsync failed: {result.stderr.strip()[-2000:]}")
        return result.stdout

    def target(self, path: str) -> str:
        return f"{self.host}:{_rsync_path(path)}"

    def kiseki(self, *args: str, input: bytes | None = None, timeout: float = 600) -> bytes:
        """Run the kiseki CLI on the remote and return its stdout."""
        command = f"{_remote_path(self.python)} -m kiseki.cli {shlex.join(args)}"
        env = f"KISEKI_HOME={_remote_path(self.home)}"
        return self.run(f"{env} {command}", input=input, timeout=timeout).stdout

    # -- setup -----------------------------------------------------------------------------

    def deploy(self) -> bool:
        """Copy this checkout's package to the remote and sync its environment.

        rsync makes repeat deploys cheap; uv runs only when the dependencies changed or
        the environment is missing. Returns whether anything changed.
        """
        root = paths.repo_root()
        self.run(f"mkdir -p {_remote_path(self.code)}")
        changes = self.rsync(
            "-a",
            "--delete",
            "--itemize-changes",
            "--exclude=__pycache__",
            "--exclude=.venv",
            "--include=/src/***",
            "--include=/pyproject.toml",
            "--include=/uv.lock",
            "--include=/README.md",
            "--exclude=*",
            f"{root}/",
            self.target(self.code) + "/",
        )
        dependencies = any(name in changes for name in ("pyproject.toml", "uv.lock"))
        has_env = self.run(f"test -x {_remote_path(self.python)}", check=False).returncode == 0
        if dependencies or not has_env:
            extras = " ".join(f"--extra {shlex.quote(extra)}" for extra in self.machine.extras)
            self.run(
                f"cd {_remote_path(self.code)} && "
                f'PATH="$HOME/.local/bin:$PATH" uv sync --frozen --no-dev {extras}',
                timeout=3600,
            )
        return bool(changes.strip()) or dependencies or not has_env

    def ensure_dataset(self, name: str, *, fetch_timeout: float = 900) -> str:
        """Make the dataset available remotely. The remote downloads it from the source
        (hashes pinned, so the result is identical) unless it already has it; if that
        fails, this machine's parsed copy is sent compressed. Returns how it got there."""
        try:
            out = self.kiseki("datasets", "--fetch", "--json", name, timeout=fetch_timeout)
            if msgspec.json.decode(out)[0]["cached"]:
                return "remote"
        except RemoteError, subprocess.TimeoutExpired:
            pass
        local = paths.datasets_dir() / name
        if not (local / "stats.json").exists():
            raise RemoteError(f"{name}: the remote could not download it and it is not cached here")
        remote = f"~/.cache/kiseki/datasets/{name}"
        self.run(f"mkdir -p {_remote_path(remote)}")
        self.rsync("-az", "--partial", "--exclude=raw", f"{local}/", self.target(remote) + "/")
        return "pushed"

    def write_profile(self, profile: Profile) -> None:
        self.run(
            f"mkdir -p {_remote_path(self.home)} && cat > {_remote_path(self.home)}/machine.json",
            input=msgspec.json.encode(profile),
        )

    # -- runs ------------------------------------------------------------------------------

    def submit(self, run_id: str, spec: RunSpec) -> None:
        self.kiseki(
            "submit",
            "--id",
            run_id,
            "--machine",
            self.machine.name,
            input=msgspec.json.encode(spec),
        )

    def control(self, action: str, run_ids: list[str]) -> bytes:
        return self.kiseki(action, "--here", "--json", *run_ids)

    def pull(self, run_ids: list[str] | None = None, *, models: bool = False) -> Path:
        """Mirror remote runs into this machine's runs/<id> (all runs when none named)."""
        local = paths.runs_dir()
        local.mkdir(parents=True, exist_ok=True)
        filters = (
            ["--include=*/"] if not run_ids else [f"--include=/{run_id}/" for run_id in run_ids]
        )
        files = [f"--include=/*/{pattern}" for pattern in SMALL_FILES]
        if models:
            files += ["--include=/*/models/", "--include=/*/models/*.safetensors"]
        self.rsync(
            "-a",
            "--prune-empty-dirs",
            *filters,
            *files,
            "--exclude=*",
            self.target(f"{self.home}/runs") + "/",
            f"{local}/",
        )
        return local

    def probe(self) -> dict:
        """GPU state and queue of the remote, from one SSH round trip."""
        script = (
            "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu "
            "--format=csv,noheader,nounits 2>/dev/null; echo '--'; "
            f"cat {_remote_path(self.home)}/queue.pid 2>/dev/null; echo '--'; "
            f"test -x {_remote_path(self.python)} && echo deployed; true"
        )
        out = self.run(script, timeout=30).stdout.decode()
        gpus_text, pid_text, deployed = (part.strip() for part in out.split("--", 2))
        gpus = []
        for line in filter(None, gpus_text.splitlines()):
            name, used, total, util = (field.strip() for field in line.split(","))
            gpus.append(
                {
                    "name": name,
                    "memory_used_mib": int(used),
                    "memory_total_mib": int(total),
                    "utilization": int(util),
                }
            )
        return {
            "gpus": gpus,
            "runner_pid": int(pid_text) if pid_text.isdigit() else None,
            "deployed": deployed == "deployed",
        }


def code_hash() -> str:
    """Hash of the package as deployed: source files, pyproject and lock."""
    root = paths.repo_root()
    digest = hashlib.sha256()
    for path in sorted([*root.glob("src/**/*.py"), root / "pyproject.toml", root / "uv.lock"]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]
