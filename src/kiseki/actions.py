"""What the CLI and the app server do, in one place: queue runs (here or on a remote),
control them, mirror remote ones, and start reports. Callers only format the results.
"""

import itertools
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import msgspec

from kiseki import machines, optimizers, paths, runner
from kiseki.report import Report, ReportSpec
from kiseki.report import normalize as normalize_report
from kiseki.spec import ModelRef, RunSpec, from_overrides, new_run_id
from kiseki.store import ACTIVE, Run

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "iterations": 20,
        "validation_interval": 5,
        "checkpoint_interval": 10,
        "population_size": 32,
        "label": "smoke",
    },
}
CONTROL = ("stop", "pause", "resume")


def remote_for(name: str):
    from kiseki.remote import Remote

    return Remote(machines.get(name))


def profile_for(machine: machines.Machine) -> runner.Profile:
    contact = machines.contact() if machine.notice else None
    return runner.Profile(
        name=machine.name, env=machine.env, notice=machine.notice, contact=contact
    )


def local_gpu_awake() -> bool | None:
    """Whether a discrete NVIDIA GPU is powered up, read without waking it."""
    for device in Path("/sys/bus/pci/devices").glob("*"):
        try:
            if (device / "vendor").read_text().strip() == "0x10de" and (
                device / "class"
            ).read_text().startswith("0x03"):
                return (device / "power" / "runtime_status").read_text().strip() == "active"
        except OSError:
            continue
    return None


def machine_records() -> list[dict[str, Any]]:
    """Every machine with its state: GPUs and runner (one SSH round trip per remote)."""
    records = []
    for name, machine in machines.all_machines().items():
        record: dict[str, Any] = {
            "name": name,
            "ssh": machine.ssh,
            "notice": machine.notice,
            "description": machine.description,
        }
        if machine.is_local:
            record |= {
                "gpu_awake": local_gpu_awake(),
                "runner": runner.runner_running(paths.home()),
            }
        else:
            try:
                record |= remote_for(name).probe()
            except Exception as failure:  # unreachable: report it, keep the others
                record["error"] = str(failure)
        records.append(record)
    return records


# -- runs --------------------------------------------------------------------------------


def build_specs(
    overrides: dict[str, Any], grid: dict[str, list[Any]] | None = None, preset: str | None = None
) -> list[RunSpec]:
    """One spec per combination of `grid` values, each on top of preset and overrides.

    Keys are RunSpec fields or optimizer params (see `spec.from_overrides`); a preset's
    population_size is dropped for optimizers without one.
    """
    base = dict(PRESETS[preset]) if preset else {}
    base |= overrides
    axes = [[(key, value) for value in values] for key, values in (grid or {}).items()]
    specs = []
    for combo in itertools.product(*axes):
        settings = base | dict(combo)
        optimizer = settings.get("optimizer", RunSpec().optimizer)
        known = optimizers.params_type(optimizer).__struct_fields__
        from_preset = "population_size" not in overrides and "population_size" not in (grid or {})
        if (
            preset
            and from_preset
            and "population_size" in settings
            and "population_size" not in known
        ):
            del settings["population_size"]
        specs.append(from_overrides(RunSpec(), settings))
    return specs


def queue(specs: list[RunSpec], machine_name: str = "local") -> list[Run]:
    """Queue runs on a machine; remote runs are mirrored here right away."""
    machine = machines.get(machine_name)
    profile = profile_for(machine)
    if machine.is_local:
        home = paths.home()
        runner.write_profile(home, profile)
        return [runner.submit(home, new_run_id(spec), spec, machine.name) for spec in specs]
    remote = remote_for(machine.name)
    remote.deploy()
    for name in sorted({spec.dataset for spec in specs}):
        remote.ensure_dataset(name)
    remote.write_profile(profile)
    ids = []
    for spec in specs:
        run_id = new_run_id(spec)
        remote.submit(run_id, spec)
        ids.append(run_id)
    remote.pull(ids)
    return [Run.find(run_id) for run_id in ids]


def refresh(runs: list[Run]) -> list[str]:
    """Pull fresh files of active runs that live on remote machines; returns errors."""
    by_machine: dict[str, list[str]] = {}
    for run in runs:
        status = run.status()
        if status.machine != "local" and status.state in (*ACTIVE, "paused"):
            by_machine.setdefault(status.machine, []).append(run.id)
    errors = []
    for name, run_ids in by_machine.items():
        try:
            remote_for(name).pull(run_ids)
        except Exception as error:  # offline remote: callers show what they have
            errors.append(f"could not refresh {name}: {error}")
    return errors


def control(action: str, runs: list[Run]) -> list[dict[str, str]]:
    """stop, pause or resume runs on the machine each lives on."""
    if action not in CONTROL:
        raise ValueError(f"unknown action {action!r}; known: {', '.join(CONTROL)}")
    remote: dict[str, list[str]] = {}
    for run in runs:
        if run.status().machine != "local":
            remote.setdefault(run.status().machine, []).append(run.id)
    results = []
    for name, run_ids in remote.items():
        target = remote_for(name)
        results += msgspec.json.decode(target.control(action, run_ids))
        target.pull(run_ids)
    return results + [control_here(run, action) for run in runs if run.status().machine == "local"]


def control_here(run: Run, action: str) -> dict[str, str]:
    state = run.status().state
    if action == "resume":
        if state not in {"paused", "stopped", "failed"}:
            return {"id": run.id, "result": f"not resumable ({state})"}
        run.update_status(state="queued", error=None)
        run.event("requeued", previous=state)
        runner.ensure_runner(paths.home())
        return {"id": run.id, "result": "queued"}
    if state == "queued":
        run.update_status(state="stopped" if action == "stop" else "paused")
        run.event("stopped" if action == "stop" else "paused", step=run.status().step)
        return {"id": run.id, "result": run.status().state}
    if state != "running":
        return {"id": run.id, "result": f"already {state}"}
    run.request(action)  # type: ignore[arg-type]
    return {"id": run.id, "result": f"{action} requested"}


# -- reports -----------------------------------------------------------------------------


def resolve_model(text: str, offline: bool = False) -> str:
    """A full `run:name` ref; models of remote runs are pulled here when missing."""
    ref = ModelRef.parse(text)
    run = Run.find(ref.run_id)
    machine = run.status().machine
    if (
        not (run.models_dir / f"{ref.name}.safetensors").exists()
        and machine != "local"
        and not offline
    ):
        remote_for(machine).pull([run.id], models=True)
    return str(ModelRef(run.id, ref.name))


def create_report(spec: ReportSpec, offline: bool = False) -> Report:
    models = [resolve_model(text, offline) for text in spec.models]
    return Report.create(normalize_report(msgspec.structs.replace(spec, models=models)))


def start_report(report: Report, *, detach: bool = False) -> subprocess.Popen:
    """Build a report on this machine's GPU in a worker process (logs to worker.log)."""
    env = dict(os.environ, **machines.get("local").env)
    with (report.path / "worker.log").open("ab") as log:
        return subprocess.Popen(
            [sys.executable, "-m", "kiseki.worker", "report", str(report.path)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=detach,
        )


def finish_report(report: Report, code: int) -> None:
    """Record a failure the worker could not record itself (killed, crashed)."""
    status = report.status()
    if status.state != "completed" and not status.error:
        status.state, status.error = "failed", f"worker exited with code {code}; see worker.log"
        report.write_status(status)
