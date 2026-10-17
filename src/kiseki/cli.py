"""The `kiseki` command line. Every command prints compact text, or JSON with --json.

  kiseki options                       datasets, models, optimizers (with params), machines
  kiseki run -o leea -d mnist -s population_size=500 [--on lugia] [--watch]
  kiseki ls | show RUN | watch RUN...  runs on this machine and mirrored remote runs
  kiseki stop | pause | resume RUN...
  kiseki compare RUN... | plot RUN... -o out.png | query "SQL"
  kiseki machines | pull [--on M] [--models]

RUN is a run id or any unique part of one. `-s key=a,b,c` queues one run per value
(several -s make a grid). Presets: smoke (seconds; checks that a configuration runs).
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import msgspec

from kiseki import datasets, machines, models, optimizers, paths, runner
from kiseki.spec import RunSpec, from_overrides, new_run_id, normalize
from kiseki.store import ACTIVE, Run

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {"iterations": 20, "validation_interval": 5, "checkpoint_interval": 10,
              "population_size": 32, "label": "smoke"},
}
TERMINAL = ("completed", "stopped", "failed", "paused")


# -- output ------------------------------------------------------------------------------


def emit(args: argparse.Namespace, data: Any, text: str) -> None:
    if args.json:
        sys.stdout.write(msgspec.json.encode(data).decode() + "\n")
    elif text:
        print(text)


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}"


def table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(header, widths, strict=True)).rstrip()]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows]
    return "\n".join(lines)


def run_record(run: Run) -> dict[str, Any]:
    return {"id": run.id, "spec": run.spec(), "status": run.status()}


def run_row(run: Run) -> list[str]:
    spec, status = run.spec(), run.status()
    return [
        run.id, status.state, f"{status.step}/{spec.iterations}", percent(status.best_accuracy),
        percent(status.test_accuracy),
        f"{status.steps_per_second:.1f}" if status.steps_per_second else "-",
        duration(status.eta_seconds) if status.state == "running" else duration(status.elapsed_seconds),
        status.machine, spec.label,
    ]


RUN_HEADER = ["id", "state", "step", "best%", "test%", "steps/s", "eta|time", "machine", "label"]


# -- resolving runs ----------------------------------------------------------------------


def find_runs(ids: list[str]) -> list[Run]:
    return [Run.find(run_id) for run_id in ids]


def refresh(runs: list[Run]) -> None:
    """Pull fresh files of active runs that live on remote machines."""
    by_machine: dict[str, list[str]] = {}
    for run in runs:
        status = run.status()
        if status.machine != "local" and status.state in (*ACTIVE, "paused"):
            by_machine.setdefault(status.machine, []).append(run.id)
    for name, run_ids in by_machine.items():
        try:
            remote_for(name).pull(run_ids)
        except Exception as error:  # offline remote: show what we have
            print(f"kiseki: could not refresh {name}: {error}", file=sys.stderr)


def remote_for(name: str):
    from kiseki.remote import Remote

    return Remote(machines.get(name))


def profile_for(machine: machines.Machine) -> runner.Profile:
    contact = machines.contact() if machine.notice else None
    return runner.Profile(name=machine.name, env=machine.env, notice=machine.notice, contact=contact)


# -- commands ----------------------------------------------------------------------------


def parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def expand_settings(settings: list[str]) -> list[dict[str, Any]]:
    axes: list[list[tuple[str, Any]]] = []
    for setting in settings:
        key, sep, raw = setting.partition("=")
        if not sep:
            raise SystemExit(f"kiseki: -s expects key=value, got {setting!r}")
        values = raw.split(",") if "," in raw and not raw.startswith("[") else [raw]
        axes.append([(key, parse_value(v)) for v in values])
    return [dict(combo) for combo in itertools.product(*axes)] if axes else [{}]


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    base: dict[str, Any] = {}
    if args.preset:
        base |= PRESETS[args.preset]
    settings = list(args.set)
    for flag in ("dataset", "optimizer", "model", "label"):
        value = getattr(args, flag)
        if value and "," in value and flag != "label":
            settings.append(f"{flag}={value}")  # -o sgd,leea: one run each
        elif value:
            base[flag] = value
    specs = []
    for combo in expand_settings(settings):
        overrides = base | combo
        optimizer = overrides.get("optimizer", RunSpec().optimizer)
        known = optimizers.params_type(optimizer).__struct_fields__
        if args.preset and "population_size" in overrides and "population_size" not in known:
            del overrides["population_size"]
        specs.append(from_overrides(RunSpec(), overrides))
    return specs


def cmd_run(args: argparse.Namespace) -> int:
    specs = build_specs(args)
    machine = machines.get(args.on)
    profile = profile_for(machine)
    queued: list[Run] = []
    if machine.is_local:
        home = paths.home()
        runner.write_profile(home, profile)
        for spec in specs:
            queued.append(runner.submit(home, new_run_id(spec), spec, machine.name))
    else:
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
        queued = find_runs(ids)
    emit(args, [run_record(run) for run in queued],
         "\n".join(f"queued {run.id} on {machine.name}" for run in queued))
    if args.watch:
        return watch(queued, args)
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    """Internal: queue a run on this machine from a spec on stdin (used over SSH)."""
    spec = normalize(msgspec.json.decode(sys.stdin.buffer.read(), type=RunSpec))
    run = runner.submit(paths.home(), args.id, spec, args.machine)
    emit(args, run_record(run), run.id)
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    runs = Run.all()
    if not args.offline:
        refresh(runs)
    if args.state:
        runs = [r for r in runs if r.status().state in args.state.split(",")]
    if args.on:
        runs = [r for r in runs if r.status().machine == args.on]
    runs = runs[-args.limit:] if args.limit else runs
    emit(args, [run_record(r) for r in runs],
         table([run_row(r) for r in runs], RUN_HEADER) if runs else "no runs")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    (run,) = find_runs([args.run])
    if not args.offline:
        refresh([run])
    spec, status = run.spec(), run.status()
    val = list(run.iter_rows("val"))
    summary_path = run.path / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    data = {"id": run.id, "path": str(run.path), "spec": spec, "status": status,
            "summary": summary, "models": run.model_names(), "val_tail": val[-5:]}
    lines = [
        f"{run.id}  {status.state} on {status.machine}  step {status.step}/{spec.iterations}",
        f"  {spec.dataset}/{spec.model}/{spec.optimizer}  seed {spec.seed}  batch {spec.batch_size}"
        f"  params {json.dumps(spec.params, separators=(',', ':'))}",
        f"  best val {percent(status.best_accuracy)}% at step {status.best_step}"
        f"  test {percent(status.test_accuracy)}%  last val {percent(status.last_accuracy)}%",
        f"  {status.steps_per_second or 0:.2f} steps/s  elapsed {duration(status.elapsed_seconds)}"
        f"  eta {duration(status.eta_seconds)}  chunk {status.chunk_size}",
        f"  val curve {sparkline([r['val_accuracy'] for r in val])}" if val else "  no validations yet",
        f"  models {', '.join(run.model_names()) or '-'}  ({run.path})",
    ]
    if status.error:
        lines.append(f"  error: {status.error}")
    emit(args, data, "\n".join(lines))
    return 0


def sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        values = [values[min(len(values) - 1, int((i + 1) * step) - 1)] for i in range(width)]
    low, high = min(values), max(values)
    bars = "▁▂▃▄▅▆▇█"
    span = (high - low) or 1.0
    line = "".join(bars[int((v - low) / span * (len(bars) - 1))] for v in values)
    return f"{line} {percent(low)}→{percent(values[-1])}%"


def watch(runs: list[Run], args: argparse.Namespace) -> int:
    interval = getattr(args, "interval", 10.0)
    last: dict[str, tuple] = {}
    while True:
        refresh(runs)
        statuses = {run.id: run.status() for run in runs}
        for run in runs:
            status = statuses[run.id]
            key = (status.state, status.step)
            if last.get(run.id) == key:
                continue
            last[run.id] = key
            if args.json:
                emit(args, {"id": run.id, "status": status}, "")
            else:
                print(f"{time.strftime('%H:%M:%S')} {run.id} {status.state} "
                      f"{status.step}/{status.iterations} val {percent(status.last_accuracy)} "
                      f"best {percent(status.best_accuracy)} "
                      f"{status.steps_per_second or 0:.2f}/s eta {duration(status.eta_seconds)}",
                      flush=True)
        if all(s.state in TERMINAL for s in statuses.values()):
            return 1 if any(s.state == "failed" for s in statuses.values()) else 0
        time.sleep(interval)


def cmd_watch(args: argparse.Namespace) -> int:
    return watch(find_runs(args.runs), args)


def cmd_control(args: argparse.Namespace) -> int:
    """stop, pause or resume, here or on the machine each run lives on."""
    if args.here:
        runs = [Run.find(run_id, paths.runs_dir()) for run_id in args.runs]
    else:
        runs = find_runs(args.runs)
        remote: dict[str, list[str]] = {}
        for run in runs:
            if run.status().machine != "local":
                remote.setdefault(run.status().machine, []).append(run.id)
        for name, run_ids in remote.items():
            target = remote_for(name)
            target.control(args.command, run_ids)
            target.pull(run_ids)
        runs = [r for r in runs if r.status().machine == "local"]
    results = [control_here(run, args.command) for run in runs]
    emit(args, results, "\n".join(f"{r['id']}: {r['result']}" for r in results))
    return 0


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


def cmd_compare(args: argparse.Namespace) -> int:
    runs = find_runs(args.runs)
    refresh(runs)
    rows, data = [], []
    for run in runs:
        spec, status = run.spec(), run.status()
        val = list(run.iter_rows("val"))
        thresholds = {t: next((r for r in val if r["val_accuracy"] >= t), None) for t in args.at}
        data.append({"id": run.id, "optimizer": spec.optimizer, "params": spec.params,
                     "best_accuracy": status.best_accuracy, "test_accuracy": status.test_accuracy,
                     "steps": status.step, "elapsed_seconds": status.elapsed_seconds,
                     "reached": {str(t): r and {"step": r["step"], "elapsed": r["elapsed"]}
                                 for t, r in thresholds.items()}})
        rows.append([run.id, spec.optimizer, percent(status.best_accuracy),
                     percent(status.test_accuracy), str(status.step),
                     duration(status.elapsed_seconds),
                     *[f"{r['step']} ({duration(r['elapsed'])})" if r else "-" for r in thresholds.values()]])
    header = ["run", "optimizer", "best%", "test%", "steps", "time",
              *[f"≥{100 * t:g}%" for t in args.at]]
    emit(args, data, table(rows, header))
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from kiseki import plots

    runs = find_runs(args.runs)
    refresh(runs)
    path = plots.curves(runs, Path(args.output), metric=args.metric, x=args.x, log_x=args.log_x)
    emit(args, {"path": str(path)}, str(path))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from kiseki.query import query

    columns, rows = query(args.sql)
    if args.json:
        emit(args, [dict(zip(columns, row, strict=True)) for row in rows], "")
    else:
        print(table([[f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
                     for row in rows], columns))
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    names = [args.on] if args.on else [n for n, m in machines.all_machines().items() if not m.is_local]
    for name in names:
        remote_for(name).pull(args.runs or None, models=args.models)
    emit(args, {"machines": names}, f"pulled from {', '.join(names) or 'no remotes'}")
    return 0


def cmd_machines(args: argparse.Namespace) -> int:
    rows, data = [], []
    for name, machine in machines.all_machines().items():
        record: dict[str, Any] = {"name": name, "ssh": machine.ssh, "notice": machine.notice}
        if machine.is_local:
            record |= {"gpu_awake": local_gpu_awake(), "runner": runner.runner_running(paths.home())}
            gpu = "awake" if record["gpu_awake"] else "asleep"
        else:
            try:
                record |= remote_for(name).probe()
                gpu = ", ".join(f"{g['name']} {g['utilization']}% "
                                f"{g['memory_used_mib']}/{g['memory_total_mib']}MiB"
                                for g in record["gpus"]) or "none"
            except Exception as error:
                record["error"] = str(error)
                gpu = f"unreachable: {str(error)[:60]}"
        data.append(record)
        rows.append([name, machine.ssh or "-", gpu])
    emit(args, data, table(rows, ["machine", "ssh", "gpu"]))
    return 0


def local_gpu_awake() -> bool | None:
    """Whether a discrete NVIDIA GPU is powered up, read without waking it."""
    for device in Path("/sys/bus/pci/devices").glob("*"):
        try:
            if (device / "vendor").read_text().strip() == "0x10de" and \
                    (device / "class").read_text().startswith("0x03"):
                return (device / "power" / "runtime_status").read_text().strip() == "active"
        except OSError:
            continue
    return None


def cmd_datasets(args: argparse.Namespace) -> int:
    names = args.names or list(datasets.NAMES)
    records = []
    for name in names:
        info = datasets.info(name)
        if args.fetch and not datasets.is_cached(name):
            datasets.load(name)
        records.append({"name": name, "title": info.title, "cached": datasets.is_cached(name),
                        "image_shape": info.image_shape, "classes": len(info.class_names),
                        "default_model": info.default_model})
    emit(args, records, table(
        [[r["name"], r["title"], "x".join(map(str, r["image_shape"])), r["default_model"],
          "yes" if r["cached"] else "no"] for r in records],
        ["dataset", "title", "shape", "model", "cached"]))
    return 0


def cmd_options(args: argparse.Namespace) -> int:
    data = {
        "datasets": {n: msgspec.to_builtins(datasets.info(n), order="deterministic")
                     for n in datasets.NAMES},
        "models": list(models.NAMES),
        "optimizers": {n: {"summary": optimizers.summary(n),
                           "params": msgspec.json.schema(optimizers.params_type(n))}
                       for n in optimizers.NAMES},
        "run": msgspec.json.schema(RunSpec),
        "presets": PRESETS,
        "machines": list(machines.all_machines()),
    }
    lines = ["datasets: " + ", ".join(datasets.NAMES), "models: " + ", ".join(models.NAMES)]
    for name in optimizers.NAMES:
        defaults = msgspec.to_builtins(optimizers.params_type(name)())
        lines.append(f"{name}: {optimizers.summary(name)}")
        lines.append("    " + " ".join(f"{k}={v}" for k, v in defaults.items()))
    lines.append("run: " + " ".join(f"{k}={v}" for k, v in msgspec.to_builtins(RunSpec()).items()
                                    if k != "params"))
    lines.append("machines: " + ", ".join(machines.all_machines()))
    emit(args, data, "\n".join(lines))
    return 0


# -- parser ------------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="kiseki", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine-readable output")
    common.add_argument("--offline", action="store_true", help="don't refresh remote runs")
    sub = root.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="queue runs")
    run.add_argument("-d", "--dataset")
    run.add_argument("-o", "--optimizer")
    run.add_argument("-m", "--model")
    run.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE[,VALUE]")
    run.add_argument("--preset", choices=sorted(PRESETS))
    run.add_argument("--label")
    run.add_argument("--on", default="local", metavar="MACHINE")
    run.add_argument("--watch", action="store_true")
    run.add_argument("--interval", type=float, default=10.0)
    run.set_defaults(handler=cmd_run)

    submit = sub.add_parser("submit", parents=[common], help=argparse.SUPPRESS)
    submit.add_argument("--id", required=True)
    submit.add_argument("--machine", required=True)
    submit.set_defaults(handler=cmd_submit)

    ls = sub.add_parser("ls", parents=[common], help="list runs")
    ls.add_argument("--state", help="comma-separated states")
    ls.add_argument("--on", metavar="MACHINE")
    ls.add_argument("-n", "--limit", type=int, default=30)
    ls.set_defaults(handler=cmd_ls)

    show = sub.add_parser("show", parents=[common], help="one run in detail")
    show.add_argument("run")
    show.set_defaults(handler=cmd_show)

    watch_cmd = sub.add_parser("watch", parents=[common], help="follow runs until they end")
    watch_cmd.add_argument("runs", nargs="+")
    watch_cmd.add_argument("--interval", type=float, default=10.0)
    watch_cmd.set_defaults(handler=cmd_watch)

    for action in ("stop", "pause", "resume"):
        control = sub.add_parser(action, parents=[common], help=f"{action} runs")
        control.add_argument("runs", nargs="+")
        control.add_argument("--here", action="store_true", help=argparse.SUPPRESS)
        control.set_defaults(handler=cmd_control)

    compare = sub.add_parser("compare", parents=[common], help="runs side by side")
    compare.add_argument("runs", nargs="+")
    compare.add_argument("--at", type=float, nargs="*", default=[0.9, 0.95, 0.98],
                         help="report when validation accuracy first reached these")
    compare.set_defaults(handler=cmd_compare)

    plot = sub.add_parser("plot", parents=[common], help="curves to a PNG")
    plot.add_argument("runs", nargs="+")
    plot.add_argument("-o", "--output", default="kiseki-plot.png")
    plot.add_argument("--metric", default="val_accuracy")
    plot.add_argument("--x", choices=["step", "elapsed"], default="step")
    plot.add_argument("--log-x", action="store_true")
    plot.set_defaults(handler=cmd_plot)

    query = sub.add_parser("query", parents=[common], help="SQL over runs (see kiseki.query)")
    query.add_argument("sql")
    query.set_defaults(handler=cmd_query)

    pull = sub.add_parser("pull", parents=[common], help="mirror remote runs here")
    pull.add_argument("runs", nargs="*")
    pull.add_argument("--on", metavar="MACHINE")
    pull.add_argument("--models", action="store_true", help="also copy saved networks")
    pull.set_defaults(handler=cmd_pull)

    data = sub.add_parser("datasets", parents=[common], help="datasets and whether cached")
    data.add_argument("names", nargs="*")
    data.add_argument("--fetch", action="store_true", help="download and parse missing ones")
    data.set_defaults(handler=cmd_datasets)

    sub.add_parser("machines", parents=[common], help="machines and their GPUs").set_defaults(
        handler=cmd_machines)
    sub.add_parser("options", parents=[common], help="what can be run").set_defaults(
        handler=cmd_options)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.handler(args)
    except (KeyError, ValueError, msgspec.ValidationError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        print(f"kiseki: {message}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(main())
