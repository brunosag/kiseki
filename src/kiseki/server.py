"""The app's HTTP server (`kiseki serve`): a JSON API over the run store and reports, an
event stream, and the built UI. It never imports JAX; training and reports run in worker
processes exactly as from the CLI (`kiseki.actions`).

Bound to 127.0.0.1. Every /api request carries the session token printed at start
(header `X-Kiseki-Token`, or `?token=` for EventSource and <img> URLs), and the Host must
be local, so web pages in a browser can neither read nor drive it.

  GET  /api/options                      datasets, models, optimizers (JSON Schema), analyses
  GET  /api/runs                         [{id, spec, status}]
  GET  /api/runs/{id}                    + summary, models, events, path
  GET  /api/runs/{id}/rows/{train|val}?offset=N
                                         {columns: {name: [values]}, offset}: rows past
                                         byte offset N, column-wise (uPlot's layout)
  GET  /api/runs/{id}/files/{path}       worker.log, models/*.safetensors, summary.json, ...
  POST /api/runs                         {overrides, grid, preset, machine} -> [{id, spec, status}]
  POST /api/runs/{id}/{stop|pause|resume}
  GET  /api/machines?fresh=1             machines with GPU state; a cache older than 30 s
                                         is returned while a new probe runs (`machines`
                                         event); fresh=1 waits for the probe
  GET  /api/reports                      [{id, spec, status}]
  GET  /api/reports/{id}                 {id, spec, status, report: report.json or null}
  GET  /api/reports/{id}/files/{path}    figures/*.png, arrays/*.npy
  POST /api/reports                      ReportSpec -> {id, spec, status}; builds in the background
  POST /api/query                        {sql} -> {columns, rows}
  GET  /api/events                       SSE: `run` and `report` events ({id, status}) when a
                                         status changes, `machines` when probed; `ping` every 15 s
"""

import asyncio
import contextlib
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import msgspec
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from kiseki import actions, paths, store
from kiseki.report import Report, ReportSpec
from kiseki.store import ACTIVE, Run

POLL_SECONDS = 1.0
REMOTE_REFRESH_SECONDS = 20.0
MACHINES_TTL = 30.0
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def ui_dir() -> Path:
    return paths.repo_root() / "ui" / "dist"


def ok(data: Any, status: int = 200) -> Response:
    return Response(msgspec.json.encode(data), status_code=status, media_type="application/json")


def error(message: str, status: int = 400) -> Response:
    return JSONResponse({"error": message}, status_code=status)


async def body(request: Request, type_: Any = dict) -> Any:
    return msgspec.json.decode(await request.body() or b"{}", type=type_)


def run_record(run: Run) -> dict[str, Any]:
    return {"id": run.id, "spec": run.spec(), "status": run.status()}


def report_record(report: Report) -> dict[str, Any]:
    return {"id": report.id, "spec": report.spec(), "status": report.status()}


def find_run(request: Request) -> Run:
    return Run.find(request.path_params["id"])


# -- state shared by handlers ------------------------------------------------------------


class State:
    def __init__(self, token: str):
        self.token = token
        self.subscribers: set[asyncio.Queue] = set()
        self.machines: list[dict[str, Any]] = []
        self.machines_at = 0.0
        self.probing: asyncio.Task | None = None
        self.reports: dict[str, Any] = {}  # report id -> Popen of builds started here

    def publish(self, event: str, data: Any) -> None:
        message = f"event: {event}\ndata: {msgspec.json.encode(data).decode()}\n\n"
        for queue in list(self.subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(message)


# -- handlers ----------------------------------------------------------------------------


async def options(request: Request) -> Response:
    return ok(actions.catalog())


async def list_runs(request: Request) -> Response:
    return ok([run_record(run) for run in Run.all()])


async def get_run(request: Request) -> Response:
    run = find_run(request)
    summary = run.path / "summary.json"
    return ok(
        run_record(run)
        | {
            "path": str(run.path),
            "models": run.model_names(),
            "summary": msgspec.json.decode(summary.read_bytes()) if summary.exists() else None,
            "events": run.rows("events")[0],
        }
    )


async def run_rows(request: Request) -> Response:
    run = find_run(request)
    kind = request.path_params["kind"]
    if kind not in {"train", "val"}:
        return error(f"unknown rows {kind!r}", 404)
    offset = int(request.query_params.get("offset", "0"))
    rows, next_offset = await asyncio.to_thread(
        store.read_ndjson, run.path / f"{kind}.ndjson", offset
    )
    keys = dict.fromkeys(key for row in rows for key in row)  # rows of older runs lack some
    columns = {key: [row.get(key) for row in rows] for key in keys}
    return ok({"columns": columns, "offset": next_offset})


class RunRequest(msgspec.Struct, kw_only=True):
    overrides: dict[str, Any] = {}
    grid: dict[str, list[Any]] = {}
    preset: str | None = None
    machine: str = "local"


async def create_runs(request: Request) -> Response:
    asked = await body(request, RunRequest)
    specs = actions.build_specs(asked.overrides, asked.grid, asked.preset)
    queued = await asyncio.to_thread(actions.queue, specs, asked.machine)
    return ok([run_record(run) for run in queued], 201)


async def control_run(request: Request) -> Response:
    run = find_run(request)
    results = await asyncio.to_thread(actions.control, request.path_params["action"], [run])
    return ok(results[0] if results else {"id": run.id, "result": "no change"})


def probe_machines(state: State) -> asyncio.Task:
    """The probe of every machine in flight, started unless one is; it publishes
    `machines` when done."""
    if state.probing is None or state.probing.done():

        async def probe():
            state.machines = await asyncio.to_thread(actions.machine_records)
            state.machines_at = time.monotonic()
            state.publish("machines", state.machines)

        state.probing = asyncio.create_task(probe())
    return state.probing


async def list_machines(request: Request) -> Response:
    """Answers from the cache when there is one: a stale cache is returned at once and
    the probe it starts arrives as a `machines` event. `fresh=1` waits for a probe."""
    state: State = request.app.state.kiseki
    if request.query_params.get("fresh") == "1" or not state.machines_at:
        await probe_machines(state)
    elif time.monotonic() - state.machines_at > MACHINES_TTL:
        probe_machines(state)
    return ok(state.machines)


def list_report_dirs() -> list[Report]:
    base = paths.reports_dir()
    if not base.exists():
        return []
    return [Report(p) for p in sorted(base.iterdir()) if (p / "spec.json").exists()]


async def list_reports(request: Request) -> Response:
    return ok([report_record(r) for r in list_report_dirs()])


async def get_report(request: Request) -> Response:
    report = Report.find(request.path_params["id"])
    document = report.path / "report.json"
    data = report_record(report)
    data["report"] = msgspec.json.decode(document.read_bytes()) if document.exists() else None
    return ok(data)


async def run_file(request: Request) -> Response:
    run = find_run(request)
    return file_within(run.path, request.path_params["path"])


async def report_file(request: Request) -> Response:
    report = Report.find(request.path_params["id"])
    return file_within(report.path, request.path_params["path"], cache=True)


def file_within(root: Path, relative: str, cache: bool = False) -> Response:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return error("not found", 404)
    binary = path.suffix in {".npy", ".safetensors"}
    media = (
        "application/octet-stream" if binary else ("text/plain" if path.suffix == ".log" else None)
    )
    headers = {"Cache-Control": "max-age=3600" if cache else "no-cache"}
    return FileResponse(path, media_type=media, headers=headers)


async def create_report(request: Request) -> Response:
    state: State = request.app.state.kiseki
    spec = await body(request, ReportSpec)
    report = await asyncio.to_thread(actions.create_report, spec)
    process = actions.start_report(report, detach=True)
    state.reports[report.id] = process

    def wait():
        actions.finish_report(report, process.wait())
        state.reports.pop(report.id, None)

    threading.Thread(target=wait, daemon=True).start()
    return ok(report_record(report), 201)


async def query(request: Request) -> Response:
    from kiseki.query import query as run_query

    asked = await body(request)
    columns, rows = await asyncio.to_thread(run_query, asked["sql"])
    return ok({"columns": columns, "rows": rows})


async def events(request: Request) -> Response:
    state: State = request.app.state.kiseki
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    state.subscribers.add(queue)

    async def stream():
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield "event: ping\ndata: {}\n\n"
        finally:
            state.subscribers.discard(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# -- background watchers -----------------------------------------------------------------


def status_files() -> dict[tuple[str, str], int]:
    """(kind, id) -> mtime_ns of each run's and report's status.json."""
    found = {}
    for kind, base in (("run", paths.runs_dir()), ("report", paths.reports_dir())):
        if not base.exists():
            continue
        for directory in base.iterdir():
            with contextlib.suppress(OSError):
                found[(kind, directory.name)] = (directory / "status.json").stat().st_mtime_ns
    return found


async def watch_statuses(state: State) -> None:
    seen = await asyncio.to_thread(status_files)
    while True:
        await asyncio.sleep(POLL_SECONDS)
        current = await asyncio.to_thread(status_files)
        for (kind, item), mtime in current.items():
            if seen.get((kind, item)) == mtime:
                continue
            with contextlib.suppress(OSError, msgspec.DecodeError, KeyError):
                if kind == "run":
                    state.publish(
                        "run", {"id": item, "status": Run(paths.runs_dir() / item).status()}
                    )
                else:
                    state.publish(
                        "report",
                        {"id": item, "status": Report(paths.reports_dir() / item).status()},
                    )
        seen = current


async def refresh_remotes() -> None:
    while True:
        runs = [
            r
            for r in Run.all()
            if r.status().machine != "local" and r.status().state in (*ACTIVE, "paused")
        ]
        if runs:
            await asyncio.to_thread(actions.refresh, runs)
        await asyncio.sleep(REMOTE_REFRESH_SECONDS)


# -- app ---------------------------------------------------------------------------------


def guard(app, token: str):
    """ASGI middleware: local Host only, and the session token on /api (streams pass through)."""

    async def guarded(scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope)
            host = request.headers.get("host", "").rsplit(":", 1)[0]
            if host not in LOCAL_HOSTS:
                return await error("forbidden host", 403)(scope, receive, send)
            if scope["path"].startswith("/api/"):
                given = request.headers.get("x-kiseki-token") or request.query_params.get(
                    "token", ""
                )
                if not secrets.compare_digest(given.encode(), token.encode()):
                    return await error("missing or wrong token", 401)(scope, receive, send)
        await app(scope, receive, send)

    return guarded


async def not_found(request: Request, failure: Exception) -> Response:
    return error(failure.args[0] if failure.args else str(failure), 404)


async def bad_request(request: Request, failure: Exception) -> Response:
    return error(str(failure), 400)


def spa(directory: Path):
    """The built UI, with index.html for client-side routes."""
    files = StaticFiles(directory=directory, html=True, check_dir=False)
    missing = Response(
        "The UI is not built: cd ui && npm install && npm run build\n",
        status_code=503,
        media_type="text/plain",
    )

    async def app(scope, receive, send):
        if not (directory / "index.html").exists():
            return await missing(scope, receive, send)
        path = directory / scope["path"].lstrip("/")
        if scope["path"] != "/" and not path.exists():
            scope = dict(scope, path="/")
        await files(scope, receive, send)

    return app


def create_app(token: str):
    state = State(token)
    tasks: list[asyncio.Task] = []

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        tasks.append(asyncio.create_task(watch_statuses(state)))
        tasks.append(asyncio.create_task(refresh_remotes()))
        probe_machines(state)  # SSH connections open while the user looks at runs
        yield
        for task in tasks:
            task.cancel()

    routes = [
        Route("/api/options", options),
        Route("/api/runs", list_runs),
        Route("/api/runs", create_runs, methods=["POST"]),
        Route("/api/runs/{id}", get_run),
        Route("/api/runs/{id}/rows/{kind}", run_rows),
        Route("/api/runs/{id}/files/{path:path}", run_file),
        Route("/api/runs/{id}/{action}", control_run, methods=["POST"]),
        Route("/api/machines", list_machines),
        Route("/api/reports", list_reports),
        Route("/api/reports", create_report, methods=["POST"]),
        Route("/api/reports/{id}", get_report),
        Route("/api/reports/{id}/files/{path:path}", report_file),
        Route("/api/query", query, methods=["POST"]),
        Route("/api/events", events),
        Mount("/", app=spa(ui_dir())),
    ]
    handlers = {
        KeyError: not_found,
        FileNotFoundError: not_found,
        ValueError: bad_request,
        msgspec.ValidationError: bad_request,
        msgspec.DecodeError: bad_request,
    }
    app = Starlette(routes=routes, lifespan=lifespan, exception_handlers=handlers)
    app.state.kiseki = state
    return guard(app, token)


def serve(port: int = 0, token: str | None = None) -> None:
    """Serve until interrupted; prints `kiseki: serving <url>` once listening."""
    import socket

    import uvicorn

    token = token or secrets.token_urlsafe(24)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}/?token={token}"
    config = uvicorn.Config(create_app(token), log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    print(f"kiseki: serving {url}", flush=True)
    server.run(sockets=[sock])
