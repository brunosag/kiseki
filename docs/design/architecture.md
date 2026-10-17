# Architecture

How kiseki is split into processes and where state lives: a light Python core (CLI, run
store, queue runners, remotes, app server), JAX worker processes for training and
reports, and a React UI in an Electron window.

## Two languages: Python with JAX, and a React UI in Electron

Training and analysis run in JAX; everything else is plain Python; the UI is React +
TypeScript in a thin Electron shell that talks to `kiseki serve`.

- **Why:** the app should be "optimized for" this machine with "a great focus on
  performance" (PREFERENCES.md). Scoring one generation of 1000 networks on a 512-image
  batch on the RTX 3050 (2026-10-17, scratch benchmarks): MNIST fp32 1250 ms in the old
  PyTorch vmap code, 661 ms in hand-tuned PyTorch with torch.compile, 471 ms in plain JAX
  (328 ms bf16); CIFAR-10 12.9 s / 7.9 s fp32, 3.5 s bf16 in JAX. Electron is Chromium, so
  agents test what the user sees, and it draws on the Intel iGPU, leaving the 4 GB of VRAM
  to experiments. Agent's design; a Codex critic (GPT-6.1 Sol) found no stack that clearly
  beats it.
- **Rejected:** Rust core + Tauri (a third language, slower agent builds, WebKit on Linux
  differs from what agents check in Chromium); native GUIs (egui, GPUI, Qt: no page
  structure for agents to inspect); PyTorch (the numbers above).
- **Revisit if:** real runs don't keep JAX's lead (see the bet in `training.md`).
- **Hard to reverse:** the worker, optimizers and analyses are JAX code throughout.

## The core never imports JAX

CLI, store, runners, remotes, server and the catalogs' `__init__` modules import no JAX;
plugins keep JAX in `impl.py`. `tests/test_boundary.py` enforces it.

- **Why:** commands start in well under a second, the discrete GPU stays asleep when
  nothing trains, and the core runs on machines without JAX.

## A run is a directory; DuckDB reads it in place

`<home>/runs/<id>/`: `spec.json` (frozen), `status.json` (rewritten atomically),
`train.ndjson`, `val.ndjson`, `events.ndjson` (append-only), `models/*.safetensors`
(saved models), `state/` (checkpoint), `summary.json`, `worker.log`. `kiseki query` runs
SQL over views of all runs (`kiseki/query.py`). Readers tolerate a torn last line.

- **Why:** no database server, nothing to migrate, rsync mirrors runs between machines,
  and agents can read any file directly.
- **Rejected:** SQLite as the store (concurrent writers across processes and machines,
  harder to mirror).

## One queue runner per machine; runs outlive their launcher

Submitting writes a queued run and starts a detached runner unless one holds
`<home>/queue.lock`. It runs queued runs one at a time as `python -m kiseki.worker train`,
marks runs whose worker died as failed, and exits when the queue is empty.
Stop/pause go through a `control` file the worker reads between blocks; pause saves a
checkpoint, and resume requeues.

- **Why:** the laptop can sleep or drop SSH without losing or duplicating runs; one run
  at a time keeps the whole GPU for it.
- **Revisit if:** small runs leave the GPU idle enough that running two at once pays.

## Remotes are SSH hosts with their own home and queue

`~/.config/kiseki/machines.toml` (untracked; holds contact details) names machines by
their `~/.ssh/config` alias. `kiseki run --on lugia` rsyncs the package to
`<remote home>/code`, runs `uv sync` only when dependencies changed, makes the dataset
available (the remote downloads it from the source with pinned hashes, else this machine
pushes its copy), writes `machine.json` and submits over one multiplexed SSH connection.
`pull` mirrors small files of remote runs (models with `--models`); read commands refresh
active remote runs.

- **Why:** "An easy way to select remote machines" and lugia (PREFERENCES.md); a remote
  needs only SSH, uv and its own disk.
- **Rejected:** running workers over a live SSH session (dies with the connection);
  copying datasets from the laptop first (lugia receives ~150 KB/s from it).

## Shared servers get a notice while runs are active

When a machine has `notice` set, its runner keeps a block in that file naming the
contact, the start and a padded forecast of the end (Brasília time), refreshed every 10
minutes and removed when the queue empties (`kiseki/notice.py`). This implements lugia's
rule, quoted in `AGENTS.md`.

## The app server: JSON API, SSE, session token

`kiseki serve` (Starlette, `kiseki/server.py`) binds 127.0.0.1, prints its URL with a
random token, and requires that token and a local Host on every /api request. Status
changes reach the UI as server-sent events from a 1 s poll of status-file mtimes; curves
load column-wise from a byte offset, so the UI fetches only new rows. Actions call the
same functions as the CLI (`kiseki/actions.py`).

- **Why:** web pages in the user's browser can neither read nor drive the API; polling
  mtimes costs microseconds and needs no inotify edge cases.
- **Rejected:** FastAPI (the old stack; pydantic adds startup time and nothing needed).

## GPU memory is allocated on demand

Workers set `XLA_PYTHON_CLIENT_PREALLOCATE=false` unless `KISEKI_GPU_MEMORY_FRACTION` is
set.

- **Why:** a report can run beside a training run, and lugia's GPU stays shared.
- **Bet:** costs nothing measurable in steps/s. Test: the same run with the fraction at
  0 and 0.75 on the laptop in performance mode. Not yet measured.

## CUDA user-space libraries pinned to 13.0

`pyproject.toml` constrains the nvidia-* wheels to the CUDA 13.0 releases.

- **Why:** lugia's driver 580 is a CUDA 13.0 driver, and its GPU went down during the
  first run with the 13.4 libraries (2026-10-17); PyTorch with 13.0 ran there without
  trouble. Newer drivers run 13.0 fine (verified on the laptop, driver 595).
- **Revisit if:** lugia's driver is upgraded, or the GPU fails again with 13.0 (then the
  libraries were not the cause).
- **Bet:** 13.0 runs cleanly on lugia once its GPU is reset. Test: a tiny kernel, then a
  smoke run. Pending the reset.

## Exclusions

- Accounts, multiple users, and access from other machines: the app is for one person on
  this laptop ("only needs to run on this machine for now", PREFERENCES.md).
