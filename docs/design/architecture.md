# Architecture

How kiseki is split into processes and where state lives: a light Python core (CLI, run
store, queue runners, remotes, app server), JAX worker processes for training and
reports, and a React UI in an Electron window.

## Two languages: Python with JAX, and a React UI in Electron

Training and analysis run in JAX; everything else is plain Python; the UI is React +
TypeScript in a thin Electron shell that talks to `kiseki serve`.

- **Why:** the app should be "optimized for" this machine with "a great focus on
  performance" (PREFERENCES.md). Scoring one step of 1000 networks on a 512-image MNIST
  batch, 50 networks at a time, on the RTX 3050 in the performance power profile, on AC
  (2026-10-17, scratch benchmarks): 878 ms in the old PyTorch vmap code, 462 ms in
  hand-tuned PyTorch with torch.compile, 299 ms in plain JAX. In a real run of the rewrite
  a whole LEEA step at those settings (scoring, selection, metrics) took 327 ms
  (3.05 steps/s): less than the old code's scoring alone. Electron is Chromium, so agents
  test what the user sees, and it draws on the Intel iGPU, leaving the 4 GB of VRAM to
  experiments. Agent's design; a Codex critic (GPT-6.1 Sol) found no stack that clearly
  beats it.
- **Note:** the first benchmarks ran in a slower power profile by mistake (1250 / 661 /
  471 ms for the three above; CIFAR-10 12.9 s / 7.9 s fp32 and 3.5 s bf16 in JAX, not
  re-measured). Compare timings only within one profile
  (`/sys/firmware/acpi/platform_profile`).
- **Rejected:** Rust core + Tauri (a third language, slower agent builds, WebKit on Linux
  differs from what agents check in Chromium); native GUIs (egui, GPUI, Qt: no page
  structure for agents to inspect); PyTorch (the numbers above).
- **Revisit if:** JAX loses its lead on a model the project adds (CIFAR-10 models are the
  likeliest; re-measure them in the performance profile first).
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

## One queue runner per home, one run at a time per machine; runs outlive their launcher

Submitting writes a queued run and starts a detached runner unless one holds
`<home>/queue.lock`. It runs queued runs one at a time as `python -m kiseki.worker train`,
marking each running with the worker's pid as it launches, marks runs whose worker died
as failed, and exits when the queue is empty. It touches only runs whose `machine` is its
own; mirrors of other machines' runs are left alone.
Stop/pause go through a `control` file the worker reads between blocks; pause saves a
checkpoint, and resume requeues. A machine can hold several homes (its own, the one
remotes use, scratch homes), so a runner takes the machine lock (`<cache>/machine.lock`)
before starting a worker and the worker inherits it: one run at a time per machine,
whichever home queued it.

While a worker runs, its runner holds a `systemd-inhibit` sleep lock (tied to the
runner's pid, so it ends with the runner) and watches the run's status file: a worker
that leaves it unchanged for 15 minutes of awake time is killed and its run requeued to
resume from its checkpoint, three times at most before the run fails.

- **Why:** the laptop can sleep or drop SSH without losing or duplicating runs; one run
  at a time keeps the whole GPU for it, also when the vps sends runs to the laptop while
  the user runs their own. The inhibitor and watchdog came after a run on 2026-10-18:
  the laptop slept while no agent turn held T3's inhibitor, and on wake the worker spun
  with the GPU idle (its CUDA context lost) until killed. Monotonic time stops during
  suspend, so a long sleep alone does not count as a stall.
- **Revisit if:** small runs leave the GPU idle enough that running two at once pays.

## Remotes are SSH hosts with their own home and queue

`~/.config/kiseki/machines.toml` (untracked; holds contact details) names machines by
their `~/.ssh/config` alias. A remote's home defaults to `~/.local/share/kiseki-remote`,
apart from the home kiseki uses when it runs there itself: the laptop and the vps both
run kiseki and are each other's remotes (agents on the vps send runs to the laptop's GPU
while it is awake). `kiseki run --on lugia` rsyncs the package to
`<remote home>/code`, runs `uv sync` only when dependencies changed, makes the dataset
available (the remote downloads it from the source with pinned hashes, else this machine
pushes its copy), writes `machine.json` and submits over one multiplexed SSH connection.
`pull` mirrors small files of remote runs (models with `--models`); read commands and the
app refresh active remote runs, and pull a run's models once they see it finish.

- **Why:** "An easy way to select remote machines" and lugia (PREFERENCES.md); a remote
  needs only SSH, uv and its own disk.
- **Rejected:** running workers over a live SSH session (dies with the connection);
  copying datasets from the laptop first (lugia receives ~150 KB/s from it); the
  machine's own home as a remote's (until 2026-10-18: on a machine that also runs kiseki,
  the submitter's `machine.json` renamed it, so its own runs never started); a home per
  controller on a remote (notice blocks are per Unix user, so two would remove each
  other's).

## Shared servers get a notice while runs are active

When a machine has `notice` set, its runner keeps a block in that file naming the
contact, the start and a padded forecast of the end (Brasília time), refreshed every 10
minutes and removed when the queue empties (`kiseki/notice.py`). The block is written
before each run starts, and a run fails instead of starting when it can't be. This
implements lugia's rule, quoted in `AGENTS.md`.

## The app server: JSON API, SSE, session token

`kiseki serve` (Starlette, `kiseki/server.py`) binds 127.0.0.1, prints its URL with a
random token, and requires that token and a local Host on every /api request. Status
changes reach the UI as server-sent events from a 1 s poll of status-file mtimes; curves
load column-wise from a byte offset, so the UI fetches only new rows. Actions call the
same functions as the CLI (`kiseki/actions.py`).

- **Why:** web pages in the user's browser can neither read nor drive the API; polling
  mtimes costs microseconds and needs no inotify edge cases.
- **Rejected:** FastAPI (the old stack; pydantic adds startup time and nothing needed).

## The UI: coss ui, laid out after Linear, in a frameless window

`ui/` is React built from coss ui components (copied into `ui/src/components/ui/`, on
Base UI). Linear's layout: the title bar and navigation on the sidebar colour, each view
in an inset panel, dense rows with a property panel, a floating bar for selected runs and
a command menu (Ctrl+K). The Electron window has no native frame: the page draws the title
bar with the window buttons GNOME's layout asks for, and Chromium's drag regions move and
maximize it; nothing opens a second window. Reports show the figures the worker draws
(full size in a dialog); curves are uPlot canvases. The server keeps port 47630 when it is
free, so settings stored per origin survive restarts; a scratch store (`KISEKI_HOME`) gets
its own.

- **Why:** "Build it from the coss ui component library and take Linear's UI design as a
  reference. Make it minimal. It should have its own title bar too, not the native one."
  (PREFERENCES.md). Layout details are the agent's.
- **Rejected:** the first UI, built by a Codex agent (GPT-6.1 Sol), which the user had
  rewritten on 2026-10-18; with it went the interactive WebGL embedding scatter (the
  report's figure shows the same embedding) and the browsable LRP gallery (the figure
  shows its first 12 images).

## The app lives in the tray; on NixOS it launches from the checkout

kiseki has a tray icon (StatusNotifierItem, shown by GNOME's AppIndicator extension).
Once the app is open, closing the window hides it and keeps the server (on the setup page,
before there is a tray, closing quits); the tray menu opens the window, starts a
new run, lists what trains and builds now and the latest settled runs, and quits, which
stops the server. The page writes the menu's live part from its store (`ui/src/lib/tray.ts`)
and sends it over the preload bridge, so the shell holds no second event stream. One app
runs per store: launching again shows its window. GNOME launches it from a desktop entry in
the user's NixOS config (`~/config/home.nix`) that runs `~/src/kiseki/ui/kiseki-app` with
a pinned Electron; the launcher installs and rebuilds the UI when they are stale, so a pull
updates the app without a NixOS rebuild.

- **Why:** "Could we also make it like my T3 Code, where it gets its own tray icon and
  menu, and closing the window keeps the server running?" (PREFERENCES.md). The menu's
  contents and the launcher are the agent's design, reviewed by no critic.
- **Rejected:** the shell watching the event stream itself (a second copy of the page's
  store). An AppImage for this machine (2026-10-18): a pull updates the checkout without
  one; other machines get one (next section).

## Other machines run an AppImage whose setup page installs the rest

`ui/scripts/appimage.sh` builds an AppImage (electron-builder, Electron 41.9.1 as in
nixpkgs, the static AppImage runtime so no libfuse2) holding the shell, a pinned uv, and a
payload: the files `Remote.deploy` ships plus `ui/dist`, under a BUILD id (version and
content hash). CI builds it on a version tag and attaches it to the GitHub Release. On
first start the shell finds no environment for its BUILD and shows the setup page
(`setup.html`, served over `kiseki://app` because no server exists yet). Each row there is
a requirement with its state and one action (`ui/electron/requirements.cjs`): the Python
environment (required: copies the payload to `~/.local/share/kiseki-app/<BUILD>/` and runs
`uv sync`, with the CUDA extra when the GPU can use it), the NVIDIA GPU (driver ≥ 580 and
compute capability ≥ 7.5 for CUDA 13; opens Ubuntu's Additional Drivers), SSH and rsync
(installs through pkexec and apt), remote machines (`machines.toml` and the `[contact]` a
notice needs; opens its folder) and the app menu entry (rewritten on each start, since the
AppImage moves with updates). Once the environment is ready, Open kiseki starts the server
from it; the same rows stay at `/setup` in the app. Reinstalling the environment from
there stops the server first. Children get the environment without the AppImage's
variables, whose mount vanishes on quit. Where AppArmor blocks unprivileged user
namespaces (Ubuntu 23.10+), electron-builder's AppRun starts Electron with `--no-sandbox`;
the window only shows kiseki's own pages.

- **Why:** "I actually do need that AppImage now. I need to run the app on an Ubuntu
  laptop. Can we have some place that lists the requirements, shows their status, and lets
  one install them by clicking a button? Similar to what T3 does with the providers."
  (PREFERENCES.md); the Ubuntu laptop trains on its own GPU and sends runs to the others,
  and gets the AppImage from GitHub Releases (the user, 2026-10-18). A directory per BUILD
  means a new version never changes code under a running server, queue runner or worker.
  The rest is the agent's design, attacked by a fresh Claude critic (Codex and Gemini were
  unavailable), which brought the per-BUILD directory, the bundled uv, the GPU capability
  check, the Additional Drivers hand-off and the environment scrubbing.
- **Rejected:** bundling Python, JAX and CUDA in the AppImage (several GB per version, and
  the CUDA wheels depend on the driver found at install); installing uv with its curl
  script (unpinned, edits shell profiles); `pkexec ubuntu-drivers install` (it can stop at
  a Secure Boot prompt nobody sees); one shared code directory updated in place.
- **Revisit if:** remotes start getting runs from clients on different versions: each
  remote still has a single code directory that the last client to submit overwrites, and
  an older queue runner on a machine runs whatever the newer app queued.
- **Bet:** a new Ubuntu 24.04 user goes from the downloaded AppImage to a trained run with
  no terminal (remote machines still need SSH keys set up by hand). Test:
  `ui/scripts/appimage-check.mjs` on the vps (Ubuntu 24.04, AppArmor restriction on, no
  libfuse2, no GPU) with a fresh HOME. Stop at the first pass; then watch the first real
  install on the Ubuntu laptop.

## GPU memory is allocated on demand

Workers set `XLA_PYTHON_CLIENT_PREALLOCATE=false` unless `KISEKI_GPU_MEMORY_FRACTION` is
set.

- **Why:** a report can run beside a training run, and lugia's GPU stays shared.
- **Bet:** costs nothing measurable in steps/s. Test: the same LEEA run (200 steps,
  population 1000) on demand and preallocated at 0.75, laptop in the performance profile.
  Stop at the first measurement. Result 2026-10-17: 3.05 steps/s on demand, 3.00
  preallocated; settled.

## CUDA user-space libraries pinned to 13.0

`pyproject.toml` constrains the nvidia-* wheels to the CUDA 13.0 releases.

- **Why:** lugia's driver 580 is a CUDA 13.0 driver, and its GPU went down during the
  first run with the 13.4 libraries (2026-10-17); PyTorch with 13.0 ran there without
  trouble. Newer drivers run 13.0 fine (verified on the laptop, driver 595).
- **Revisit if:** lugia's driver is upgraded, or the GPU fails again with 13.0 (then the
  libraries were not the cause).
- **Bet:** 13.0 runs cleanly on lugia once its GPU is reset. Test: a tiny kernel, then a
  smoke run. Stop at the first failure: then the pin was not the cause, and lugia's GPU
  goes to its administrators before kiseki runs there again. Pending the reset.

## Exclusions

- Accounts, multiple users, and access from other machines: the app is for one person on
  this laptop ("only needs to run on this machine for now", PREFERENCES.md).
