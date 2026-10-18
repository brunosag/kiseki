# kiseki app

React + TypeScript (Vite, Tailwind 4) on coss ui and Base UI, with uPlot for curves, in a
frameless Electron window that draws its own title bar. Electron comes from nixpkgs, not
npm.

```sh
cd ui
pnpm install --frozen-lockfile
pnpm build
./kiseki-app            # or: KISEKI_HOME=/tmp/kiseki-smoke ./kiseki-app
```

`kiseki-app` uses `$ELECTRON_BIN`, else `electron` on PATH, else
`nix shell nixpkgs#electron`. It starts `.venv/bin/python -m kiseki.cli serve`, reads the
URL and token it prints, and stops it on quit; training runs belong to the queue runner
and outlive the app. `KISEKI_SERVER_URL='http://127.0.0.1:PORT/?token=TOKEN'` attaches to
a running server instead. `KISEKI_DEBUG_PORT` opens a loopback DevTools port for the
scripts below.

## Views

- **Runs:** every run grouped by state (or dataset, optimizer), live over SSE, with a
  filter, a search and a selection that compares, reports, stops, pauses or resumes.
  j/k or arrows move, x selects, Esc clears.
- **Run:** headline numbers, validation and training curves (accuracy or loss, by step or
  time), activity, saved models and the worker log, and the full setup in a side panel.
- **Compare:** validation curves overlaid, and a table keyed by color with each run's
  numbers and when it first reaches an accuracy you choose.
- **Reports:** each report as a document: its models, then every analysis's text, tables
  and figures, with an outline. While building, the analyses tick off.
- **Machines:** each machine's GPU and queue, probed in the background or on Refresh.
- **New run** (`c`): optimizer, dataset, model, machine and preset as chips, then the
  optimizer's parameters and training settings. Commas try several values, one run each.
- **New report:** finished runs' best models, the split, the analyses.
- **Ctrl+K:** every action, place, run and report.

## Checks

`pnpm test` (the pure helpers in `src/lib`), `pnpm check`, `pnpm build`. To see the app
as the user does, `node scripts/verify.mjs [OUT_DIR]` launches it on a fresh scratch
store, queues two smoke runs through the form, compares them, builds a report, walks every
view (the run, compare and runs views in light too), opens a figure, uses the command menu,
closes the window and checks the server stopped; screenshots land in OUT_DIR. `node scripts/shots.mjs OUT name=/path ...`
screenshots an app already running with `KISEKI_DEBUG_PORT=9327`.
