# kiseki app

React + TypeScript (Vite, Tailwind), uPlot curves and a WebGL scatter for embeddings, in
an Electron window. Runtime dependencies: React, React DOM, uPlot. Electron comes from
nixpkgs, not npm.

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

- **Runs:** live table (status events over SSE), search, filters, selection. A run shows
  validation accuracy and loss against step or elapsed time, its frozen spec, events,
  saved models (download) and worker log, with stop, pause and resume.
- **New run:** forms from the catalog's JSON Schemas, comma-separated values as a grid,
  presets, and the machine to run on (remotes deploy first).
- **Compare:** selected runs overlaid, and the first step and time each reaches an
  accuracy threshold.
- **Reports:** create from saved models, every section's prose, tables and figures, an
  interactive t-SNE/PCA scatter (by class or prediction) and the LRP gallery.
- **Machines:** this laptop's GPU power state (memory and load while awake) and each
  remote's GPUs and queue, refreshed in the background.

## Checks

`pnpm test` (contract helpers in `src/core.ts`), `pnpm check`, `pnpm build`. To look at
the app as the user does, launch it with `KISEKI_DEBUG_PORT=9327`, then from the repo
root `node ui/scripts/verify.mjs` screenshots every view into `ui/out/` (it needs a
completed report in the store), `node ui/scripts/verify-actions.mjs` queues two smoke
runs and a report through the forms (scratch stores only), and
`node ui/scripts/verify-lifecycle.mjs` checks that quitting stops the server.
