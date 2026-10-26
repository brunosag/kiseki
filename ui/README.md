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

On the laptop, GNOME's app menu has kiseki: a desktop entry in `~/config/home.nix` runs
this checkout's `kiseki-app` (`~/src/kiseki`) with a pinned Electron. Unless the app is
already running, `kiseki-app` installs packages and rebuilds `dist/` when they are older
than the lockfile or sources (errors go to `~/.local/state/kiseki-app.log` and a desktop
notification). It then uses `$ELECTRON_BIN`, else `electron` on PATH, else
`nix shell nixpkgs#electron`. The app starts `.venv/bin/python -m kiseki.cli serve`, reads
the URL and token it prints, and stops it on quit; training runs belong to the queue
runner and outlive the app. `KISEKI_SERVER_URL='http://127.0.0.1:PORT/?token=TOKEN'`
attaches to a running server instead. `KISEKI_DEBUG_PORT` opens a loopback DevTools port
for the scripts below.

The app sits in the tray: closing the window hides it and keeps the server, launching
again shows it, and the tray's Quit stops the server (design:
`../docs/design/architecture.md`).

## The AppImage

Other Linux machines (an Ubuntu laptop) run the AppImage from the GitHub Release that a
`v<version>` tag builds (`.github/workflows/appimage.yml`; the tag must match
`ui/package.json` and `pyproject.toml`). On first start it shows the setup page, which
installs the Python environment with a click; the same list is at Setup in the app.
`scripts/appimage.sh` builds one into `release/` (on the vps: it needs no Nix, and
saves the laptop's memory). `node scripts/appimage-check.mjs APPIMAGE [OUT_DIR]` walks it
as a new user would, under Xvfb with a fresh HOME: setup, install, Open kiseki, a smoke run
(about 1 GB of downloads, 10 minutes on the vps). Design: `../docs/design/architecture.md`.

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

## Code

- `electron/main.cjs` starts `kiseki serve`, opens the window, keeps the tray icon and
  menu, hides the window on close and stops the server on quit; `electron/preload.cjs`
  gives the page `window.kisekiWindow` (typed in `src/lib/shell.ts`): GNOME's button
  layout, window state and buttons for the title bar it draws, and the tray menu.
  `electron/requirements.cjs` checks and installs what the machine needs (the setup
  view's rows), and gives packaged builds their per-version code directory.
  `electron/icon.svg` is the app icon (`icon.png`, 512 px, rendered from it, is the
  AppImage's), `tray.svg` the panel icon's source (render `tray.png` and `tray@2x.png`
  from it with `rsvg-convert -w 16` and `-w 32`).
- The AppImage: `electron-builder.config.cjs`, `scripts/appimage.sh` (payload, BUILD id,
  pinned uv), `scripts/appimage-check.mjs` (end to end on a fresh HOME).
- `setup.html` + `src/setup.tsx` show `src/views/setup.tsx` alone, before a server exists;
  `src/lib/setup.ts` is its store.
- `src/app.tsx` is the frame (title bar, sidebar, the view in an inset panel), the
  dialogs and the global keys. `src/views/` holds one file per route and dialog;
  `src/components/` the app's own pieces (`chart.tsx` wraps uPlot, `pick.tsx` is the
  property chip, `status-icon.tsx` the state glyphs).
- `src/components/ui/` is coss ui as its registry ships it (`npx shadcn@latest add
  @coss/<name>`, registry in `components.json`).
- `src/lib/`: `live.ts` is the store every view reads (catalog, runs, reports, machines,
  kept current by SSE); `api.ts` the token and fetching; `rows.ts` curve rows by byte
  offset; `schema.ts` the run form's fields and grids; `format.ts`, `reports.ts`,
  `markdown.ts`, `curves.ts`, `tray.ts` (the tray menu's live part) are pure helpers with
  tests beside them; `types.ts` mirrors the server's JSON (endpoints: the docstring of
  `src/kiseki/server.py`).

The look: 13px text, weight 500 at most, chrome in `bg-sidebar`, views in the inset
`bg-background` panel, dense 40px rows. Series colors come from `SERIES` in
`lib/theme.ts` in fixed order (validated for both modes); text never wears them.

## Checks

`pnpm test` (the pure helpers in `src/lib`), `pnpm check`, `pnpm build`. To see the app
as the user does, `node scripts/verify.mjs [OUT_DIR]` launches it on a fresh scratch
store, queues two smoke runs through the form, compares them, builds a report, walks every
view (the run, compare and runs views in light too), opens a figure, uses the command menu,
closes the window and checks it hid with the server alive, launches again and checks the
window came back, quits and checks the server stopped; screenshots land in OUT_DIR. `node scripts/shots.mjs OUT name=/path ...`
screenshots an app already running with `KISEKI_DEBUG_PORT=9327`.
