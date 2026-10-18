# UI

React on coss ui (shadcn-style components over Base UI) in a frameless Electron window,
laid out after Linear. Design and its reasons: `../docs/design/architecture.md` (The
UI). Running and checks: `README.md`.

## Map

- `electron/main.cjs` starts `kiseki serve`, opens the window and stops the server on
  quit; `electron/preload.cjs` gives the page `window.kisekiWindow` (GNOME's button
  layout, window state, minimize/maximize/close) for the title bar it draws.
- `src/app.tsx` is the frame (title bar, sidebar, the view in an inset panel), the
  dialogs and the global keys. `src/views/` holds one file per route and dialog;
  `src/components/` the app's own pieces (`chart.tsx` wraps uPlot, `pick.tsx` is the
  property chip, `status-icon.tsx` the state glyphs).
- `src/components/ui/` is coss ui as its registry ships it (`npx shadcn@latest add
  @coss/<name>`, registry in `components.json`). Restyle at the call site, not in these
  files, so a component can be re-added from the registry. The registry also owns
  `src/hooks/use-media-query.ts`, `src/lib/utils.ts` and `src/lib/segmented-control.ts`.
- `src/lib/`: `live.ts` is the store every view reads (catalog, runs, reports, machines,
  kept current by SSE); `api.ts` the token and fetching; `rows.ts` curve rows by byte
  offset; `schema.ts` the run form's fields and grids; `format.ts`, `reports.ts`,
  `markdown.ts`, `curves.ts` are pure helpers with tests beside them; `types.ts` mirrors
  the server's JSON (endpoints: the docstring of `src/kiseki/server.py`).

## Rules

- The server is the contract: fix data problems there, not by guessing in the UI.
- Look: 13px text, weight 500 at most, chrome in `bg-sidebar`, views in the inset
  `bg-background` panel, dense 40px rows. Series colors come from `SERIES` in
  `lib/theme.ts` in fixed order (validated for both modes); text never wears them.
- Report text is Markdown parsed to React elements (`lib/markdown.ts`); never render it as
  HTML.
- Settings that should outlive the window go through `useStored` (localStorage). They
  belong to the page's origin, so the server keeps port 47630 when it is free.

## Gotchas

- SSE carries status changes (`run`, `report`) and probes (`machines`), not curve rows:
  `useRows` fetches new rows on each status change, and a reconnect reloads the lists.
- Validation rows use `val_accuracy` and `val_loss`; training rows `accuracy` and `loss`;
  both have `step`, only validation rows `elapsed`.
- Electron shutdown: destroy the windows before waiting for the Python child. Open SSE
  connections otherwise hold uvicorn's graceful shutdown until the kill timer.
- `-webkit-app-region: drag` (the `drag` utility) swallows clicks: every control inside
  the title bar sits in a `no-drag` container. Dragging and double-click (GNOME's
  titlebar action) are Chromium's own; the page never sees the double-click.
- CDP input goes straight to the page and skips the drag regions; title bar behaviour
  needs real input (`ydotool`, relative moves only: an absolute move passes the top-left
  hot corner and opens GNOME's overview).
- Training checks use a scratch store (`KISEKI_HOME=/tmp/...`). Queue on lugia only under
  the repo's rules (`../AGENTS.md`).
