# UI

Running it, its code and its look: `README.md`.

## Rules

- The server is the contract: fix data problems there, not by guessing in the UI.
- Report text is Markdown parsed to React elements (`lib/markdown.ts`); never render it as
  HTML.
- Training checks use a scratch store (`KISEKI_HOME=/tmp/...`). Queue on lugia only under
  the repo's rules (`../AGENTS.md`).

## Gotchas

- `src/components/ui/` is coss ui as its registry ships it: restyle at the call site, not
  in these files, so a component can be re-added from the registry. The registry also owns
  `src/hooks/use-media-query.ts`, `src/lib/utils.ts` and `src/lib/segmented-control.ts`.
- Settings that should outlive the window go through `useStored` (localStorage). They
  belong to the page's origin, so the server keeps port 47630 when it is free.
- SSE carries status changes (`run`, `report`) and probes (`machines`), not curve rows:
  `useRows` fetches new rows on each status change, and a reconnect reloads the lists.
- Validation rows use `val_accuracy` and `val_loss`; training rows `accuracy` and `loss`;
  both have `step`, only validation rows `elapsed`.
- Electron shutdown: destroy the windows before waiting for the Python child. Open SSE
  connections otherwise hold uvicorn's graceful shutdown until the kill timer.
- The setup page is served by the shell over `kiseki://app`, the app by the server; the
  shell accepts setup calls from both origins only (`allowed` in `main.cjs`). Views it
  shares with the app must not touch `live.ts` or `api.ts`, which need a server.
- Packaged, `app.isPackaged` is true and the Python side runs from
  `~/.local/share/kiseki-app/<BUILD>/`, never from the AppImage's mount; a checkout runs
  its own `.venv`.
- Once the app is open, closing the window only hides it (on the setup page, before the
  tray exists, it quits): quit a scripted app with SIGTERM (or the tray's Quit),
  and a second `kiseki-app` on the same store just shows the first one's window and exits.
  Scratch stores get their own app, lock and tray icon.
- The Wayland app id comes from `app.setDesktopName("kiseki.desktop")`; without it
  Electron reports `electron` and GNOME can't match the window to the desktop entry.
- Read the tray menu without clicking the panel: its D-Bus menu, e.g. `gdbus call
  --session --dest <item> --object-path /com/canonical/dbusmenu --method
  com.canonical.dbusmenu.GetLayout -- 0 -1 '["label"]'` (items: `RegisteredStatusNotifierItems`
  on `org.kde.StatusNotifierWatcher`); `...dbusmenu.Event -- <id> clicked '<0>' 0` clicks one.
- `-webkit-app-region: drag` (the `drag` utility) swallows clicks: every control inside
  the title bar sits in a `no-drag` container. Dragging and double-click (GNOME's
  titlebar action) are Chromium's own; the page never sees the double-click.
- CDP input goes straight to the page and skips the drag regions; title bar behaviour
  needs real input (`ydotool`, relative moves only: an absolute move passes the top-left
  hot corner and opens GNOME's overview).
