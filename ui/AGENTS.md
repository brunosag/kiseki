# UI

- Map: `src/api.ts` holds the token, SSE, fetching and incremental curve rows;
  `src/core.ts` schema parsing, formatting and threshold crossings; `src/types.ts` mirrors
  the server's JSON (`src/kiseki/server.py` docstring lists the endpoints). Views are in
  `src/views/`; `components/Curve.tsx` wraps uPlot, `Scatter.tsx` is the WebGL scatter,
  `InteractiveReport.tsx` loads report arrays, `src/npy.ts` decodes NumPy files.
  `electron/main.cjs` owns the server and window lifecycle.
- The server is the contract: fix data problems there, not by guessing in the UI.
- SSE carries status changes (`run`, `report`) and probes (`machines`), not curve rows:
  open chart views poll rows by byte offset, and a reconnect reloads lists.
- Validation rows use `val_accuracy` and `val_loss`; training rows `accuracy` and `loss`;
  both have `step` and `elapsed`. Runs from before 2026-10-17 lack training `elapsed`.
- Report `.npy` files include float16 relevance maps and int64 indices and labels: read
  each array's dtype, never assume float32.
- Electron shutdown: close the windows before waiting for the Python child. Open SSE
  connections otherwise hold uvicorn's graceful shutdown until the kill timer.
- Training checks use a scratch store (`KISEKI_HOME=/tmp/kiseki-smoke`). Queue on lugia
  only under the repo's rules (`../AGENTS.md`).
