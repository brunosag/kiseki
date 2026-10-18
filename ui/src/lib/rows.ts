// A run's curves (train.ndjson, val.ndjson), read column-wise past a byte offset. Each
// status change of the run fetches only the new rows; what was read stays here for the
// session, so going back to a run draws at once.
import { useEffect, useReducer } from "react";
import { api } from "./api";
import { appendColumns, type Columns } from "./curves";
import type { Rows, Status } from "./types";

type Kind = "train" | "val";

interface Entry {
  columns: Columns;
  offset: number;
  step: number;
  pending?: Promise<void>;
}

const entries = new Map<string, Entry>();

function load(key: string, path: string, step: number): Promise<void> {
  let entry = entries.get(key);
  // A run restarted from an earlier checkpoint rewrites its curves: read them again.
  if (!entry || step < entry.step) {
    entry = { columns: {}, offset: 0, step };
    entries.set(key, entry);
  }
  if (entry.pending) return entry.pending;
  const current = entry;
  current.pending = api<Rows>(`${path}?offset=${current.offset}`)
    .then((rows) => {
      if (Object.keys(rows.columns).length) current.columns = appendColumns(current.columns, rows.columns);
      current.offset = rows.offset;
      current.step = Math.max(current.step, step);
    })
    .finally(() => (current.pending = undefined));
  return current.pending;
}

/** The run's rows of one kind, refreshed whenever its status changes. */
export function useRows(id: string, kind: Kind, status: Status | undefined): Columns | null {
  return useRowsOf([{ id, status }], kind)[0];
}

/** Several runs' rows of one kind (a comparison), in the order given. */
export function useRowsOf(runs: { id: string; status?: Status }[], kind: Kind): (Columns | null)[] {
  const [, redraw] = useReducer((n: number) => n + 1, 0);
  const version = runs.map(({ id, status }) => `${id}/${status?.state}/${status?.step}/${status?.updated_at}`).join();
  useEffect(() => {
    let live = true;
    for (const { id, status } of runs)
      load(`${id}/${kind}`, `runs/${encodeURIComponent(id)}/rows/${kind}`, status?.step ?? 0).then(
        () => live && redraw(),
        () => {},
      );
    return () => {
      live = false;
    };
  }, [kind, version]);
  return runs.map(({ id }) => entries.get(`${id}/${kind}`)?.columns ?? null);
}
