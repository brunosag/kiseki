// What every view reads: the catalog, runs, reports and machines, loaded once and kept
// current by the server's event stream (`run`, `report`, `machines`). A reconnect
// reloads the lists, since events may have been missed meanwhile.
import { useSyncExternalStore } from "react";
import { api, eventSource } from "./api";
import type { Machine, Options, Report, Run, Status } from "./types";

interface Live {
  options: Options | null;
  runs: Run[] | null;
  reports: Report[] | null;
  machines: Machine[] | null;
  connected: boolean;
}

let live: Live = { options: null, runs: null, reports: null, machines: null, connected: true };
const listeners = new Set<() => void>();

function set(patch: Partial<Live>) {
  live = { ...live, ...patch };
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useLive(): Live {
  return useSyncExternalStore(subscribe, () => live);
}

/** Ids begin with their creation time (20261017-145539-…), so this is newest first. */
function newestFirst<T extends { id: string }>(items: T[]): T[] {
  return items.sort((a, b) => (a.id < b.id ? 1 : a.id > b.id ? -1 : 0));
}

/** Items merged by id, incoming ones replacing those already listed. */
function mergeById<T extends { id: string }>(list: T[], incoming: T[]): T[] {
  const byId = new Map(list.map((item) => [item.id, item]));
  for (const item of incoming) byId.set(item.id, item);
  return newestFirst([...byId.values()]);
}

async function loadLists() {
  const [runs, reports] = await Promise.all([api<Run[]>("runs"), api<Report[]>("reports")]);
  set({ runs: newestFirst(runs), reports: newestFirst(reports) });
}

function updated<T extends { id: string; status: Status }>(
  current: () => T[] | null,
  id: string,
  status: Status,
  fetch: () => Promise<T>,
  apply: (list: T[]) => void,
) {
  const list = current();
  if (!list) return;
  const index = list.findIndex((item) => item.id === id);
  if (index < 0) {
    // Created elsewhere (the CLI, another window): fetch the whole record, and merge it
    // into the list as it is by then (other events may have changed it meanwhile).
    fetch().then((item) => apply(mergeById(current() ?? [], [item])), () => {});
    return;
  }
  const next = list.slice();
  next[index] = { ...next[index], status };
  apply(next);
}

export function start(): void {
  api<Options>("options").then((options) => set({ options }));
  loadLists();
  api<Machine[]>("machines").then((machines) => set({ machines }));
  const source = eventSource();
  let lost = false;
  source.onopen = () => {
    set({ connected: true });
    if (lost) loadLists();
    lost = false;
  };
  source.onerror = () => {
    lost = true;
    set({ connected: false });
  };
  source.addEventListener("run", (event) => {
    const { id, status } = JSON.parse(event.data);
    updated(() => live.runs, id, status, () => api<Run>(`runs/${id}`), (runs) => set({ runs }));
  });
  source.addEventListener("report", (event) => {
    const { id, status } = JSON.parse(event.data);
    updated(() => live.reports, id, status, () => api<Report>(`reports/${id}`), (reports) => set({ reports }));
  });
  source.addEventListener("machines", (event) => set({ machines: JSON.parse(event.data) }));
}

export function addRuns(created: Run[]): void {
  set({ runs: mergeById(live.runs ?? [], created) });
}

export function addReport(report: Report): void {
  set({ reports: mergeById(live.reports ?? [], [report]) });
}

/** Probes every machine now; the answer also arrives as a `machines` event. */
export async function probeMachines(): Promise<void> {
  set({ machines: await api<Machine[]>("machines?fresh=1") });
}
