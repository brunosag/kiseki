// The setup view's state: the shell's requirement rows (electron/requirements.cjs), kept
// current as install jobs report. Checked once on first use; Re-check probes again.
import { useSyncExternalStore } from "react";
import { type Requirement, type SetupJob, shell } from "./shell.ts";

interface Setup {
  rows: Requirement[] | null;
  checking: boolean;
}

let setup: Setup = { rows: null, checking: false };
let started = false;
const listeners = new Set<() => void>();

function set(patch: Partial<Setup>) {
  setup = { ...setup, ...patch };
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  if (!started && shell) {
    started = true;
    shell.onSetupJob(applyJob);
    check();
  }
  return () => listeners.delete(listener);
}

export function useSetup(): Setup {
  return useSyncExternalStore(subscribe, () => setup);
}

export async function check(fresh = false): Promise<void> {
  if (!shell) return;
  set({ checking: true });
  try {
    set({ rows: await shell.checkSetup(fresh) });
  } finally {
    set({ checking: false });
  }
}

function applyJob({ id, ...job }: SetupJob & { id: string }) {
  set({ rows: setup.rows?.map((row) => (row.id === id ? { ...row, job } : row)) ?? null });
  // A finished job changes what the machine has: look again (the environment's probe too).
  if (job.state !== "running") check(id === "env");
}

export function run(id: string): void {
  shell?.runSetup(id);
}

/** Whether the app can open: every required row usable and nothing installing. */
export function canOpen(rows: Requirement[] | null): boolean {
  return !!rows && rows.every((row) => (!row.required || row.state !== "missing") && row.job?.state !== "running");
}

export type Health = "missing" | "warning" | "ok" | "off";

/** A row's dot: a missing required row is a problem, a missing extra only off. */
export function rowHealth(row: Requirement): Health {
  if (row.state === "missing") return row.required ? "missing" : "off";
  return row.state;
}

/** The nav's dot: a missing required row, else a warning something here can fix. */
export function setupHealth(rows: Requirement[] | null): "missing" | "warning" | null {
  if (rows?.some((row) => rowHealth(row) === "missing")) return "missing";
  if (rows?.some((row) => row.state === "warning" && row.action)) return "warning";
  return null;
}
