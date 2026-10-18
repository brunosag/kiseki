// How numbers, times and runs read in the UI.
import { properties } from "./schema.ts";
import type { Json, Options, Run, Schema } from "./types";

export const DASH = "—";

export function percent(value: number | null | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value) ? DASH : `${(value * 100).toFixed(digits)}%`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return DASH;
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "now", "5m", "3h", "2d", then "Oct 17". */
export function ago(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return DASH;
  const then = new Date(iso);
  const seconds = (now - then.getTime()) / 1000;
  if (seconds < 45) return "now";
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  if (seconds < 7 * 86400) return `${Math.round(seconds / 86400)}d`;
  return `${MONTHS[then.getMonth()]} ${then.getDate()}`;
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const at = new Date(iso);
  const time = at.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
  return `${MONTHS[at.getMonth()]} ${at.getDate()}, ${time}`;
}

/** When a run or report was made, from its id (20261017-155537-…), as an ISO time. */
export function idTime(id: string): string | null {
  const match = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/.exec(id);
  if (!match) return null;
  const [, year, month, day, hour, minute, second] = match;
  return `${year}-${month}-${day}T${hour}:${minute}:${second}Z`;
}

/** Numbers as people write them: 0.05, 1000, 1e-4. */
export function number(value: Json | undefined): string {
  if (typeof value !== "number") return value == null ? DASH : String(value);
  if (Number.isInteger(value)) return String(value);
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) return Number(value.toPrecision(3)).toExponential();
  return String(Number(value.toPrecision(4)));
}

export function count(value: number): string {
  return value.toLocaleString("en-US");
}

/** The run id's random suffix, the short name runs go by: …-mnist-sgd-43ac → 43ac. */
export function shortId(id: string): string {
  return id.slice(id.lastIndexOf("-") + 1);
}

export function words(key: string): string {
  return key.replaceAll("_", " ");
}

export function optimizerTitle(options: Options | null, key: string): string {
  return options?.optimizers[key]?.title ?? key;
}

export function datasetTitle(options: Options | null, key: string): string {
  return options?.datasets[key]?.title ?? key;
}

export function analysisTitle(options: Options | null, key: string): string {
  return options?.analyses[key]?.title ?? words(key);
}

/** A run as the app names it: its optimizer and short id, "SGD 43ac". */
export function runName(options: Options | null, run: Run): string {
  return `${optimizerTitle(options, run.spec.optimizer)} ${shortId(run.id)}`;
}

/** "NVIDIA GeForce RTX 3050 Laptop GPU" → "RTX 3050 Laptop GPU". */
export function gpuName(name: string): string {
  return name.replace(/^NVIDIA (GeForce )?/, "");
}

function defaults(schema: Schema): Record<string, Json> {
  return Object.fromEntries(Object.entries(properties(schema)).map(([key, property]) => [key, property.default ?? null]));
}

/** The params a run sets away from its optimizer's defaults: what sets it apart. */
export function changedParams(run: Run, options: Options | null): [string, Json][] {
  const schema = options?.optimizers[run.spec.optimizer]?.params;
  const base = schema ? defaults(schema) : {};
  return Object.entries(run.spec.params).filter(
    ([key, value]) => JSON.stringify(value) !== JSON.stringify(base[key]),
  );
}

export function isActive(state: string): boolean {
  return state === "queued" || state === "running";
}

export function canResume(state: string): boolean {
  return state === "paused" || state === "stopped" || state === "failed";
}

export function progress(run: Run): number {
  const { step = 0, iterations } = run.status;
  const total = iterations ?? run.spec.iterations;
  return total ? Math.min(1, step / total) : 0;
}
