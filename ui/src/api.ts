import { useEffect, useRef, useState } from "react";
import type { Rows, Status } from "./types";
const query = new URLSearchParams(location.search);
const supplied = query.get("token");
if (supplied) {
  sessionStorage.setItem("kiseki-token", supplied);
  history.replaceState(null, "", location.pathname + location.hash);
}
const token = supplied || sessionStorage.getItem("kiseki-token") || "";
function filesURL(kind: "reports" | "runs", id: string, path: string) {
  return `/api/${kind}/${encodeURIComponent(id)}/files/${path.split("/").map(encodeURIComponent).join("/")}?token=${encodeURIComponent(token)}`;
}
export function fileURL(id: string, path: string) {
  return filesURL("reports", id, path);
}
export function runFileURL(id: string, path: string) {
  return filesURL("runs", id, path);
}
export async function api<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      "X-Kiseki-Token": token,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let message = res.statusText;
    try {
      message = (await res.json()).error || message;
    } catch {
      /* non-JSON server failure */
    }
    throw new Error(`${res.status}: ${message}`);
  }
  return res.json();
}
export function eventStream() {
  return new EventSource(`/api/events?token=${encodeURIComponent(token)}`);
}
export function useResource<T>(path: string | null, refresh = 0) {
  const [state, set] = useState<{
    path: string | null;
    data?: T;
    error?: string;
  }>({ path });
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    api<T>(path, undefined, controller.signal).then(
      (data) => set({ path, data }),
      (e) => {
        if (!controller.signal.aborted) set({ path, error: String(e.message) });
      },
    );
    return () => controller.abort();
  }, [path, refresh]);
  return state.path === path ? state : { path };
}
// Curves use byte offsets; completed curves remain in this bounded in-memory cache.
const rows = new Map<string, Rows>();
const inflight = new Map<string, Promise<Rows>>();
const epochs = new Map<string, number>();
export function observeRunStatus(
  id: string,
  previous: Status | undefined,
  next: Status,
) {
  if (!previous) return;
  const restarted =
    (next.step ?? 0) < (previous.step ?? 0) ||
    (previous.state !== next.state &&
      (previous.state !== "running" || next.state === "queued"));
  if (restarted) resetRunRows(id);
}
export function resetRunRows(id: string) {
  for (const kind of ["train", "val"]) {
    const path = `runs/${id}/rows/${kind}`;
    rows.delete(path);
    inflight.delete(path);
    epochs.set(path, (epochs.get(path) ?? 0) + 1);
  }
  window.dispatchEvent(new CustomEvent("kiseki:rows-reset", { detail: id }));
}
async function moreRows(path: string): Promise<Rows> {
  if (inflight.has(path)) return inflight.get(path)!;
  const previous = rows.get(path) ?? { columns: {}, offset: 0 },
    epoch = epochs.get(path) ?? 0;
  const promise: Promise<Rows> = api<Rows>(`${path}?offset=${previous.offset}`)
    .then((next) => {
      if ((epochs.get(path) ?? 0) !== epoch) return moreRows(path);
      if (next.offset === previous.offset && rows.has(path)) return previous;
      const columns = { ...previous.columns };
      for (const [key, values] of Object.entries(next.columns))
        columns[key] = (columns[key] ?? []).concat(values);
      const value = { columns, offset: next.offset };
      rows.delete(path);
      rows.set(path, value);
      if (rows.size > 48) rows.delete(rows.keys().next().value!);
      return value;
    })
    .finally(() => {
      if (inflight.get(path) === promise) inflight.delete(path);
    });
  inflight.set(path, promise);
  return promise;
}
export function useRows(id: string, kind: "train" | "val", live: boolean) {
  const path = `runs/${id}/rows/${kind}`;
  const [revision, bump] = useState(0);
  useEffect(() => {
    const reset = (event: Event) => {
      if ((event as CustomEvent).detail === id) bump((v) => v + 1);
    };
    window.addEventListener("kiseki:rows-reset", reset);
    return () => window.removeEventListener("kiseki:rows-reset", reset);
  }, [id]);
  const [state, set] = useState<{ path: string; data?: Rows; error?: string }>({
    path,
    data: rows.get(path),
  });
  useEffect(() => {
    let active = true,
      timer: ReturnType<typeof setTimeout>;
    set({ path, data: rows.get(path) });
    const load = async () => {
      try {
        const data = await moreRows(path);
        if (active) set({ path, data });
      } catch (e) {
        if (active) set({ path, error: String(e) });
      }
      if (active && live) timer = setTimeout(load, 2000);
    };
    void load();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [path, live, revision]);
  return state.path === path ? state : { path, data: rows.get(path) };
}
export function useViewTiming(name: string, ready: boolean) {
  const measured = useRef(false);
  useEffect(() => {
    measured.current = false;
    performance.mark(`kiseki:${name}:start`);
  }, [name]);
  useEffect(() => {
    if (!ready || measured.current) return;
    let frame = requestAnimationFrame(() => {
      frame = requestAnimationFrame(() => {
        const last = performance
          .getEntriesByName(`kiseki:${name}:start`)
          .at(-1);
        if (last && !measured.current) {
          measured.current = true;
          performance.measure(`kiseki:${name}:ready`, {
            start: last.startTime,
            end: performance.now(),
          });
        }
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [name, ready]);
}
