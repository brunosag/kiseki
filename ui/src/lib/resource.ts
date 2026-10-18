// One fetched resource for a view, fetched again when `version` changes; the last
// answer stays on screen while the next one loads.
import { useEffect, useState } from "react";
import { api } from "./api";

export function useResource<T>(path: string | null, version: unknown = 0): { data?: T; error?: string } {
  const [state, setState] = useState<{ path: string | null; data?: T; error?: string }>({ path });
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    api<T>(path, undefined, controller.signal).then(
      (data) => setState({ path, data }),
      (error) => !controller.signal.aborted && setState({ path, error: (error as Error).message }),
    );
    return () => controller.abort();
  }, [path, version]);
  return state.path === path ? state : {};
}

/** A text file (a worker log), fetched again when `version` changes. */
export function useText(url: string | null, version: unknown = 0): string | null {
  const [text, setText] = useState<string | null>(null);
  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    fetch(url, { signal: controller.signal })
      .then((response) => (response.ok ? response.text() : ""))
      .then(setText, () => {});
    return () => controller.abort();
  }, [url, version]);
  return text;
}
