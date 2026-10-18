// Client-side routes on the Navigation API: plain <a href="/runs/…"> links navigate in
// place, and the title bar's back and forward follow the same history.
import { useMemo, useSyncExternalStore } from "react";

export type Route =
  | { name: "runs" }
  | { name: "run"; id: string }
  | { name: "compare"; ids: string[] }
  | { name: "reports" }
  | { name: "report"; id: string }
  | { name: "machines" };

export function parseRoute(path: string, search = ""): Route {
  const [first, second] = path.split("/").filter(Boolean).map(decodeURIComponent);
  if (first === "runs" && second) return { name: "run", id: second };
  if (first === "compare") {
    const ids = new URLSearchParams(search).get("runs")?.split(",").filter(Boolean) ?? [];
    return { name: "compare", ids };
  }
  if (first === "reports") return second ? { name: "report", id: second } : { name: "reports" };
  if (first === "machines") return { name: "machines" };
  return { name: "runs" };
}

export const href = {
  run: (id: string) => `/runs/${encodeURIComponent(id)}`,
  report: (id: string) => `/reports/${encodeURIComponent(id)}`,
  compare: (ids: string[]) => `/compare?runs=${ids.map(encodeURIComponent).join(",")}`,
};

export function startRouter(): void {
  navigation.addEventListener("navigate", (event) => {
    if (!event.canIntercept || event.hashChange || event.downloadRequest !== null) return;
    if (new URL(event.destination.url).pathname.startsWith("/api/")) return;
    event.intercept({ handler: async () => {} });
  });
}

function subscribe(listener: () => void) {
  navigation.addEventListener("currententrychange", listener);
  return () => navigation.removeEventListener("currententrychange", listener);
}

export function useRoute(): Route {
  const where = useSyncExternalStore(subscribe, () => location.pathname + location.search);
  return useMemo(() => {
    const url = new URL(where, location.origin);
    return parseRoute(url.pathname, url.search);
  }, [where]);
}

export function useHistory(): { back: boolean; forward: boolean } {
  const back = useSyncExternalStore(subscribe, () => navigation.canGoBack);
  const forward = useSyncExternalStore(subscribe, () => navigation.canGoForward);
  return { back, forward };
}

export function navigate(to: string): void {
  navigation.navigate(to);
}
