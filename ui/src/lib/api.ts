// HTTP to `kiseki serve`. The window opens on `/?token=…`; the token moves to
// sessionStorage and out of the address, and rides on every request.

const given = new URLSearchParams(location.search).get("token");
if (given) {
  sessionStorage.setItem("kiseki-token", given);
  history.replaceState(history.state, "", location.pathname);
}
const token = given ?? sessionStorage.getItem("kiseki-token") ?? "";

export async function api<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      "X-Kiseki-Token": token,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const failure = await response.json().catch(() => null);
    throw new Error(failure?.error ?? `${response.status} ${response.statusText}`);
  }
  return response.json();
}

/** A run's or report's file (figures, models, logs), for <img src>, <a href> and fetch. */
export function fileURL(kind: "runs" | "reports", id: string, path: string): string {
  const parts = path.split("/").map(encodeURIComponent).join("/");
  return `/api/${kind}/${encodeURIComponent(id)}/files/${parts}?token=${encodeURIComponent(token)}`;
}

export function eventSource(): EventSource {
  return new EventSource(`/api/events?token=${encodeURIComponent(token)}`);
}
