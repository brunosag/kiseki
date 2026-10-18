// Small Electron inspection helper, using Node 24's built-in WebSocket.
import { readFile } from "node:fs/promises";
const port = process.env.KISEKI_DEBUG_PORT || "9327";
const targets = await (
  await fetch(`http://127.0.0.1:${port}/json/list`)
).json();
const page = targets.find(
  (t) => t.type === "page" && t.url.startsWith("http://127.0.0.1:"),
);
if (!page) throw new Error("No Electron page on the debugging port");
const socket = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
export const events = [];
socket.onmessage = (event) => {
  const result = JSON.parse(event.data);
  if (!result.id) {
    events.push(result);
    if (events.length > 5000) events.shift();
  }
  if (result.id) {
    const p = pending.get(result.id);
    if (p) {
      pending.delete(result.id);
      result.error
        ? p.reject(new Error(result.error.message))
        : p.resolve(result.result);
    }
  }
};
await new Promise((resolve, reject) => {
  socket.onopen = resolve;
  socket.onerror = reject;
});
export function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const key = ++id;
    pending.set(key, { resolve, reject });
    socket.send(JSON.stringify({ id: key, method, params }));
  });
}
export async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  });
  if (result.exceptionDetails)
    throw new Error(
      result.exceptionDetails.exception?.description ||
        result.exceptionDetails.text,
    );
  return result.result.value;
}
export async function waitFor(expression, timeout = 20_000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (await evaluate(expression)) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out: ${expression}`);
}
export function close() {
  socket.close();
}
if (process.argv[1]?.endsWith("cdp.mjs")) {
  const source =
    process.argv[2] === "--file"
      ? await readFile(process.argv[3], "utf8")
      : process.argv[2] || "document.body.innerText";
  try {
    console.log(JSON.stringify(await evaluate(source), null, 2));
  } finally {
    close();
  }
}
