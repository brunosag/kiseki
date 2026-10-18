// A minimal Chrome DevTools Protocol client for the app's window (KISEKI_DEBUG_PORT).
export async function connect(port = process.env.KISEKI_DEBUG_PORT ?? "9327") {
  const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  // The app's page, or the setup page before a server exists.
  const page = targets.find(
    (target) => target.type === "page" && (target.url.startsWith("http://127.0.0.1") || target.url.startsWith("kiseki://")),
  );
  if (!page) throw new Error(`no app page on port ${port}`);
  const socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = reject;
  });
  let next = 0;
  const pending = new Map();
  const listeners = new Map();
  socket.onmessage = (message) => {
    const data = JSON.parse(message.data);
    if (data.id != null) {
      const { resolve, reject } = pending.get(data.id);
      pending.delete(data.id);
      data.error ? reject(new Error(data.error.message)) : resolve(data.result);
    } else for (const listener of listeners.get(data.method) ?? []) listener(data.params);
  };
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++next;
      pending.set(id, { resolve, reject });
      socket.send(JSON.stringify({ id, method, params }));
    });
  const on = (method, listener) => listeners.set(method, [...(listeners.get(method) ?? []), listener]);
  /** Evaluates an expression in the page (awaiting promises) and returns its value. */
  const evaluate = async (expression) => {
    const { result, exceptionDetails } = await send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (exceptionDetails) throw new Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
    return result.value;
  };
  return { send, on, evaluate, close: () => socket.close() };
}
