const { app, BrowserWindow, dialog, session } = require("electron");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const repo = path.resolve(__dirname, "../..");
let server,
  window,
  quitting = false;
app.setName("kiseki");
app.commandLine.appendSwitch("ozone-platform", "wayland");
app.commandLine.appendSwitch("enable-features", "WaylandWindowDecorations");
if (process.env.KISEKI_DEBUG_PORT) {
  app.commandLine.appendSwitch("remote-debugging-address", "127.0.0.1");
  app.commandLine.appendSwitch(
    "remote-debugging-port",
    process.env.KISEKI_DEBUG_PORT,
  );
}
function validURL(text) {
  const url = new URL(text);
  if (
    url.protocol !== "http:" ||
    !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) ||
    !url.searchParams.get("token")
  )
    throw new Error("Expected a local kiseki URL with its session token");
  return url;
}
async function startServer() {
  if (process.env.KISEKI_SERVER_URL)
    return validURL(process.env.KISEKI_SERVER_URL);
  const python = path.join(repo, ".venv/bin/python");
  if (!fs.existsSync(python))
    throw new Error(
      `Python environment missing: ${python}. Set up the repository's .venv first.`,
    );
  return new Promise((resolve, reject) => {
    let output = "",
      tail = "";
    server = spawn(python, ["-m", "kiseki.cli", "serve"], {
      cwd: repo,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    });
    const timer = setTimeout(
      () =>
        reject(
          new Error(
            "The kiseki server did not become ready within 30 seconds.",
          ),
        ),
      30_000,
    );
    const fail = (error) => {
      clearTimeout(timer);
      reject(error);
    };
    server.once("error", fail);
    server.once("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`kiseki server exited (${code}). ${tail}`));
      if (window && !quitting) {
        dialog.showErrorBox(
          "kiseki server stopped",
          `The backend exited with code ${code}. Restart the application to reconnect.`,
        );
        app.quit();
      }
    });
    server.stderr.on("data", (chunk) => {
      tail = (tail + chunk.toString()).slice(-4000);
    });
    server.stdout.on("data", (chunk) => {
      output = (output + chunk.toString()).slice(-8000);
      const match = /^kiseki: serving (http[^\s]+)$/m.exec(output);
      if (match) {
        clearTimeout(timer);
        try {
          resolve(validURL(match[1]));
        } catch (error) {
          fail(error);
        }
      }
    });
  });
}
app.whenReady().then(async () => {
  try {
    const url = await startServer();
    session.defaultSession.setPermissionRequestHandler(
      (_contents, _permission, callback) => callback(false),
    );
    session.defaultSession.webRequest.onHeadersReceived((details, callback) =>
      callback({
        responseHeaders: {
          ...details.responseHeaders,
          "Content-Security-Policy": [
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
          ],
        },
      }),
    );
    window = new BrowserWindow({
      width: 1440,
      height: 960,
      minWidth: 980,
      minHeight: 680,
      title: "kiseki",
      backgroundColor: "#111619",
      autoHideMenuBar: true,
      show: false,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        spellcheck: false,
      },
    });
    window.webContents.on("will-navigate", (event, destination) => {
      if (new URL(destination).origin !== url.origin) event.preventDefault();
    });
    window.webContents.setWindowOpenHandler(({ url: destination }) => {
      const target = new URL(destination);
      if (
        target.origin === url.origin &&
        /^\/api\/reports\/[^/]+\/files\//.test(target.pathname)
      )
        return {
          action: "allow",
          overrideBrowserWindowOptions: {
            autoHideMenuBar: true,
            webPreferences: {
              nodeIntegration: false,
              contextIsolation: true,
              sandbox: true,
            },
          },
        };
      return { action: "deny" };
    });
    window.once("ready-to-show", () => {
      window.show();
      window.maximize();
    });
    await window.loadURL(url.href);
  } catch (error) {
    dialog.showErrorBox("Could not start kiseki", String(error.message));
    app.quit();
  }
});
app.on("window-all-closed", () => {
  if (!quitting) app.quit();
});
app.on("before-quit", (event) => {
  if (
    !server ||
    server.exitCode !== null ||
    server.signalCode !== null ||
    quitting
  )
    return;
  event.preventDefault();
  quitting = true;
  // Close renderer streams before waiting for uvicorn's graceful shutdown.
  for (const win of BrowserWindow.getAllWindows()) win.destroy();
  const force = setTimeout(() => {
    server.kill("SIGKILL");
  }, 3000);
  server.once("exit", () => {
    clearTimeout(force);
    app.quit();
  });
  server.kill("SIGTERM");
});
process.on("SIGTERM", () => app.quit());
process.on("SIGINT", () => app.quit());
