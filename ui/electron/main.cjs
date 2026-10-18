// The desktop shell: starts `kiseki serve`, opens a frameless window on its URL (the page
// draws its own title bar through preload.cjs), and stops the server on quit. Training
// runs belong to the queue runner and outlive the app.
const { app, BrowserWindow, dialog, ipcMain, nativeTheme, session } = require("electron");
const { execFile, spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const repo = path.resolve(__dirname, "../..");
const LOCAL = ["127.0.0.1", "localhost", "[::1]"];
const CSP =
  "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; " +
  "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; " +
  "frame-ancestors 'none'";

let server;
let window;
let quitting = false;

app.setName("kiseki");
// A store of its own (KISEKI_HOME, as in scratch checks) gets its own saved settings, so
// it never inherits the real app's choices, such as the machine new runs go to.
if (process.env.KISEKI_HOME) app.setPath("userData", path.join(path.resolve(process.env.KISEKI_HOME), "app"));
app.commandLine.appendSwitch("ozone-platform", "wayland");
if (process.env.KISEKI_DEBUG_PORT) {
  app.commandLine.appendSwitch("remote-debugging-address", "127.0.0.1");
  app.commandLine.appendSwitch("remote-debugging-port", process.env.KISEKI_DEBUG_PORT);
}

function localURL(text) {
  const url = new URL(text);
  if (url.protocol !== "http:" || !LOCAL.includes(url.hostname) || !url.searchParams.get("token"))
    throw new Error("Expected a local kiseki URL with its session token");
  return url;
}

function startServer() {
  if (process.env.KISEKI_SERVER_URL) return Promise.resolve(localURL(process.env.KISEKI_SERVER_URL));
  const python = path.join(repo, ".venv/bin/python");
  if (!fs.existsSync(python))
    return Promise.reject(new Error(`No Python environment at ${python}: run uv sync first.`));
  return new Promise((resolve, reject) => {
    let output = "";
    let errors = "";
    server = spawn(python, ["-m", "kiseki.cli", "serve"], {
      cwd: repo,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    });
    const timer = setTimeout(() => reject(new Error("The server did not start within 30 s.")), 30_000);
    server.once("error", reject);
    server.once("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`The server exited (${code}). ${errors}`));
      if (window && !quitting) {
        dialog.showErrorBox("kiseki stopped", `The server exited with code ${code}.`);
        app.quit();
      }
    });
    server.stderr.on("data", (chunk) => (errors = (errors + chunk).slice(-4000)));
    server.stdout.on("data", (chunk) => {
      output = (output + chunk).slice(-8000);
      const match = /^kiseki: serving (http\S+)$/m.exec(output);
      if (!match) return;
      clearTimeout(timer);
      try {
        resolve(localURL(match[1]));
      } catch (error) {
        reject(error);
      }
    });
  });
}

// GNOME's title bar buttons, e.g. "appmenu:close" (close on the right only).
function buttonLayout() {
  return new Promise((resolve) => {
    execFile(
      "gsettings",
      ["get", "org.gnome.desktop.wm.preferences", "button-layout"],
      { timeout: 2000 },
      (error, stdout) => {
        const known = new Set(["minimize", "maximize", "close"]);
        const [left = "", right = ""] = (error ? "appmenu:close" : stdout.trim().replace(/'/g, "")).split(":");
        const side = (text) => text.split(",").filter((button) => known.has(button));
        resolve({ left: side(left), right: side(right) });
      },
    );
  });
}

function sendState() {
  if (!window || window.isDestroyed()) return;
  window.webContents.send("window:state", {
    maximized: window.isMaximized(),
    focused: window.isFocused(),
  });
}

ipcMain.handle("window:layout", () => buttonLayout());
ipcMain.handle("window:state", () => ({ maximized: window.isMaximized(), focused: window.isFocused() }));
ipcMain.on("window:minimize", () => window?.minimize());
ipcMain.on("window:maximize", () => (window?.isMaximized() ? window.unmaximize() : window?.maximize()));
ipcMain.on("window:close", () => window?.close());

app.whenReady().then(async () => {
  try {
    const url = await startServer();
    session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
    session.defaultSession.webRequest.onHeadersReceived((details, callback) =>
      callback({ responseHeaders: { ...details.responseHeaders, "Content-Security-Policy": [CSP] } }),
    );
    window = new BrowserWindow({
      width: 1440,
      height: 900,
      minWidth: 960,
      minHeight: 640,
      title: "kiseki",
      frame: false,
      backgroundColor: nativeTheme.shouldUseDarkColors ? "#101011" : "#fafafa",
      show: false,
      webPreferences: {
        preload: path.join(__dirname, "preload.cjs"),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
        spellcheck: false,
      },
    });
    for (const event of ["maximize", "unmaximize", "focus", "blur", "enter-full-screen", "leave-full-screen"])
      window.on(event, sendState);
    window.webContents.on("will-navigate", (event, destination) => {
      if (new URL(destination).origin !== url.origin) event.preventDefault();
    });
    // One window: everything, figures included, opens inside it.
    window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    window.once("ready-to-show", () => {
      window.maximize();
      window.show();
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
  if (!server || server.exitCode !== null || server.signalCode !== null || quitting) return;
  event.preventDefault();
  quitting = true;
  // Open event streams hold uvicorn's graceful shutdown: close the windows first.
  for (const open of BrowserWindow.getAllWindows()) open.destroy();
  const force = setTimeout(() => server.kill("SIGKILL"), 3000);
  server.once("exit", () => {
    clearTimeout(force);
    app.quit();
  });
  server.kill("SIGTERM");
});

process.on("SIGTERM", () => app.quit());
process.on("SIGINT", () => app.quit());
