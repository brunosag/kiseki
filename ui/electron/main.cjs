// The desktop shell: starts `kiseki serve`, opens a frameless window on its URL (the page
// draws its own title bar through preload.cjs), and puts kiseki in the tray. Closing the
// window hides it and keeps the server; Quit in the tray menu stops the server. One app
// per store: launching again shows the window. Training runs belong to the queue runner
// and outlive the app.
const { app, BrowserWindow, Menu, Tray, dialog, ipcMain, nativeImage, nativeTheme, session } = require("electron");
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
let tray;
let trayItems = [];
let quitting = false;

app.setName("kiseki");
// The Wayland app id, which GNOME matches to kiseki.desktop for the dock icon.
app.setDesktopName("kiseki.desktop");
// A store of its own (KISEKI_HOME, as in scratch checks) gets its own saved settings, so
// it never inherits the real app's choices, such as the machine new runs go to.
if (process.env.KISEKI_HOME) app.setPath("userData", path.join(path.resolve(process.env.KISEKI_HOME), "app"));
// The lock lives in userData, so a scratch store runs beside the real app.
const primary = app.requestSingleInstanceLock();
if (!primary) app.quit();
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

function show() {
  if (!window || window.isDestroyed()) return;
  if (window.isMinimized()) window.restore();
  window.show();
  window.focus();
}

/** A tray item the page sent, rebuilt from the fields a menu item may have. */
function trayItem(item, depth = 0) {
  // "&" marks a mnemonic in Electron's labels; "&&" is a literal one.
  const label = String(item?.label ?? "").slice(0, 200).replaceAll("&", "&&");
  const action = item?.action;
  const open = typeof action?.open === "string" && action.open.startsWith("/") && !action.open.startsWith("//");
  const submenu = depth === 0 && Array.isArray(item?.submenu) ? item.submenu.slice(0, 20).map((sub) => trayItem(sub, 1)) : null;
  if (submenu) return { label, submenu };
  if (open || action?.newRun === true) return { label, click: () => trayAction(open ? { open: action.open } : { newRun: true }) };
  return { label, enabled: false };
}

function trayAction(action) {
  show();
  window?.webContents.send("tray:action", action);
}

function updateTray() {
  tray?.setContextMenu(
    Menu.buildFromTemplate([
      { label: "Open kiseki", click: show },
      { label: "New run…", click: () => trayAction({ newRun: true }) },
      ...(trayItems.length ? [{ type: "separator" }, ...trayItems] : []),
      { type: "separator" },
      { label: "Quit kiseki", click: () => app.quit() },
    ]),
  );
}

ipcMain.on("tray:items", (event, items) => {
  if (event.sender !== window?.webContents || !Array.isArray(items)) return;
  trayItems = items.slice(0, 30).map((item) => trayItem(item));
  updateTray();
});
app.on("second-instance", show);

ipcMain.handle("window:layout", () => buttonLayout());
ipcMain.handle("window:state", () => ({ maximized: window.isMaximized(), focused: window.isFocused() }));
ipcMain.on("window:minimize", () => window?.minimize());
ipcMain.on("window:maximize", () => (window?.isMaximized() ? window.unmaximize() : window?.maximize()));
ipcMain.on("window:close", () => window?.close());

app.whenReady().then(async () => {
  if (!primary) return;
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
    // Closing hides; the tray brings the window back, and Quit there really quits.
    window.on("close", (event) => {
      if (quitting) return;
      event.preventDefault();
      window.hide();
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
    tray = new Tray(nativeImage.createFromPath(path.join(__dirname, "tray.png")));
    tray.setToolTip("kiseki");
    tray.on("click", show);
    updateTray();
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
  const first = !quitting;
  quitting = true;
  if (!first || !server || server.exitCode !== null || server.signalCode !== null) return;
  event.preventDefault();
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
