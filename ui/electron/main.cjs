// The desktop shell: starts `kiseki serve`, opens a frameless window on its URL (the page
// draws its own title bar through preload.cjs), and puts kiseki in the tray. Closing the
// window hides it and keeps the server; Quit in the tray menu stops the server. One app
// per store: launching again shows the window. Training runs belong to the queue runner
// and outlive the app.
//
// Until the Python environment exists (a fresh AppImage), the window shows the setup page
// instead, from the bundled UI over kiseki://app; it lists what this machine needs
// (requirements.cjs), installs it, and then opens the app.
const { app, BrowserWindow, Menu, Tray, dialog, ipcMain, nativeImage, nativeTheme, net, protocol, session, shell } =
  require("electron");
const { execFile, spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const requirements = require("./requirements.cjs");

const ctx = requirements.context({
  packaged: app.isPackaged,
  resources: process.resourcesPath,
  repo: path.resolve(__dirname, "../.."),
});
const SETUP = "kiseki://app";
const LOCAL = ["127.0.0.1", "localhost", "[::1]"];
const CSP =
  "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; " +
  "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; " +
  "frame-ancestors 'none'";

let server;
let serverURL;
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
// Wayland where the session has it (GNOME on NixOS), else X11 (Ubuntu on Xorg, Xvfb).
if (process.env.WAYLAND_DISPLAY) app.commandLine.appendSwitch("ozone-platform", "wayland");
protocol.registerSchemesAsPrivileged([{ scheme: "kiseki", privileges: { standard: true, secure: true, supportFetchAPI: true } }]);
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

/** Origins the window may show and that may call the shell: the setup page, the server. */
function allowed(text) {
  try {
    const url = new URL(text);
    const origin = `${url.protocol}//${url.host}`;
    return origin === SETUP || (!!serverURL && origin === serverURL.origin);
  } catch {
    return false;
  }
}

function startServer() {
  if (process.env.KISEKI_SERVER_URL) return Promise.resolve(localURL(process.env.KISEKI_SERVER_URL));
  const python = requirements.python(ctx);
  if (!fs.existsSync(python)) return Promise.reject(new Error(`No Python environment at ${python}.`));
  return new Promise((resolve, reject) => {
    let output = "";
    let errors = "";
    const child = spawn(python, ["-m", "kiseki.cli", "serve"], {
      cwd: ctx.code,
      env: { ...requirements.childEnv(), PYTHONUNBUFFERED: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    });
    server = child;
    let started = false;
    const timer = setTimeout(() => {
      reject(new Error("The server did not start within 30 s."));
      child.kill("SIGKILL");
    }, 30_000);
    child.once("error", reject);
    child.once("exit", (code) => {
      clearTimeout(timer);
      if (server === child) server = null;
      reject(new Error(`The server exited (${code}). ${errors}`));
      // Once it served, losing it sends the window back to Setup, where Open retries.
      if (started && window && !quitting && !child.stoppedOnPurpose) {
        dialog.showErrorBox("kiseki stopped", `The server exited with code ${code}. ${errors.slice(-500)}`);
        openSetup().catch(() => app.quit());
      }
    });
    child.stderr.on("data", (chunk) => (errors = (errors + chunk).slice(-4000)));
    child.stdout.on("data", (chunk) => {
      output = (output + chunk).slice(-8000);
      const match = /^kiseki: serving (http\S+)$/m.exec(output);
      if (!match) return;
      clearTimeout(timer);
      started = true;
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

/** Install jobs by requirement id: their state and last lines, which the page reattaches to. */
const jobs = new Map();

function sendJob(id) {
  if (window && !window.isDestroyed()) window.webContents.send("setup:job", { id, ...jobs.get(id) });
}

function fromShell(event) {
  return event.sender === window?.webContents && allowed(event.senderFrame?.url ?? "");
}

ipcMain.handle("setup:check", async (event, fresh) => {
  if (!fromShell(event)) return [];
  const rows = await requirements.check(ctx, { fresh: fresh === true });
  return rows.map((row) => ({ ...row, job: jobs.get(row.id) ?? null }));
});

ipcMain.handle("setup:act", async (event, id) => {
  if (!fromShell(event) || typeof id !== "string" || jobs.get(id)?.state === "running") return;
  // The environment the server runs from is never changed under it: back to the setup page.
  if (id === "env" && server) {
    const { response } = await dialog.showMessageBox(window, {
      type: "question",
      buttons: ["Reinstall", "Cancel"],
      defaultId: 1,
      message: "Reinstall the Python environment?",
      detail: "The app goes back to Setup while it installs. Runs training now keep going.",
    });
    if (response !== 0) return;
    await stopServer();
    await openSetup();
  }
  const job = { state: "running", lines: [], error: null };
  jobs.set(id, job);
  sendJob(id);
  const log = (line) => {
    job.lines = [...job.lines, line].slice(-200);
    sendJob(id);
  };
  try {
    await requirements.act(id, ctx, { log, openPath: (target) => shell.openPath(target) });
    job.state = "done";
  } catch (error) {
    job.state = "failed";
    job.error = String(error.message ?? error);
  }
  sendJob(id);
});

ipcMain.handle("setup:open", async (event) => {
  if (!fromShell(event) || server) return;
  try {
    await openApp();
  } catch (error) {
    dialog.showErrorBox("Could not start kiseki", String(error.message));
    throw error;
  }
});

function stopServer() {
  if (!server || server.exitCode !== null || server.signalCode !== null) return Promise.resolve();
  const stopping = server;
  stopping.stoppedOnPurpose = true;
  server = null;
  return new Promise((resolve) => {
    const force = setTimeout(() => stopping.kill("SIGKILL"), 3000);
    stopping.once("exit", () => {
      clearTimeout(force);
      resolve();
    });
    // Open event streams hold uvicorn's graceful shutdown: leave the page first.
    window?.webContents.loadURL("about:blank").finally(() => stopping.kill("SIGTERM"));
  });
}

/** The bundled UI for the setup page, which needs no server. */
function serveBundle() {
  const dist = ctx.packaged ? path.join(ctx.bundle, "ui/dist") : path.join(ctx.code, "ui/dist");
  protocol.handle("kiseki", async (request) => {
    const file = path.normalize(path.join(dist, decodeURIComponent(new URL(request.url).pathname)));
    if (!file.startsWith(dist + path.sep)) return new Response("not found", { status: 404 });
    const response = await net.fetch(pathToFileURL(file).href);
    const headers = new Headers(response.headers);
    headers.set("Content-Security-Policy", CSP);
    return new Response(response.body, { status: response.status, headers });
  });
}

function createWindow() {
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
  // Closing hides once the tray can bring the window back; before that (setup) it quits,
  // after asking if that would cut an install short.
  window.on("close", (event) => {
    if (quitting) return;
    if (!tray) {
      const running = [...jobs.values()].some((job) => job.state === "running");
      const stop = () =>
        dialog.showMessageBoxSync(window, {
          type: "warning",
          buttons: ["Keep installing", "Quit"],
          defaultId: 0,
          message: "An install is still running",
          detail: "Quitting stops it; it starts over next time.",
        }) === 1;
      if (running && !stop()) event.preventDefault();
      return;
    }
    event.preventDefault();
    window.hide();
  });
  for (const event of ["maximize", "unmaximize", "focus", "blur", "enter-full-screen", "leave-full-screen"])
    window.on(event, sendState);
  window.webContents.on("will-navigate", (event, destination) => {
    if (!allowed(destination)) event.preventDefault();
  });
  // One window: everything, figures included, opens inside it.
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.once("ready-to-show", () => {
    window.maximize();
    window.show();
  });
}

async function openSetup() {
  await window.loadURL(`${SETUP}/setup.html`);
}

async function openApp() {
  serverURL = await startServer();
  if (!tray) {
    tray = new Tray(nativeImage.createFromPath(path.join(__dirname, "tray.png")));
    tray.setToolTip("kiseki");
    tray.on("click", show);
    updateTray();
  }
  await window.loadURL(serverURL.href);
}

app.whenReady().then(async () => {
  if (!primary) return;
  try {
    session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
    session.defaultSession.webRequest.onHeadersReceived((details, callback) =>
      callback({ responseHeaders: { ...details.responseHeaders, "Content-Security-Policy": [CSP] } }),
    );
    serveBundle();
    requirements.refreshDesktopEntry(ctx);
    createWindow();
    if (requirements.envReady(ctx) || process.env.KISEKI_SERVER_URL) {
      // A server that won't start leaves the app at Setup, where a reinstall can fix it.
      try {
        await openApp();
        return;
      } catch (error) {
        dialog.showErrorBox("Could not start kiseki", String(error.message));
      }
    }
    await openSetup();
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
