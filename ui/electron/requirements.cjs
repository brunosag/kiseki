// What the app needs on this machine, as rows the setup view lists: how each is checked,
// what its state means, and the one action that fixes it. Runs in Electron's main process
// because the Python side may not exist yet. Row builders are pure (requirements.test.cjs);
// `check` and `act` do the I/O.
//
// Packaged (the AppImage), the bundle's Python code (resources/kiseki, named by its BUILD
// id) is copied to ~/.local/share/kiseki-app/<BUILD>/ with its own .venv: a new version
// never changes the code a running server, queue runner or worker uses. A checkout uses its
// own .venv.
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const DRIVER = 580; // JAX's CUDA 13 wheels need at least this driver
const CAPABILITY = 7.5; // CUDA 13 dropped Maxwell, Pascal and Volta
const TITLE = "Train optimizers and compare the networks they make";
const PROBE =
  "from kiseki.worker import runtime; runtime.ensure_driver_path(); import platform, jax; " +
  "print(platform.python_version(), jax.default_backend())";
/** "NVIDIA GeForce RTX 3050 Laptop GPU" → "RTX 3050 Laptop GPU" (as ui/src/lib/format.ts). */
function gpuName(name) {
  return name.replace(/^NVIDIA (GeForce )?/, "");
}

/** The first GPU from `nvidia-smi --query-gpu=name,driver_version[,compute_cap]` (drivers
 * before 510 can't report the capability: null). */
function parseGpu(text) {
  const fields = (text.split("\n")[0] ?? "").split(",").map((field) => field.trim());
  if (fields.length < 2 || fields.length > 3 || !/^\d+\./.test(fields[1])) return null;
  const [name, driver, capability] = fields;
  return { name, driver, major: Number.parseInt(driver, 10), capability: capability ? Number(capability) : null };
}

function usable(gpu) {
  return !!gpu && gpu.major >= DRIVER && gpu.capability !== null && gpu.capability >= CAPABILITY;
}

function gpuRow(gpu, { drivers }) {
  const row = { id: "gpu", title: "NVIDIA GPU", required: false };
  const open = drivers ? { action: { label: "Open Additional Drivers", kind: "open" } } : {};
  if (!gpu) return { ...row, state: "missing", detail: "No NVIDIA driver found: runs train on the CPU", ...open };
  const name = gpuName(gpu.name);
  const version = `driver ${gpu.driver}`;
  if (gpu.capability !== null && gpu.capability < CAPABILITY)
    return {
      ...row,
      state: "warning",
      version,
      detail: `${name} is too old for CUDA 13 (compute capability ${gpu.capability}, needs ${CAPABILITY}): runs train on the CPU`,
    };
  if (gpu.major < DRIVER)
    return {
      ...row,
      state: "warning",
      version,
      detail: `${name}: driver ${gpu.major} is older than ${DRIVER}, which GPU training needs`,
      ...open,
    };
  return { ...row, state: "ok", version, detail: `${name}: runs train on it` };
}

function envRow({ ready, python, backend, error, gpuUsable }) {
  const row = { id: "env", title: "Python environment", required: true };
  if (!ready)
    return {
      ...row,
      state: "missing",
      detail: error ?? `Python, JAX and kiseki, about ${gpuUsable ? "4 GB with GPU support" : "1 GB"}`,
      action: { label: error ? "Reinstall" : "Install", kind: "install" },
    };
  const reinstall = { action: { label: "Reinstall", kind: "install" } };
  if (backend !== "gpu" && gpuUsable)
    return {
      ...row,
      state: "warning",
      version: `Python ${python}`,
      detail: "JAX runs on the CPU although the GPU could train: reinstall for GPU support",
      ...reinstall,
    };
  return { ...row, state: "ok", version: `Python ${python}`, detail: `JAX runs on the ${backend.toUpperCase()}`, ...reinstall };
}

function toolsRow({ ssh, rsync, apt }) {
  const row = { id: "tools", title: "SSH and rsync", required: false };
  const missing = [ssh ? null : "ssh", rsync ? null : "rsync"].filter(Boolean);
  if (!missing.length)
    return { ...row, state: "ok", version: `${ssh} · rsync ${rsync}`, detail: "Send runs to remote machines" };
  return {
    ...row,
    state: "missing",
    detail: `${missing.join(" and ")} ${missing.length > 1 ? "are" : "is"} missing: needed to send runs to remote machines`,
    ...(apt ? { action: { label: "Install", kind: "install" } } : {}),
  };
}

function machinesRow(info) {
  const row = { id: "machines", title: "Remote machines", required: false };
  if (!info) return { ...row, state: "missing", detail: "Install the Python environment first" };
  const open = { action: { label: "Open folder", kind: "open" } };
  if (!info.machines.length)
    return {
      ...row,
      state: "missing",
      detail: `None yet: copy machines.toml from another machine to ${info.path}, with SSH access to its hosts`,
      ...open,
    };
  const version = info.machines.join(", ");
  if (info.notice.length && !info.contact)
    return {
      ...row,
      state: "warning",
      version,
      detail: `${info.notice.join(", ")} posts a notice with your name and email: add [contact] to ${info.path}`,
      ...open,
    };
  return { ...row, state: "ok", version, detail: info.path, ...open };
}

/** A desktop entry's quoted argument: `"`, `` ` ``, `$` and `\\` escaped, then the
 * backslashes once more for the string, and `%` doubled for field codes. */
function execArgument(text) {
  return `"${text.replace(/["`$\\]/g, "\\$&").replaceAll("\\", "\\\\").replaceAll("%", "%%")}"`;
}

function desktopEntry(appImage, icon) {
  return [
    "[Desktop Entry]",
    "Type=Application",
    "Name=kiseki",
    `Comment=${TITLE}`,
    `Exec=${execArgument(appImage)}`,
    `Icon=${icon}`,
    "Categories=Development;Science;",
    "StartupNotify=true",
    "Terminal=false",
    "",
  ].join("\n");
}

function desktopRow(installed) {
  const row = { id: "desktop", title: "App menu entry", required: false };
  return installed
    ? { ...row, state: "ok", detail: "kiseki is in the app menu, pointing at this AppImage" }
    : { ...row, state: "missing", detail: "Add kiseki to the app menu", action: { label: "Add", kind: "install" } };
}

/** Where this app's Python code and tools are. Packaged: the bundle (resources/kiseki), a
 * pinned uv beside it, and the per-BUILD directory the bundle is copied to. */
function context({ packaged, resources, repo, home = os.homedir(), env = process.env }) {
  const data = path.join(env.XDG_DATA_HOME || path.join(home, ".local/share"));
  // APPIMAGE is ours only when packaged: a checkout launched from another AppImage inherits it.
  const base = { packaged, home, data, appImage: (packaged && env.APPIMAGE) || null };
  if (!packaged) return { ...base, code: repo, uv: "uv", icon: path.join(repo, "ui/electron/icon.svg") };
  const bundle = path.join(resources, "kiseki");
  const build = fs.readFileSync(path.join(bundle, "BUILD"), "utf8").trim();
  return {
    ...base,
    bundle,
    code: path.join(data, "kiseki-app", build),
    uv: path.join(resources, "bin/uv"),
    icon: path.join(resources, "icon.svg"),
  };
}

/** This process's environment without the AppImage's: its mount vanishes when the app quits,
 * and the server, queue runner and workers outlive it. */
function childEnv(env = process.env) {
  const mount = env.APPDIR;
  const clean = {};
  for (const [key, value] of Object.entries(env)) {
    if (["APPDIR", "APPIMAGE", "ARGV0", "OWD", "ELECTRON_RUN_AS_NODE"].includes(key) || key.startsWith("APPIMAGE_")) continue;
    if (!mount || !value.includes(mount)) clean[key] = value;
    else if (key.endsWith("PATH")) {
      const kept = value.split(":").filter((part) => part && !part.startsWith(mount));
      if (kept.length) clean[key] = kept.join(":");
    }
  }
  return clean;
}

/** Runs a command; resolves {code, stdout, stderr}, code null when it could not start. */
function exec(command, args, { cwd, env = childEnv(), timeout = 10_000, onLine } = {}) {
  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";
    let child;
    try {
      child = spawn(command, args, { cwd, env, stdio: ["ignore", "pipe", "pipe"] });
    } catch {
      resolve({ code: null, stdout, stderr });
      return;
    }
    const timer = timeout ? setTimeout(() => child.kill("SIGKILL"), timeout) : null;
    // Whole lines to onLine: a chunk can end mid-line, so each stream keeps its remainder.
    const pending = { stdout: "", stderr: "" };
    const lines = (stream, chunk, end = false) => {
      if (!onLine) return;
      const parts = (pending[stream] + chunk).split(/[\r\n]+/);
      pending[stream] = end ? "" : parts.pop();
      for (const line of parts) if (line.trim()) onLine(line.trimEnd());
    };
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      lines("stdout", String(chunk));
    });
    child.stderr.on("data", (chunk) => {
      stderr = (stderr + chunk).slice(-20_000);
      lines("stderr", String(chunk));
    });
    child.once("error", () => resolve({ code: null, stdout, stderr }));
    child.once("close", (code) => {
      if (timer) clearTimeout(timer);
      lines("stdout", "", true);
      lines("stderr", "", true);
      resolve({ code, stdout, stderr });
    });
  });
}

function python(ctx) {
  return path.join(ctx.code, ".venv/bin/python");
}

/** The environment's last probe, kept in its .venv with the time its packages last changed. */
function probeFile(ctx) {
  return path.join(ctx.code, ".venv/kiseki-probe.json");
}

function packagesChanged(ctx) {
  const lib = path.join(ctx.code, ".venv/lib");
  try {
    const version = fs.readdirSync(lib).find((name) => name.startsWith("python"));
    return fs.statSync(path.join(lib, version, "site-packages")).mtimeMs;
  } catch {
    return null;
  }
}

/** Whether the app can start its server: the environment exists (and, packaged, its
 * install finished, which the probe file marks). */
function envReady(ctx) {
  return fs.existsSync(python(ctx)) && (!ctx.packaged || fs.existsSync(probeFile(ctx)));
}

/** Probing imports JAX (seconds), so its answer is kept until the packages change; `fresh`
 * probes again. Only a finished install (`installed`) marks a bundle's environment ready:
 * a probe of a half-synced one must not. */
async function envState(ctx, fresh, { installed = false } = {}) {
  if (!fs.existsSync(python(ctx))) return { ready: false };
  if (!fresh) {
    if (!envReady(ctx)) return { ready: false }; // a bundle's install never finished
    try {
      const saved = JSON.parse(fs.readFileSync(probeFile(ctx), "utf8"));
      if (saved.packages === packagesChanged(ctx)) return { ready: true, python: saved.python, backend: saved.backend };
    } catch {}
  }
  const result = await exec(python(ctx), ["-c", PROBE], {
    cwd: ctx.code,
    env: { ...childEnv(), XLA_PYTHON_CLIENT_PREALLOCATE: "false", TF_CPP_MIN_LOG_LEVEL: "3" },
    timeout: 120_000,
  });
  const [version, backend] = result.stdout.trim().split(/\s+/);
  if (result.code !== 0 || !backend)
    return { ready: false, error: `Broken: ${result.stderr.trim().split("\n").pop() || "it did not start"}` };
  if (installed || !ctx.packaged || fs.existsSync(probeFile(ctx)))
    fs.writeFileSync(probeFile(ctx), JSON.stringify({ packages: packagesChanged(ctx), python: version, backend }));
  else return { ready: false };
  return { ready: true, python: version, backend };
}

async function machinesInfo(ctx) {
  const result = await exec(python(ctx), ["-m", "kiseki.cli", "machines", "--config", "--json"], { cwd: ctx.code });
  try {
    return JSON.parse(result.stdout);
  } catch {
    return null;
  }
}

async function gpuInfo() {
  for (const fields of ["name,driver_version,compute_cap", "name,driver_version"]) {
    const result = await exec("nvidia-smi", [`--query-gpu=${fields}`, "--format=csv,noheader"]);
    if (result.code === null) return null;
    if (result.code === 0) return parseGpu(result.stdout);
  }
  return null;
}

async function has(command) {
  return (await exec("sh", ["-c", `command -v ${command}`])).code === 0;
}

function desktopPaths(ctx) {
  return {
    entry: path.join(ctx.data, "applications/kiseki.desktop"),
    icon: path.join(ctx.data, "icons/hicolor/scalable/apps/kiseki.svg"),
  };
}

function writeDesktopEntry(ctx) {
  const { entry, icon } = desktopPaths(ctx);
  fs.mkdirSync(path.dirname(icon), { recursive: true });
  fs.copyFileSync(ctx.icon, icon);
  fs.mkdirSync(path.dirname(entry), { recursive: true });
  fs.writeFileSync(entry, desktopEntry(ctx.appImage, icon));
}

/** An AppImage moves with each update: keep an existing entry pointing at this one. */
function refreshDesktopEntry(ctx) {
  const { entry, icon } = desktopPaths(ctx);
  if (!ctx.appImage || !fs.existsSync(entry)) return;
  if (fs.readFileSync(entry, "utf8") !== desktopEntry(ctx.appImage, icon)) writeDesktopEntry(ctx);
}

/** Every row, required first. `fresh` probes the environment again. */
async function check(ctx, { fresh = false } = {}) {
  const [gpu, drivers, ssh, rsync, apt, env] = await Promise.all([
    gpuInfo(),
    has("software-properties-gtk"),
    exec("ssh", ["-V"]).then(({ code, stderr }) => (code === 0 ? stderr.trim().split(/[ ,]/)[0] : null)),
    exec("rsync", ["--version"]).then(({ code, stdout }) => (code === 0 ? stdout.match(/version (\S+)/)?.[1] : null)),
    has("pkexec").then(async (pkexec) => pkexec && (await has("apt-get"))),
    envState(ctx, fresh),
  ]);
  const rows = [
    envRow({ ...env, gpuUsable: usable(gpu) }),
    gpuRow(gpu, { drivers }),
    toolsRow({ ssh, rsync: rsync ?? null, apt }),
    machinesRow(env.ready ? await machinesInfo(ctx) : null),
  ];
  if (ctx.appImage) rows.push(desktopRow(fs.existsSync(desktopPaths(ctx).entry)));
  return rows;
}

function freeBytes(directory) {
  fs.mkdirSync(directory, { recursive: true });
  const stats = fs.statfsSync(directory);
  return stats.bavail * stats.bsize;
}

async function installEnv(ctx, log) {
  const gpu = usable(await gpuInfo());
  const needed = gpu ? 6e9 : 2e9;
  const cache = process.env.UV_CACHE_DIR || process.env.XDG_CACHE_HOME || path.join(ctx.home, ".cache");
  for (const directory of [ctx.data, cache]) {
    const free = freeBytes(directory);
    if (free < needed)
      throw new Error(`${directory} has ${(free / 1e9).toFixed(1)} GB free; the environment needs ${needed / 1e9} GB`);
  }
  if (ctx.packaged && !fs.existsSync(ctx.code)) {
    const partial = `${ctx.code}.partial`;
    fs.rmSync(partial, { recursive: true, force: true });
    fs.cpSync(ctx.bundle, partial, { recursive: true });
    fs.renameSync(partial, ctx.code);
  }
  fs.rmSync(probeFile(ctx), { force: true });
  // A checkout keeps its dev tools; a bundle installs only what runs.
  const args = ["sync", "--frozen", ...(ctx.packaged ? ["--no-dev"] : []), ...(gpu ? ["--extra", "cuda"] : [])];
  log(`uv ${args.join(" ")}${gpu ? "" : " (no usable NVIDIA GPU: CPU only)"}`);
  const result = await exec(ctx.uv, args, { cwd: ctx.code, timeout: 0, onLine: log });
  if (result.code !== 0) throw new Error(result.code === null ? `could not run ${ctx.uv}` : "uv sync failed");
  const state = await envState(ctx, true, { installed: true });
  if (!state.ready) throw new Error(state.error);
  log(`Python ${state.python}, JAX on the ${state.backend.toUpperCase()}`);
}

/** The row's action: an install (output lines go to `log`) or opening something. */
async function act(id, ctx, { log, openPath }) {
  if (id === "env") return installEnv(ctx, log);
  if (id === "gpu") {
    spawn("software-properties-gtk", ["--open-tab=4"], { env: childEnv(), detached: true, stdio: "ignore" }).unref();
    return;
  }
  if (id === "tools") {
    const apt = ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "-o", "DPkg::Lock::Timeout=300"];
    const result = await exec("pkexec", [...apt, "install", "-y", "openssh-client", "rsync"], { timeout: 0, onLine: log });
    if (result.code !== 0) throw new Error(result.code === 126 ? "authorization was refused" : "apt-get failed");
    return;
  }
  if (id === "machines") {
    const info = await machinesInfo(ctx);
    if (!info) throw new Error("the Python environment is not ready");
    fs.mkdirSync(path.dirname(info.path), { recursive: true });
    return openPath(path.dirname(info.path));
  }
  if (id === "desktop" && ctx.appImage) return writeDesktopEntry(ctx);
  throw new Error(`nothing to do for ${id}`);
}

module.exports = {
  act,
  check,
  childEnv,
  context,
  desktopEntry,
  envReady,
  envRow,
  exec,
  gpuRow,
  machinesRow,
  parseGpu,
  python,
  refreshDesktopEntry,
  toolsRow,
};
