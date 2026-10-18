const assert = require("node:assert/strict");
const { test } = require("node:test");
const { desktopEntry, envRow, exec, gpuRow, machinesRow, parseGpu, toolsRow } = require("./requirements.cjs");

test("nvidia-smi's answer, first GPU", () => {
  assert.deepEqual(parseGpu("NVIDIA GeForce RTX 3050 Laptop GPU, 580.95.05, 8.6\n"), {
    name: "NVIDIA GeForce RTX 3050 Laptop GPU",
    driver: "580.95.05",
    major: 580,
    capability: 8.6,
  });
  assert.deepEqual(parseGpu("NVIDIA GeForce GTX 1080, 470.256.02"), {
    name: "NVIDIA GeForce GTX 1080",
    driver: "470.256.02",
    major: 470,
    capability: null,
  });
  assert.equal(parseGpu("No devices were found\n"), null);
  assert.equal(parseGpu(""), null);
});

test("a GPU JAX's CUDA 13 build can use", () => {
  const gpu = parseGpu("NVIDIA GeForce RTX 3050 Laptop GPU, 580.95.05, 8.6");
  assert.deepEqual(gpuRow(gpu, { drivers: true }), {
    id: "gpu",
    title: "NVIDIA GPU",
    required: false,
    state: "ok",
    version: "driver 580.95.05",
    detail: "RTX 3050 Laptop GPU: runs train on it",
  });
});

test("an old driver, an old GPU, no GPU", () => {
  const old = gpuRow(parseGpu("NVIDIA GeForce RTX 3060, 550.120, 8.6"), { drivers: true });
  assert.equal(old.state, "warning");
  assert.equal(old.detail, "RTX 3060: driver 550 is older than 580, which GPU training needs");
  assert.deepEqual(old.action, { label: "Open Additional Drivers", kind: "open" });

  const pascal = gpuRow(parseGpu("NVIDIA GeForce GTX 1080, 580.95.05, 6.1"), { drivers: true });
  assert.equal(pascal.state, "warning");
  assert.equal(pascal.detail, "GTX 1080 is too old for CUDA 13 (compute capability 6.1, needs 7.5): runs train on the CPU");
  assert.equal(pascal.action, undefined);

  const unknown = gpuRow(parseGpu("NVIDIA GeForce GTX 1080, 470.256.02"), { drivers: true });
  assert.equal(unknown.detail, "GTX 1080: driver 470 is older than 580, which GPU training needs");

  const none = gpuRow(null, { drivers: false });
  assert.equal(none.state, "missing");
  assert.equal(none.detail, "No NVIDIA driver found: runs train on the CPU");
  assert.equal(none.action, undefined);
});

test("the environment: missing, ready, or without the GPU it could use", () => {
  assert.deepEqual(envRow({ ready: false, gpuUsable: true }), {
    id: "env",
    title: "Python environment",
    required: true,
    state: "missing",
    detail: "Python, JAX and kiseki, about 4 GB with GPU support",
    action: { label: "Install", kind: "install" },
  });
  assert.equal(envRow({ ready: false, gpuUsable: false }).detail, "Python, JAX and kiseki, about 1 GB");
  assert.deepEqual(envRow({ ready: true, python: "3.14.2", backend: "gpu", gpuUsable: true }), {
    id: "env",
    title: "Python environment",
    required: true,
    state: "ok",
    version: "Python 3.14.2",
    detail: "JAX runs on the GPU",
    action: { label: "Reinstall", kind: "install" },
  });
  const cpu = envRow({ ready: true, python: "3.14.2", backend: "cpu", gpuUsable: true });
  assert.equal(cpu.state, "warning");
  assert.equal(cpu.detail, "JAX runs on the CPU although the GPU could train: reinstall for GPU support");
  assert.equal(envRow({ ready: true, python: "3.14.2", backend: "cpu", gpuUsable: false }).state, "ok");
  assert.equal(envRow({ ready: true, python: "3.14.2", backend: "cpu", gpuUsable: false }).detail, "JAX runs on the CPU");
});

test("SSH and rsync for remote machines", () => {
  assert.deepEqual(toolsRow({ ssh: "OpenSSH_9.6p1", rsync: "3.2.7", apt: true }), {
    id: "tools",
    title: "SSH and rsync",
    required: false,
    state: "ok",
    version: "OpenSSH_9.6p1 · rsync 3.2.7",
    detail: "Send runs to remote machines",
  });
  const missing = toolsRow({ ssh: "OpenSSH_9.6p1", rsync: null, apt: true });
  assert.equal(missing.state, "missing");
  assert.equal(missing.detail, "rsync is missing: needed to send runs to remote machines");
  assert.deepEqual(missing.action, { label: "Install", kind: "install" });
  assert.equal(toolsRow({ ssh: null, rsync: null, apt: false }).action, undefined);
});

test("remote machines need the contact a shared server's notice names", () => {
  const path = "/home/ada/.config/kiseki/machines.toml";
  assert.deepEqual(machinesRow(null), {
    id: "machines",
    title: "Remote machines",
    required: false,
    state: "missing",
    detail: "Install the Python environment first",
  });
  assert.deepEqual(machinesRow({ path, exists: false, machines: [], notice: [], contact: false }), {
    id: "machines",
    title: "Remote machines",
    required: false,
    state: "missing",
    detail: `None yet: copy machines.toml from another machine to ${path}, with SSH access to its hosts`,
    action: { label: "Open folder", kind: "open" },
  });
  assert.deepEqual(machinesRow({ path, exists: true, machines: ["lugia", "vps"], notice: ["lugia"], contact: true }), {
    id: "machines",
    title: "Remote machines",
    required: false,
    state: "ok",
    version: "lugia, vps",
    detail: path,
    action: { label: "Open folder", kind: "open" },
  });
  const nameless = machinesRow({ path, exists: true, machines: ["lugia"], notice: ["lugia"], contact: false });
  assert.equal(nameless.state, "warning");
  assert.equal(nameless.detail, `lugia posts a notice with your name and email: add [contact] to ${path}`);
});

test("the desktop entry points at the AppImage where it is now", () => {
  assert.equal(
    desktopEntry('/home/ada/Apps/kiseki 100%$"0.3".AppImage', "/home/ada/.local/share/icons/hicolor/scalable/apps/kiseki.svg"),
    [
      "[Desktop Entry]",
      "Type=Application",
      "Name=kiseki",
      "Comment=Train optimizers and compare the networks they make",
      String.raw`Exec="/home/ada/Apps/kiseki 100%%\\$\\"0.3\\".AppImage"`,
      "Icon=/home/ada/.local/share/icons/hicolor/scalable/apps/kiseki.svg",
      "Categories=Development;Science;",
      "StartupNotify=true",
      "Terminal=false",
      "",
    ].join("\n"),
  );
});

test("output arrives as whole lines, however the chunks split", async () => {
  const lines = [];
  const result = await exec("sh", ["-c", "printf 'Down'; sleep 0.1; printf 'loaded jax\\n + numpy'; printf 'err\\n' >&2"], {
    onLine: (line) => lines.push(line),
  });
  assert.equal(result.code, 0);
  assert.deepEqual(lines.sort(), [" + numpy", "Downloaded jax", "err"]);
});
