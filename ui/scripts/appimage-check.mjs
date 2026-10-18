// The AppImage as a new user meets it, on a fresh HOME with a bare PATH under Xvfb (run it
// on the vps, an Ubuntu 24.04 like the users'): the setup page lists the requirements, the
// Python environment installs from it, Open kiseki starts the app, and a smoke run trains.
//   node scripts/appimage-check.mjs APPIMAGE [OUT_DIR]   (needs xvfb-run; downloads ~1 GB)
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { connect } from "./cdp.mjs";

const appImage = resolve(process.argv[2] ?? "");
const out = process.argv[3] ?? "/tmp/kiseki-appimage-check";
const port = "9361";
const home = mkdtempSync(join(tmpdir(), "kiseki-home-"));
mkdirSync(out, { recursive: true });
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
const checks = [];
const check = (name, ok, detail = "") => {
  checks.push(!!ok);
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` (${detail})` : ""}`);
};
const env = { HOME: home, PATH: "/usr/bin:/bin", LANG: "C.UTF-8", KISEKI_DEBUG_PORT: port };
const app = spawn("xvfb-run", ["-a", "-s", "-screen 0 1600x1000x24", appImage], { env, stdio: "ignore", detached: true });

let page;
for (let i = 0; i < 120 && !page; i++) {
  await sleep(500);
  page = await connect(port).catch(() => undefined);
}
if (!page) throw new Error("no page within a minute");
const js = (expression) => page.evaluate(expression);
async function until(expression, seconds) {
  for (let i = 0; i < seconds * 2; i++) {
    if (await js(expression).catch(() => false)) return true;
    await sleep(500);
  }
  return false;
}
async function shot(name) {
  await sleep(500);
  const { data } = await page.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(out, `${name}.png`), Buffer.from(data, "base64"));
}
const rows = () => js(`[...document.querySelectorAll('[aria-label=Requirements] > li')].map((li) => li.innerText.split("\\n").join(" | "))`);
const click = (row, label) =>
  js(`(() => {
    const li = [...document.querySelectorAll('[aria-label=Requirements] > li')].find((li) => li.innerText.startsWith(${JSON.stringify(row)}));
    const button = [...(li?.querySelectorAll('button') ?? [])].find((b) => b.innerText.trim() === ${JSON.stringify(label)});
    button?.click();
    return !!button;
  })()`);

try {
  check("the setup page opens", await until("location.href.startsWith('kiseki://app/setup.html') && !!document.querySelector('[aria-label=Requirements]')", 60));
  console.log((await rows()).join("\n"));
  await shot("1-setup");
  check("Open kiseki waits for the environment", await js("[...document.querySelectorAll('button')].find((b) => b.innerText.includes('Open kiseki')).disabled"));
  check("the environment has an Install button", await click("Python environment", "Install"));
  const installed = await until(
    `[...document.querySelectorAll('[aria-label=Requirements] > li')][0].innerText.includes('JAX runs on')`,
    1200,
  );
  await shot("2-installed");
  console.log((await rows()).join("\n"));
  check("the environment installs", installed);
  await js("[...document.querySelectorAll('button')].find((b) => b.innerText.includes('Open kiseki')).click()");
  check("Open kiseki opens the app", await until("location.href.startsWith('http://127.0.0.1') && !!document.querySelector('nav, aside, [data-sidebar]')", 60));
  await shot("3-app");
  // A smoke run, from the environment the app installed.
  const code = join(home, ".local/share/kiseki-app", readdirSync(join(home, ".local/share/kiseki-app")).find((name) => !name.endsWith(".partial")));
  const queued = spawnSync(join(code, ".venv/bin/kiseki"), ["run", "--preset", "smoke", "-o", "sgd", "--json"], { env, encoding: "utf8" });
  check("a smoke run queues", queued.status === 0, queued.stderr.trim().split("\n").pop());
  check("and trains", await until("fetch('/api/runs', {headers: {'X-Kiseki-Token': sessionStorage.getItem('kiseki-token')}}).then((r) => r.json()).then((runs) => runs.some((run) => run.status.state === 'completed'))", 600));
  await js("location.assign('/setup')");
  check("Setup is in the app too", await until("!!document.querySelector('[aria-label=Requirements]')", 30));
  await shot("4-app-setup");
} catch (error) {
  check("the walk finished", false, error.message);
  await shot("failure").catch(() => {});
} finally {
  page.close();
  process.kill(-app.pid, "SIGTERM");
}
console.log(`home: ${home}\nscreenshots: ${out}`);
process.exit(checks.every(Boolean) ? 0 : 1);
