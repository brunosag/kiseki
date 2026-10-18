// The desktop app, end to end, on a scratch store: launches it, queues two smoke runs
// through the New run form, compares them, builds a report from them, walks every view
// (the run, compare and runs views in light too), opens a figure, uses the command menu,
// then closes the window (it hides, the server stays), launches the app again (the window
// comes back), quits it and checks that the server went with it. Screenshots land in
// OUT_DIR.
//   node scripts/verify.mjs [OUT_DIR]        (default /tmp/kiseki-verify; trains for seconds)
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { connect } from "./cdp.mjs";

const out = process.argv[2] ?? "/tmp/kiseki-verify";
const port = "9341";
const home = mkdtempSync(join(tmpdir(), "kiseki-verify-"));
mkdirSync(out, { recursive: true });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const checks = [];
const check = (name, ok, detail = "") => {
  checks.push({ name, ok: !!ok, detail });
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` (${detail})` : ""}`);
};

const launch = () =>
  spawn(new URL("../kiseki-app", import.meta.url).pathname, [], {
    env: { ...process.env, KISEKI_HOME: home, KISEKI_DEBUG_PORT: port },
    stdio: "ignore",
  });
const app = launch();
let page;
for (let i = 0; i < 120 && !page; i++) {
  await sleep(500);
  page = await connect(port).catch(() => undefined);
}
if (!page) throw new Error("the app did not open a page within a minute");

const errors = [];
await page.send("Runtime.enable");
page.on("Runtime.exceptionThrown", ({ exceptionDetails }) =>
  errors.push(exceptionDetails.exception?.description ?? exceptionDetails.text),
);
page.on("Runtime.consoleAPICalled", ({ type, args }) => {
  if (type === "error") errors.push(args.map((arg) => arg.value ?? arg.description).join(" "));
});

const js = (expression) => page.evaluate(expression);
async function until(name, expression, seconds = 20) {
  for (let i = 0; i < seconds * 4; i++) {
    if (await js(expression).catch(() => false)) return true;
    await sleep(250);
  }
  check(name, false, "timed out");
  return false;
}
async function expect(name, expression) {
  if (await until(name, expression, 5)) check(name, true);
}
async function shot(name) {
  await sleep(600);
  const { data } = await page.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(out, `${name}.png`), Buffer.from(data, "base64"));
}
/** A real mouse click at the centre of the first element matching `selector` (and `text`). */
async function click(selector, text = "") {
  const box = await js(`(() => {
    const element = [...document.querySelectorAll(${JSON.stringify(selector)})]
      .find((candidate) => candidate.textContent.includes(${JSON.stringify(text)}) && candidate.getClientRects().length);
    if (!element) return null;
    element.scrollIntoView({ block: "nearest" });
    const rect = element.getBoundingClientRect();
    return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
  })()`);
  if (!box) throw new Error(`nothing to click: ${selector} ${text}`);
  for (const type of ["mouseMoved", "mousePressed", "mouseReleased"])
    await page.send("Input.dispatchMouseEvent", { type, ...box, button: "left", clickCount: 1 });
  await sleep(250);
}
async function key(key, modifiers = 0, code = key) {
  for (const type of ["keyDown", "keyUp"])
    await page.send("Input.dispatchKeyEvent", { type, key, code, modifiers, windowsVirtualKeyCode: key.length === 1 ? key.toUpperCase().charCodeAt(0) : key === "Enter" ? 13 : 27 });
  await sleep(200);
}
const go = (path) => js(`navigation.navigate(${JSON.stringify(path)}).finished.then(() => true)`);
const token = `sessionStorage.getItem("kiseki-token")`;
const api = (path) => js(`fetch("/api/${path}", { headers: { "X-Kiseki-Token": ${token} } }).then((r) => r.json())`);

try {
  await until("the app loads", `!!document.querySelector("main")`, 30);
  check("custom title bar", await js(`!!document.querySelector(".drag") && !!window.kisekiWindow`));
  check("empty store shows its empty state", await js(`document.body.innerText.includes("No runs yet")`));
  // Keys reach the page once its window is focused, which a fresh profile can take a moment to be.
  await until("the window takes focus", "document.hasFocus()");
  await sleep(300);

  // New run: SGD, smoke preset, two seeds -> two runs.
  await key("c");
  await until("New run opens", `!!document.querySelector("[role=dialog] form")`);
  await click("[aria-label=Optimizer]");
  await click("[role=option]", "SGD");
  await click("[aria-label=Preset]");
  await click("[role=option]", "smoke");
  await js(`(() => { const input = document.querySelector("#field-seed"); input.focus(); input.select(); return true; })()`);
  await page.send("Input.insertText", { text: "1, 2" });
  await sleep(200);
  check("a comma makes a grid", await js(`document.querySelector("[role=dialog] button[type=submit]").textContent.includes("Start 2 runs")`));
  await shot("new-run");
  await key("Enter", 2);
  await until("two runs complete", `fetch("/api/runs", { headers: { "X-Kiseki-Token": ${token} } }).then((r) => r.json()).then((runs) => runs.length === 2 && runs.every((run) => run.status.state === "completed"))`, 180);
  const runs = await api("runs");
  check("runs carry the form's choices", runs.every((run) => run.spec.optimizer === "sgd" && run.spec.label === "smoke") && new Set(runs.map((run) => run.spec.seed)).size === 2);
  check("a scratch store queues on this machine", runs.every((run) => (run.status.machine ?? "local") === "local"));
  await go("/runs");
  await until("the list shows both", `document.querySelectorAll("[data-row]").length === 2`);
  await shot("runs");

  await go(`/runs/${runs[0].id}`);
  await until("the run page draws its curves", `!!document.querySelector(".uplot canvas")`);
  await shot("run");

  // Select both, compare, then report.
  await go("/runs");
  for (const run of runs) await click(`[aria-label="Select run ${run.id.split("-").pop()}"]`);
  await click("button", "Compare");
  await until("compare lists both", `document.querySelectorAll("tbody tr").length === 2 && !!document.querySelector(".uplot canvas")`);
  await shot("compare");
  await click("button", "Report");
  await until("New report opens with both models", `document.querySelectorAll("[role=dialog] [role=checkbox][data-checked]").length > 2`);
  const keep = "Accuracy and per-class metrics";
  // Analysis checkboxes carry their summary as a title; models do not.
  const others = await js(`[...document.querySelectorAll("[role=dialog] label[title]")].map((label) => label.textContent).filter((text) => text !== ${JSON.stringify(keep)})`);
  for (const label of others) await click("[role=dialog] label", label);
  await shot("new-report");
  await key("Enter", 2);
  await until("the report builds", `!!document.querySelector("article h2") && document.body.innerText.includes(${JSON.stringify(keep)}) && !!document.querySelector("article table")`, 180);
  await shot("report");
  await click("article button[title='Full size']");
  check("a figure opens full size in the window", await until("the figure dialog opens", `!!document.querySelector("[data-slot=dialog-popup] img")`));
  await shot("figure");
  await key("Escape");
  await go("/reports");
  await shot("reports");
  await go("/machines");
  await until("machines are listed", `document.querySelectorAll("[aria-label=Machines] li").length > 0`);
  await shot("machines");

  // Command menu: Ctrl+K, search, Enter.
  await key("k", 2);
  await until("the command menu opens", `!!document.querySelector("[data-slot=command-dialog-popup]")`);
  await page.send("Input.insertText", { text: "reports" });
  await sleep(300);
  await shot("command-menu");
  await key("Enter");
  check("the command menu navigates", await js(`location.pathname === "/reports"`));

  // Light mode follows the system.
  await page.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
  await sleep(300);
  check("light mode follows the system", await js(`!document.documentElement.classList.contains("dark")`));
  await go(`/runs/${runs[0].id}`);
  await shot("light-run");
  await go(`/compare?runs=${runs.map((run) => run.id).join(",")}`);
  await shot("light-compare");
  await go("/runs");
  await shot("light-runs");
  await page.send("Emulation.setEmulatedMedia", { features: [] });

  check("no page errors", !errors.length, errors.join(" | "));
} catch (error) {
  check("the walk finished", false, error.message);
  await shot("failure").catch(() => {});
} finally {
  // Closing the window hides it in the tray; launching again brings it back; quitting
  // stops the server.
  await js("window.kisekiWindow.close(), true").catch(() => {});
  await expect("closing the window hides it", "document.visibilityState === 'hidden'");
  check("the server outlives the window", servers().length === 1, servers().join(" "));
  const again = launch();
  const second = await Promise.race([new Promise((resolve) => again.once("exit", (code) => resolve(code === 0))), sleep(30000).then(() => false)]);
  check("a second launch exits", second);
  await expect("and shows the window", "document.visibilityState === 'visible'");
  page.close();
  again.kill("SIGTERM"); // gone already, unless the first app died and it took over
  app.kill("SIGTERM");
  const exited = await Promise.race([new Promise((resolve) => app.once("exit", () => resolve(true))), sleep(15000).then(() => false)]);
  await sleep(4000);
  const left = servers();
  check("quitting exits the app", exited);
  check("the server stops with it", !left.length, left.join(" "));
}
console.log(`screenshots: ${out}`);
process.exit(checks.every((item) => item.ok) ? 0 : 1);

function servers() {
  return readdirSync("/proc").filter((pid) => {
    try {
      // The queue runner outlives the app by design (it exits once the queue is empty).
      return (
        /^\d+$/.test(pid) &&
        readFileSync(`/proc/${pid}/environ`, "utf8").includes(`KISEKI_HOME=${home}\0`) &&
        readFileSync(`/proc/${pid}/cmdline`, "utf8").includes("serve")
      );
    } catch {
      return false;
    }
  });
}
