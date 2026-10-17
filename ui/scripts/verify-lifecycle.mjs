// Run with KISEKI_DEBUG_PORT pointing at an Electron instance that owns its server.
import { evaluate as ev, waitFor, close } from "./cdp.mjs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { writeFile } from "node:fs/promises";
const exec = promisify(execFile);
try {
  await waitFor('!!document.querySelector(".run-table")');
  const info = await ev(
    '({origin:location.origin,nodeIntegration:typeof require,firstViewReadyMs:performance.getEntriesByName("kiseki:runs:ready").map(e=>e.startTime+e.duration)[0],navigation:performance.getEntriesByType("navigation").map(e=>({domContentLoadedMs:e.domContentLoadedEventEnd,loadMs:e.loadEventEnd}))})',
  );
  const { stdout } = await exec("ps", ["-eo", "pid=,ppid=,args="]);
  const processes = stdout
    .trim()
    .split("\n")
    .map((line) => {
      const m = /^\s*(\d+)\s+(\d+)\s+(.*)$/.exec(line);
      return { pid: Number(m[1]), ppid: Number(m[2]), args: m[3] };
    });
  const main = processes.find(
    (p) =>
      p.args.startsWith("/nix/store/") &&
      p.args.endsWith("/ui/electron/main.cjs"),
  );
  const child = processes.find(
    (p) => p.ppid === main.pid && p.args.includes("-m kiseki.cli serve"),
  );
  if (!child)
    throw new Error("This Electron instance does not own a kiseki server");
  const live = (pid) => {
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  };
  const start = Date.now();
  process.kill(main.pid, "SIGTERM");
  for (let i = 0; i < 70 && (live(main.pid) || live(child.pid)); i++)
    await new Promise((r) => setTimeout(r, 100));
  const result = {
    ...info,
    electronPid: main.pid,
    serverPid: child.pid,
    serverWasChild: true,
    electronExited: !live(main.pid),
    serverExited: !live(child.pid),
    quitMs: Date.now() - start,
  };
  await writeFile("ui/out/lifecycle.json", JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
  if (!result.serverExited || !result.electronExited)
    throw new Error("Electron or its server survived quit");
} finally {
  close();
}
