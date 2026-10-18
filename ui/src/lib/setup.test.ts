import assert from "node:assert/strict";
import { test } from "node:test";
import { canOpen, setupHealth } from "./setup.ts";
import type { Requirement } from "./shell.ts";

function row(id: string, required: boolean, state: Requirement["state"], job: Requirement["job"] = null): Requirement {
  return { id, title: id, required, state, detail: "", job };
}

test("the app opens once every required row is ready and nothing installs", () => {
  assert.equal(canOpen([row("env", true, "ok"), row("gpu", false, "missing")]), true);
  assert.equal(canOpen([row("env", true, "warning"), row("gpu", false, "ok")]), true);
  assert.equal(canOpen([row("env", true, "missing")]), false);
  assert.equal(canOpen([row("env", true, "ok"), row("tools", false, "missing", { state: "running", lines: [], error: null })]), false);
  assert.equal(canOpen(null), false);
});

test("the nav's dot: red for a required row, amber for a fixable warning, none for missing extras", () => {
  const fixable = (r: Requirement): Requirement => ({ ...r, action: { label: "Reinstall", kind: "install" } });
  assert.equal(setupHealth([row("env", true, "ok"), row("machines", false, "missing")]), null);
  assert.equal(setupHealth([fixable(row("env", true, "warning"))]), "warning");
  assert.equal(setupHealth([row("env", true, "ok"), row("gpu", false, "warning")]), null);
  assert.equal(setupHealth([row("env", true, "missing"), fixable(row("gpu", false, "warning"))]), "missing");
  assert.equal(setupHealth(null), null);
});
