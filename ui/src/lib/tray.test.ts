import assert from "node:assert/strict";
import { test } from "node:test";
import { trayItems } from "./tray.ts";
import type { Options, Report, Run, Status } from "./types.ts";

const options = {
  optimizers: { sgd: { title: "SGD" }, leea: { title: "LEEA" } },
  analyses: {},
} as unknown as Options;

function run(id: string, optimizer: string, status: Status): Run {
  return { id, spec: { optimizer, iterations: 1000 } as Run["spec"], status };
}

const training = run("20261018-101500-mnist-sgd-43ac", "sgd", {
  state: "running",
  step: 425,
  iterations: 1000,
  best_accuracy: 0.9134,
  machine: "laptop",
});

test("training runs lead, each opening its run, then the queue", () => {
  const queued = [
    run("20261018-101600-mnist-leea-aa01", "leea", { state: "queued" }),
    run("20261018-101700-mnist-leea-aa02", "leea", { state: "queued" }),
  ];
  assert.deepEqual(trayItems([...queued, training], [], options), [
    { label: "SGD 43ac · 42% · 91.3% · laptop", action: { open: "/runs/20261018-101500-mnist-sgd-43ac" } },
    { label: "2 queued", action: { open: "/" } },
  ]);
});

test("a run without an accuracy yet shows its progress alone", () => {
  const starting = run("20261018-101500-mnist-leea-b7f0", "leea", { state: "running", step: 0, machine: "lugia" });
  assert.deepEqual(trayItems([starting], [], options), [
    { label: "LEEA b7f0 · 0% · lugia", action: { open: "/runs/20261018-101500-mnist-leea-b7f0" } },
  ]);
});

test("reports being built show the analyses done", () => {
  const report: Report = {
    id: "20261018-110000-report-9c1d",
    spec: { models: ["a-sgd-43ac:best"], labels: ["baseline"], split: "test", analyses: { cka: {}, loss: {} }, seed: 0 },
    status: { state: "running", seconds: { cka: 3.2 } },
  };
  assert.deepEqual(trayItems([], [report], options), [
    { label: "Report baseline · 1/2", action: { open: "/reports/20261018-110000-report-9c1d" } },
  ]);
});

test("when nothing trains, the menu says so, and recent runs sit in a submenu", () => {
  const done = [
    run("20261018-090000-mnist-sgd-0001", "sgd", { state: "completed", test_accuracy: 0.9871, finished_at: "2026-10-18T09:10:00Z" }),
    run("20261018-091000-mnist-leea-0002", "leea", { state: "failed", finished_at: "2026-10-18T09:30:00Z" }),
    run("20261018-092000-mnist-sgd-0003", "sgd", { state: "stopped", updated_at: "2026-10-18T09:20:00Z" }),
    run("20261018-093000-mnist-sgd-0004", "sgd", { state: "paused", updated_at: "2026-10-18T09:40:00Z" }),
    run("20261018-094000-mnist-sgd-0005", "sgd", { state: "completed", best_accuracy: 0.95, finished_at: "2026-10-18T09:45:00Z" }),
    run("20261018-080000-mnist-sgd-0006", "sgd", { state: "completed", finished_at: "2026-10-18T08:10:00Z" }),
  ];
  assert.deepEqual(trayItems(done, [], options), [
    { label: "Nothing training" },
    {
      label: "Recent runs",
      submenu: [
        { label: "SGD 0005 · 95.0%", action: { open: "/runs/20261018-094000-mnist-sgd-0005" } },
        { label: "SGD 0004 · paused", action: { open: "/runs/20261018-093000-mnist-sgd-0004" } },
        { label: "LEEA 0002 · failed", action: { open: "/runs/20261018-091000-mnist-leea-0002" } },
        { label: "SGD 0003 · stopped", action: { open: "/runs/20261018-092000-mnist-sgd-0003" } },
        { label: "SGD 0001 · 98.7%", action: { open: "/runs/20261018-090000-mnist-sgd-0001" } },
      ],
    },
  ]);
});

test("before the lists load there is nothing to show", () => {
  assert.deepEqual(trayItems(null, null, null), []);
});
