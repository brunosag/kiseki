import assert from "node:assert/strict";
import { test } from "node:test";
import { refParts, reportTitle } from "./reports.ts";
import type { Options, Report, Run } from "./types.ts";

const options = {
  optimizers: { sgd: { title: "SGD" }, leea: { title: "LEEA" } },
} as unknown as Options;
const run = (id: string, optimizer: string, label = "") => ({ id, spec: { optimizer, label } }) as unknown as Run;
const runs = [
  run("20261017-145539-mnist-sgd-43ac", "sgd"),
  run("20261017-145556-mnist-leea-9815", "leea"),
  run("20261017-150000-mnist-sgd-77aa", "sgd"),
  run("20261017-160000-mnist-sgd-b0b0", "sgd", "smoke"),
];
const report = (models: string[], labels: string[] = []) => ({ id: "r", spec: { models, labels } }) as unknown as Report;

test("a report is named after the optimizers it compares", () => {
  assert.equal(
    reportTitle(report(["20261017-145539-mnist-sgd-43ac:best", "20261017-145556-mnist-leea-9815:best"]), runs, options),
    "SGD vs LEEA",
  );
});

test("models of one optimizer go by their runs, and labels win", () => {
  assert.equal(
    reportTitle(report(["20261017-145539-mnist-sgd-43ac:best", "20261017-150000-mnist-sgd-77aa:final"]), runs, options),
    "SGD 43ac vs SGD 77aa final",
  );
  assert.equal(reportTitle(report(["20261017-145539-mnist-sgd-43ac:best"], ["baseline"]), runs, options), "baseline");
});

test("a run's own label names its model, as in the report itself", () => {
  assert.equal(
    reportTitle(report(["20261017-160000-mnist-sgd-b0b0:best", "20261017-150000-mnist-sgd-77aa:best"]), runs, options),
    "smoke b0b0 vs SGD 77aa",
  );
});

test("a model reference is a run id and a model name", () => {
  assert.deepEqual(refParts("20261017-145539-mnist-sgd-43ac:step-10"), { run: "20261017-145539-mnist-sgd-43ac", model: "step-10" });
  assert.deepEqual(refParts("20261017-145539-mnist-sgd-43ac"), { run: "20261017-145539-mnist-sgd-43ac", model: "best" });
});
