import assert from "node:assert/strict";
import { test } from "node:test";
import { ago, changedParams, duration, idTime, number, percent, shortId } from "./format.ts";
import type { Options, Run } from "./types.ts";

test("percentages and missing values", () => {
  assert.equal(percent(0.98629), "98.6%");
  assert.equal(percent(0.5, 2), "50.00%");
  assert.equal(percent(null), "—");
});

test("durations read at the scale they have", () => {
  assert.equal(duration(7.288), "7.3s");
  assert.equal(duration(140.8), "2m 21s");
  assert.equal(duration(3 * 3600 + 5 * 60), "3h 05m");
  assert.equal(duration(50 * 3600), "2d 2h");
});

test("relative times turn into dates after a week", () => {
  const now = Date.parse("2026-10-18T12:00:00Z");
  assert.equal(ago("2026-10-18T11:59:30Z", now), "now");
  assert.equal(ago("2026-10-18T11:55:00Z", now), "5m");
  assert.equal(ago("2026-10-18T09:00:00Z", now), "3h");
  assert.equal(ago("2026-10-16T12:00:00Z", now), "2d");
  assert.equal(ago("2026-10-01T12:00:00Z", now), "Oct 1");
});

test("numbers as people write them", () => {
  assert.equal(number(0.05), "0.05");
  assert.equal(number(1000), "1000");
  assert.equal(number(0.0001), "1e-4");
  assert.equal(number(0.123456), "0.1235");
});

test("a run goes by its id's suffix", () => {
  assert.equal(shortId("20261017-145539-mnist-sgd-43ac"), "43ac");
});

test("only params away from the optimizer's defaults set a run apart", () => {
  const options = {
    optimizers: {
      sgd: {
        title: "SGD",
        summary: "",
        params: {
          $ref: "#/$defs/Params",
          $defs: { Params: { properties: { learning_rate: { default: 0.01 }, momentum: { default: 0 } } } },
        },
      },
    },
  } as unknown as Options;
  const run = { id: "x", spec: { optimizer: "sgd", params: { learning_rate: 0.05, momentum: 0 } } } as unknown as Run;
  assert.deepEqual(changedParams(run, options), [["learning_rate", 0.05]]);
});

test("an id's leading timestamp is when it was made, in UTC", () => {
  assert.equal(idTime("20261017-155537-8298"), "2026-10-17T15:55:37Z");
  assert.equal(idTime("20261017-145539-mnist-sgd-43ac"), "2026-10-17T14:55:39Z");
  assert.equal(idTime("report"), null);
});
