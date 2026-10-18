import assert from "node:assert/strict";
import { test } from "node:test";
import { accuracyRange, appendColumns, crossing } from "./curves.ts";

test("appended rows pad columns that only one side has", () => {
  const columns = appendColumns({ step: [1, 2], loss: [3, 2] }, { step: [3], grad_norm: [0.5] });
  assert.deepEqual(columns, { step: [1, 2, 3], loss: [3, 2, null], grad_norm: [null, null, 0.5] });
});

test("the first validation at the threshold, with its time", () => {
  const val = { step: [10, 20, 30], elapsed: [4.6, 9.2, 13.8], val_accuracy: [0.5, 0.91, 0.95] };
  assert.deepEqual(crossing(val, 0.9), { step: 20, elapsed: 9.2 });
  assert.equal(crossing(val, 0.99), null);
});

test("an accuracy axis without data spans 0–100%", () => {
  assert.deepEqual(accuracyRange(null, null), [0, 1]);
  assert.deepEqual(accuracyRange(Infinity, -Infinity), [0, 1]);
  const [low, high] = accuracyRange(0.9, 0.95);
  assert.equal(Number(low.toFixed(3)), 0.895); // the 0.5 pp minimum pad beats 8% of 5 pp
  assert.equal(Number(high.toFixed(3)), 0.955);
});
