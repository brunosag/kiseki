import assert from "node:assert/strict";
import { test } from "node:test";
import { fields, parse, request, runCount, toField } from "./schema.ts";
import type { Schema } from "./types.ts";

const run: Schema = {
  $ref: "#/$defs/RunSpec",
  $defs: {
    RunSpec: {
      properties: {
        seed: { type: "integer", default: 42 },
        iterations: { type: "integer", minimum: 1, default: 100000, description: "Generations, or SGD steps" },
        target_accuracy: { anyOf: [{ type: "number", maximum: 1 }, { type: "null" }], default: null },
        precision: { enum: ["bf16", "fp32"], default: "fp32" },
        label: { type: "string", default: "" },
      },
    },
  },
};

test("fields follow the schema, nullable numbers included", () => {
  const [seed, iterations, target, precision] = fields(run);
  assert.equal(seed.kind, "integer");
  assert.equal(iterations.min, 1);
  assert.equal(iterations.description, "Generations, or SGD steps");
  assert.equal(target.kind, "number");
  assert.equal(target.nullable, true);
  assert.deepEqual(precision.choices, ["bf16", "fp32"]);
});

test("commas make a grid, except in list fields", () => {
  const rate = toField("learning_rate", { type: "number", default: 0.01 });
  assert.deepEqual(parse("0.01, 0.05", rate), [0.01, 0.05]);
  const noise = toField("noise", { type: "array", items: { type: "number" }, default: [0, 0.1] });
  assert.deepEqual(parse("0, 0.2, 0.5", noise), [[0, 0.2, 0.5]]);
  assert.throws(() => parse("1.5", toField("seed", { type: "integer" })), /whole number/);
});

test("a request carries only what changed; its grid sets the run count", () => {
  const known = fields(run);
  const sent = request({ seed: "1, 2, 3", iterations: "100000", precision: "bf16, fp32", label: "sweep" }, known);
  assert.deepEqual(sent.overrides, { label: "sweep" });
  assert.deepEqual(sent.grid, { seed: [1, 2, 3], precision: ["bf16", "fp32"] });
  assert.equal(runCount(sent.grid), 6);
  assert.deepEqual(request({ target_accuracy: "" }, known), { overrides: {}, grid: {} });
});
