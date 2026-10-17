import test from "node:test";
import assert from "node:assert/strict";
import { firstReached, parseField, properties, runRequest } from "./core.ts";
import { decodeNpy, float16 } from "./npy.ts";

test("schema references, nullable fields, strict bounds and even populations", () => {
  const schema = {
    $ref: "#/$defs/Params",
    $defs: {
      Params: {
        properties: {
          population_size: { type: "integer", minimum: 2, multipleOf: 2 },
        },
      },
    },
  };
  assert.deepEqual(properties(schema), {
    population_size: { type: "integer", minimum: 2, multipleOf: 2 },
  });
  assert.deepEqual(
    parseField("", {
      anyOf: [{ type: "integer", minimum: 1 }, { type: "null" }],
    }),
    [null],
  );
  assert.throws(
    () => parseField("3", properties(schema).population_size),
    /multiple of 2/,
  );
  assert.throws(
    () => parseField("0", { type: "number", exclusiveMinimum: 0 }),
    /> 0/,
  );
  assert.throws(
    () => parseField("1", { type: "number", exclusiveMaximum: 1 }),
    /< 1/,
  );
  assert.throws(() => parseField("1.5", { type: "integer" }), /integer/);
});

test("numeric comma lists create independent grid axes; array params remain one array", () => {
  assert.deepEqual(
    runRequest(
      { seed: "42, 43", iterations: "20", "params.sigma": "0.01, 0.02" },
      {
        seed: { type: "integer" },
        iterations: { type: "integer" },
        "params.sigma": { type: "number", exclusiveMinimum: 0 },
      },
    ),
    {
      overrides: { iterations: 20 },
      grid: { seed: [42, 43], "params.sigma": [0.01, 0.02] },
      count: 4,
    },
  );
  assert.deepEqual(
    parseField("0, .1, .5", { type: "array", items: { type: "number" } }, true),
    [[0, 0.1, 0.5]],
  );
  assert.deepEqual(parseField("baseline, repeat", { type: "string" }, true), [
    "baseline, repeat",
  ]);
  assert.throws(() => parseField("NaN", { type: "number" }), /finite/);
});

test("accuracy threshold reports the first observed sample, including a regression", () => {
  const columns = {
    step: [0, 5, 10, 15],
    elapsed: [0.2, 2, 5, 7],
    val_accuracy: [0.1, 0.95, 0.8, 0.99],
  };
  assert.deepEqual(firstReached(columns, 0.9), { step: 5, elapsed: 2 });
  assert.deepEqual(firstReached(columns, 0.98), { step: 15, elapsed: 7 });
  assert.equal(firstReached(columns, 1), null);
  assert.equal(firstReached({}, 0.5), null);
});

function npy(
  descr: string,
  payload: number[],
  shape = "2,",
  fortran = false,
): ArrayBuffer {
  const header = Buffer.from(
    `{'descr': '${descr}', 'fortran_order': ${fortran ? "True" : "False"}, 'shape': (${shape}), }\n`,
  );
  const prefix = Buffer.from([
    0x93,
    78,
    85,
    77,
    80,
    89,
    1,
    0,
    header.length & 255,
    header.length >> 8,
  ]);
  return Uint8Array.from(Buffer.concat([prefix, header, Buffer.from(payload)]))
    .buffer;
}
test("NumPy reads real report dtypes without assuming float32 or aligned payloads", () => {
  const half = decodeNpy(npy("<f2", [0, 60, 0, 184]));
  assert.deepEqual(half.shape, [2]);
  assert.deepEqual([...half.data], [1, -0.5]);
  assert.deepEqual(
    [...decodeNpy(npy("<f4", [0, 0, 128, 63, 0, 0, 32, 192])).data],
    [1, -2.5],
  );
  assert.deepEqual(
    [
      ...decodeNpy(
        npy("<i8", [42, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 0, 0]),
      ).data,
    ],
    [42, 127],
  );
  assert.deepEqual(
    [...decodeNpy(npy(">f4", [63, 128, 0, 0, 192, 32, 0, 0])).data],
    [1, -2.5],
  );
  assert.equal(float16(0x0001), 2 ** -24);
  assert.equal(float16(0x7c00), Infinity);
});
test("NumPy rejects malformed and unsupported arrays", () => {
  assert.throws(() => decodeNpy(new ArrayBuffer(12)), /Invalid NumPy/);
  assert.throws(() => decodeNpy(npy("<f4", [0, 0, 0, 0])), /size/);
  assert.throws(
    () => decodeNpy(npy("<f2", [0, 60, 0, 184], "2,", true)),
    /Fortran/,
  );
});
