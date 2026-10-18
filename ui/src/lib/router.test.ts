import assert from "node:assert/strict";
import { test } from "node:test";
import { href, parseRoute } from "./router.ts";

test("routes from paths", () => {
  assert.deepEqual(parseRoute("/"), { name: "runs" });
  assert.deepEqual(parseRoute("/runs/20261017-145539-mnist-sgd-43ac"), {
    name: "run",
    id: "20261017-145539-mnist-sgd-43ac",
  });
  assert.deepEqual(parseRoute("/compare", "?runs=a,b"), { name: "compare", ids: ["a", "b"] });
  assert.deepEqual(parseRoute("/reports/r1"), { name: "report", id: "r1" });
  assert.deepEqual(parseRoute("/machines"), { name: "machines" });
});

test("links parse back to their routes", () => {
  assert.deepEqual(parseRoute(...(href.compare(["a", "b"]).split("?") as [string, string])), {
    name: "compare",
    ids: ["a", "b"],
  });
});
