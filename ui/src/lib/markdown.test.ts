import assert from "node:assert/strict";
import { test } from "node:test";
import { blocks, inline } from "./markdown.ts";

test("paragraphs and pipe tables, as the analyses write them", () => {
  const text = [
    "**noise** (levels 0, 0.05): accuracy %",
    "",
    "| model | 0 | 0.05 |",
    "|---|---|---|",
    "| sgd 43ac | 98.6 | 98.7 |",
    "| leea 9815 | 84.9 | 84.9 |",
    "",
    "ECE: mean |accuracy − confidence|",
    "over bins.",
  ].join("\n");
  assert.deepEqual(blocks(text), [
    { kind: "paragraph", text: "**noise** (levels 0, 0.05): accuracy %" },
    {
      kind: "table",
      head: ["model", "0", "0.05"],
      rows: [
        ["sgd 43ac", "98.6", "98.7"],
        ["leea 9815", "84.9", "84.9"],
      ],
    },
    { kind: "paragraph", text: "ECE: mean |accuracy − confidence| over bins." },
  ]);
});

test("bold and code spans inside text", () => {
  assert.deepEqual(inline("**a vs b**: agree on `96%` of images"), [
    { bold: "a vs b" },
    ": agree on ",
    { code: "96%" },
    " of images",
  ]);
});
