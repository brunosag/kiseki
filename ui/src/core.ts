import type { Schema, Value, Columns } from "./types.ts";

export function resolveSchema(schema: Schema): Schema {
  if (!schema.$ref) return schema;
  const key = schema.$ref.split("/").at(-1)!;
  if (!schema.$defs?.[key])
    throw new Error(`Unsupported schema reference: ${schema.$ref}`);
  return schema.$defs[key];
}
export function properties(schema: Schema) {
  return resolveSchema(schema).properties ?? {};
}
export function fieldSchema(schema: Schema): Schema {
  return schema.anyOf
    ? {
        ...schema,
        ...schema.anyOf.find((s) => s.type !== "null"),
        anyOf: undefined,
      }
    : schema;
}
export function defaultFields(schema: Schema): Record<string, string> {
  return Object.fromEntries(
    Object.entries(properties(schema)).map(([key, s]) => [
      key,
      s.default === null || s.default === undefined
        ? ""
        : Array.isArray(s.default)
          ? s.default.join(", ")
          : String(s.default),
    ]),
  );
}
function parseScalar(text: string, original: Schema): Value {
  const s = fieldSchema(original);
  if (!text.trim() && original.anyOf?.some((s) => s.type === "null"))
    return null;
  if (s.type === "integer" || s.type === "number") {
    if (!text.trim()) throw new Error("Enter a number");
    const n = Number(text);
    if (!Number.isFinite(n) || (s.type === "integer" && !Number.isInteger(n)))
      throw new Error(
        `Enter ${s.type === "integer" ? "an integer" : "a finite number"}`,
      );
    if (s.minimum !== undefined && n < s.minimum)
      throw new Error(`Must be ≥ ${s.minimum}`);
    if (s.maximum !== undefined && n > s.maximum)
      throw new Error(`Must be ≤ ${s.maximum}`);
    if (s.exclusiveMinimum !== undefined && n <= s.exclusiveMinimum)
      throw new Error(`Must be > ${s.exclusiveMinimum}`);
    if (s.exclusiveMaximum !== undefined && n >= s.exclusiveMaximum)
      throw new Error(`Must be < ${s.exclusiveMaximum}`);
    if (
      s.multipleOf &&
      Math.abs(n / s.multipleOf - Math.round(n / s.multipleOf)) > 1e-8
    )
      throw new Error(`Must be a multiple of ${s.multipleOf}`);
    return n;
  }
  if (s.type === "boolean") {
    if (text !== "true" && text !== "false")
      throw new Error("Choose true or false");
    return text === "true";
  }
  if (s.enum && !s.enum.includes(text))
    throw new Error("Choose one of the available values");
  return text;
}
export function parseField(
  text: string,
  schema: Schema,
  grid = false,
): Value[] {
  const s = fieldSchema(schema);
  if (s.type === "array")
    return [
      text.trim()
        ? text.split(",").map((x) => parseScalar(x.trim(), s.items ?? {}))
        : [],
    ];
  return (
    grid && s.type !== "string" && !s.enum && s.type !== "boolean"
      ? text.split(",")
      : [text]
  ).map((x) => parseScalar(x.trim(), schema));
}
export function runRequest(
  fields: Record<string, string>,
  schemas: Record<string, Schema>,
) {
  const overrides: Record<string, Value> = {},
    grid: Record<string, Value[]> = {};
  let count = 1;
  for (const [key, s] of Object.entries(schemas)) {
    let values: Value[];
    try {
      values = parseField(fields[key] ?? "", s, true);
    } catch (error) {
      throw new Error(`${key}: ${(error as Error).message}`);
    }
    if (values.length > 1) {
      grid[key] = values;
      count *= values.length;
    } else overrides[key] = values[0];
  }
  if (count > 256)
    throw new Error(
      "A grid can contain at most 256 runs. Narrow the comma lists.",
    );
  return { overrides, grid, count };
}
export function firstReached(
  columns: Columns,
  accuracy: number,
): { step: number; elapsed: number | undefined } | null {
  const index =
    columns.val_accuracy?.findIndex(
      (value) => Number.isFinite(value) && value >= accuracy,
    ) ?? -1;
  return index < 0
    ? null
    : { step: columns.step[index], elapsed: columns.elapsed?.[index] };
}
// A list with `incoming` in place of same-id items and the rest appended: an SSE refresh
// can list runs before the POST that created them returns.
export function mergeById<T extends { id: string }>(
  list: T[],
  incoming: T[],
): T[] {
  const byId = new Map(incoming.map((item) => [item.id, item]));
  const merged = list.map((item) => byId.get(item.id) ?? item);
  const listed = new Set(list.map((item) => item.id));
  return [...merged, ...incoming.filter((item) => !listed.has(item.id))];
}
export const percent = (n?: number | null) =>
  n == null ? "—" : `${(n * 100).toFixed(2)}%`;
export const number = (n?: number | null) =>
  n == null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 3 });
export function duration(n?: number | null): string {
  if (n == null) return "—";
  if (n < 60) return `${n.toFixed(1)}s`;
  if (n < 3600) return `${Math.floor(n / 60)}m ${Math.floor(n % 60)}s`;
  return `${Math.floor(n / 3600)}h ${Math.floor((n % 3600) / 60)}m`;
}
export const runLabel = (r: {
  id: string;
  spec: { label: string; optimizer: string };
}) => r.spec.label || `${r.spec.optimizer} · ${r.id.slice(-4)}`;
export const colors = [
  "#b8cf83",
  "#8db8df",
  "#dfaa79",
  "#b29fdb",
  "#df8d9c",
  "#72bdb4",
  "#ceb95d",
  "#95a8bd",
  "#d08ed0",
  "#a9c8ba",
];
