// Form fields from the catalog's JSON Schemas (msgspec's output), and what the forms
// send: a comma in a scalar field makes a grid, one run per combination.
import type { Json, Schema } from "./types";

export interface Field {
  key: string;
  kind: "number" | "integer" | "boolean" | "choice" | "text" | "list";
  nullable: boolean;
  default: Json;
  description: string;
  choices: Json[];
  min?: number;
  max?: number;
}

export function properties(schema: Schema): Record<string, Schema> {
  const name = schema.$ref?.split("/").pop();
  return (name ? schema.$defs?.[name] : schema)?.properties ?? {};
}

export function toField(key: string, property: Schema): Field {
  const options = property.anyOf ?? [property];
  const nullable = options.some((option) => option.type === "null");
  const value = options.find((option) => option.type !== "null") ?? property;
  const kind: Field["kind"] = value.enum
    ? "choice"
    : value.type === "integer"
      ? "integer"
      : value.type === "number"
        ? "number"
        : value.type === "boolean"
          ? "boolean"
          : value.type === "array"
            ? "list"
            : "text";
  return {
    key,
    kind,
    nullable,
    default: property.default ?? null,
    description: value.description ?? property.description ?? "",
    choices: value.enum ?? [],
    min: value.minimum ?? value.exclusiveMinimum,
    max: value.maximum ?? value.exclusiveMaximum,
  };
}

export function fields(schema: Schema): Field[] {
  return Object.entries(properties(schema)).map(([key, property]) => toField(key, property));
}

/** A field's default as the form shows it. */
export function shown(value: Json): string {
  if (value == null) return "";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

function scalar(text: string, field: Field): Json {
  if (text === "" && field.nullable) return null;
  if (field.kind === "integer" || field.kind === "number") {
    const value = Number(text);
    if (text === "" || !Number.isFinite(value)) throw new Error(`${field.key}: “${text}” is not a number`);
    if (field.kind === "integer" && !Number.isInteger(value))
      throw new Error(`${field.key}: “${text}” is not a whole number`);
    return value;
  }
  if (field.kind === "boolean") return text === "true";
  return text;
}

/** The values typed into a field: one value, or several separated by commas (a grid).
 * A list field's commas separate its items instead. */
export function parse(text: string, field: Field): Json[] {
  const trimmed = text.trim();
  if (field.kind === "list")
    return [trimmed ? trimmed.split(",").map((item) => scalar(item.trim(), { ...field, kind: "number" })) : []];
  if (field.kind === "text" || field.kind === "choice" || field.kind === "boolean") {
    const parts = field.kind === "text" ? [trimmed] : trimmed.split(",").map((part) => part.trim());
    return parts.map((part) => scalar(part, field));
  }
  return trimmed.split(",").map((part) => scalar(part.trim(), field));
}

export interface RunRequest {
  overrides: Record<string, Json>;
  grid: Record<string, Json[]>;
}

/** What the form sends for the fields the user changed from their defaults. */
export function request(values: Record<string, string>, known: Field[]): RunRequest {
  const out: RunRequest = { overrides: {}, grid: {} };
  for (const field of known) {
    const text = values[field.key];
    if (text === undefined || text.trim() === shown(field.default)) continue;
    const parsed = parse(text, field);
    if (parsed.length > 1) out.grid[field.key] = parsed;
    else out.overrides[field.key] = parsed[0];
  }
  return out;
}

export function runCount(grid: Record<string, Json[]>): number {
  return Object.values(grid).reduce((total, values) => total * values.length, 1);
}
