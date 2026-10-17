import { fieldSchema } from "../core";
import type { Schema } from "../types";
export function SchemaFields({
  schemas,
  values,
  onChange,
  grid = false,
}: {
  schemas: Record<string, Schema>;
  values: Record<string, string>;
  onChange: (key: string, value: string) => void;
  grid?: boolean;
}) {
  return (
    <div className="fields">
      {Object.entries(schemas).map(([name, original]) => {
        const s = fieldSchema(original),
          value = values[name] ?? "";
        const hints = [
          s.minimum !== undefined ? `≥ ${s.minimum}` : "",
          s.exclusiveMinimum !== undefined ? `> ${s.exclusiveMinimum}` : "",
          s.maximum !== undefined ? `≤ ${s.maximum}` : "",
          s.exclusiveMaximum !== undefined ? `< ${s.exclusiveMaximum}` : "",
          s.multipleOf ? `multiple of ${s.multipleOf}` : "",
        ]
          .filter(Boolean)
          .join(" · ");
        return (
          <label key={name} className="field">
            <span>{name.replaceAll("_", " ")}</span>
            {s.enum ? (
              <select
                value={value}
                onChange={(e) => onChange(name, e.target.value)}
              >
                {s.enum.map((v) => (
                  <option key={String(v)} value={String(v)}>
                    {String(v)}
                  </option>
                ))}
              </select>
            ) : s.type === "boolean" ? (
              <select
                value={value}
                onChange={(e) => onChange(name, e.target.value)}
              >
                <option value="false">False</option>
                <option value="true">True</option>
              </select>
            ) : (
              <input
                value={value}
                onChange={(e) => onChange(name, e.target.value)}
                placeholder={
                  original.anyOf?.some((s) => s.type === "null")
                    ? "Auto (null)"
                    : ""
                }
                inputMode={
                  s.type === "number" ||
                  s.type === "integer" ||
                  s.type === "array"
                    ? "decimal"
                    : "text"
                }
                title={
                  grid
                    ? "Comma-separated numeric values create a grid"
                    : undefined
                }
              />
            )}
            <small>
              {s.description || original.description}
              {hints && <em>{hints}</em>}
            </small>
          </label>
        );
      })}
    </div>
  );
}
