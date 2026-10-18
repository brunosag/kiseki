// Pure helpers for curve columns (see rows.ts for how they are fetched).

export type Columns = Record<string, (number | null)[]>;

/** Columns with `more` appended; a column missing on either side is padded with null. */
export function appendColumns(columns: Columns, more: Columns): Columns {
  const before = Object.values(columns)[0]?.length ?? 0;
  const added = Object.values(more)[0]?.length ?? 0;
  const out: Columns = {};
  for (const key of new Set([...Object.keys(columns), ...Object.keys(more)])) {
    out[key] = [...(columns[key] ?? Array(before).fill(null)), ...(more[key] ?? Array(added).fill(null))];
  }
  return out;
}

/** The first validation that reached `threshold`: its step and elapsed seconds. */
export function crossing(
  columns: Columns | null,
  threshold: number,
): { step: number; elapsed: number | null } | null {
  const accuracy = columns?.val_accuracy ?? [];
  const index = accuracy.findIndex((value) => value != null && value >= threshold);
  if (index < 0) return null;
  return { step: columns?.step?.[index] ?? 0, elapsed: columns?.elapsed?.[index] ?? null };
}

/** An accuracy axis fitted to the data with a little air, within 0–1; 0–1 when empty. */
export function accuracyRange(min: number | null, max: number | null): [number, number] {
  if (min == null || max == null || !Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  const pad = Math.max((max - min) * 0.08, 0.005);
  return [Math.max(0, min - pad), Math.min(1, max + pad)];
}
