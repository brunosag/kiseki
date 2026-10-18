// Reports in lists: the saved models they compare, what they're called, how far they got.
import { optimizerTitle, shortId } from "./format.ts";
import type { Options, Report, Run } from "./types.ts";

/** A saved model's reference, `<run id>:<model>` (the server's ModelRef). */
export function modelRef(run: string, model = "best"): string {
  return `${run}:${model}`;
}

export function refParts(ref: string): { run: string; model: string } {
  const colon = ref.indexOf(":");
  return colon < 0 ? { run: ref, model: "best" } : { run: ref.slice(0, colon), model: ref.slice(colon + 1) || "best" };
}

export function runOf(ref: string, runs: Run[] | null): Run | undefined {
  const { run } = refParts(ref);
  return runs?.find((candidate) => candidate.id === run);
}

/** Each model's name as the worker writes it in the report: the report's label, else the
 * run's label or optimizer, its short id, and the model when not the best. */
function subjectNames(report: Report, runs: Run[] | null, options: Options | null): string[] {
  return report.spec.models.map((ref, index) => {
    if (report.spec.labels[index]) return report.spec.labels[index];
    const { run: id, model } = refParts(ref);
    const run = runOf(ref, runs);
    const name = `${run ? run.spec.label || optimizerTitle(options, run.spec.optimizer) : "Run"} ${shortId(id)}`;
    return model === "best" ? name : `${name} ${model}`;
  });
}

/** "SGD vs LEEA" when each model has its own optimizer, else the models' names. */
export function reportTitle(report: Report, runs: Run[] | null, options: Options | null): string {
  const names = subjectNames(report, runs, options);
  const optimizers = report.spec.models.map((ref) => runOf(ref, runs)?.spec.optimizer);
  const distinct = !report.spec.labels.length && optimizers.every(Boolean) && new Set(optimizers).size === names.length;
  if (distinct && report.spec.models.every((ref) => refParts(ref).model === "best"))
    return optimizers.map((key) => optimizerTitle(options, key!)).join(" vs ");
  return names.length > 3 ? `${names.length} models` : names.join(" vs ");
}

/** Analyses done out of those asked for. */
export function reportProgress(report: Report): [number, number] {
  return [Object.keys(report.status.seconds ?? {}).length, Object.keys(report.spec.analyses).length];
}
