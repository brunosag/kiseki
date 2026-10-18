// The tray menu's live part, which the page hands to the Electron shell: what trains and
// builds now, then the latest settled runs. The shell adds Open, New run and Quit around
// it. Labels change only at whole percents, so a training run rebuilds the menu rarely.
import { isActive, percent, progress, runName } from "./format.ts";
import { reportProgress, reportTitle } from "./reports.ts";
import { href } from "./router.ts";
import type { Options, Report, Run } from "./types.ts";

export type TrayAction = { open: string } | { newRun: true };
export type TrayItem = { label: string; action?: TrayAction; submenu?: TrayItem[] };

const RECENT = 5;

function trainingLabel(run: Run, options: Options | null): string {
  const { best_accuracy, machine } = run.status;
  const parts = [runName(options, run), `${Math.floor(progress(run) * 100)}%`];
  if (best_accuracy != null) parts.push(percent(best_accuracy));
  if (machine) parts.push(machine);
  return parts.join(" · ");
}

function settledLabel(run: Run, options: Options | null): string {
  const { state, test_accuracy, best_accuracy } = run.status;
  const accuracy = test_accuracy ?? best_accuracy;
  const outcome = state === "completed" && accuracy != null ? percent(accuracy) : state;
  return `${runName(options, run)} · ${outcome}`;
}

function settledAt(run: Run): string {
  return run.status.finished_at ?? run.status.updated_at ?? "";
}

export function trayItems(runs: Run[] | null, reports: Report[] | null, options: Options | null): TrayItem[] {
  if (!runs) return [];
  const items: TrayItem[] = runs
    .filter((run) => run.status.state === "running")
    .map((run) => ({ label: trainingLabel(run, options), action: { open: href.run(run.id) } }));
  const queued = runs.filter((run) => run.status.state === "queued").length;
  if (queued) items.push({ label: `${queued} queued`, action: { open: "/" } });
  for (const report of reports ?? []) {
    if (!isActive(report.status.state)) continue;
    const [done, total] = reportProgress(report);
    items.push({
      label: `Report ${reportTitle(report, runs, options)} · ${done}/${total}`,
      action: { open: href.report(report.id) },
    });
  }
  if (!items.length) items.push({ label: "Nothing training" });
  const recent = runs
    .filter((run) => !isActive(run.status.state))
    .sort((a, b) => (settledAt(a) < settledAt(b) ? 1 : settledAt(a) > settledAt(b) ? -1 : 0))
    .slice(0, RECENT)
    .map((run) => ({ label: settledLabel(run, options), action: { open: href.run(run.id) } }));
  if (recent.length) items.push({ label: "Recent runs", submenu: recent });
  return items;
}
