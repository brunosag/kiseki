// Every report, newest first: what it compares, on which data, and how far it got.
import { PlusIcon } from "lucide-react";
import { PageBody, PageHeader } from "@/components/page";
import { StatusIcon } from "@/components/status-icon";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { ago, analysisTitle, datasetTitle, idTime, shortId } from "@/lib/format";
import { useLive } from "@/lib/live";
import { reportProgress, reportTitle, runOf } from "@/lib/reports";
import { href, navigate } from "@/lib/router";
import type { Options, Report, Run } from "@/lib/types";

export function ReportsView({ onNewReport }: { onNewReport: (refs?: string[]) => void }) {
  const { reports, runs, options } = useLive();
  return (
    <>
      <PageHeader
        actions={
          <Button variant="ghost" size="sm" className="text-[13px]" onClick={() => onNewReport()}>
            <PlusIcon />
            New report
          </Button>
        }
      >
        <span className="font-medium">Reports</span>
      </PageHeader>
      <PageBody>
        {reports && !reports.length ? (
          <Empty className="h-full">
            <EmptyHeader>
              <EmptyTitle className="font-medium text-base">No reports yet</EmptyTitle>
              <EmptyDescription>
                A report runs analyses on trained models side by side: accuracy, calibration, weights,
                representations and more.
              </EmptyDescription>
            </EmptyHeader>
            <Button size="sm" onClick={() => onNewReport()}>
              New report
            </Button>
          </Empty>
        ) : (
          <div role="list" aria-label="Reports" className="py-1">
            {reports?.map((report) => (
              <ReportRow key={report.id} report={report} runs={runs} options={options} />
            ))}
          </div>
        )}
      </PageBody>
    </>
  );
}

function ReportRow({ report, runs, options }: { report: Report; runs: Run[] | null; options: Options | null }) {
  const { status, spec } = report;
  const [done, total] = reportProgress(report);
  const dataset = runOf(spec.models[0] ?? "", runs)?.spec.dataset;
  const created = idTime(report.id);
  const open = () => navigate(href.report(report.id));
  return (
    <div
      role="listitem"
      tabIndex={0}
      onClick={open}
      onKeyDown={(event) => event.key === "Enter" && open()}
      className="flex h-10 cursor-default items-center gap-3 px-4 text-[13px] outline-none hover:bg-accent focus-visible:bg-accent"
    >
      <StatusIcon state={status.state} progress={total ? done / total : 0} />
      <span className="w-10 shrink-0 font-mono text-muted-foreground text-xs">{shortId(report.id)}</span>
      <span className="flex min-w-0 flex-1 items-center gap-2">
        <span className="shrink-0 font-medium">{reportTitle(report, runs, options)}</span>
        {dataset && <span className="shrink-0 text-muted-foreground">{datasetTitle(options, dataset)}</span>}
        {spec.split === "val" && (
          <Badge variant="outline" className="shrink-0 font-normal text-muted-foreground">
            validation split
          </Badge>
        )}
        {status.state === "failed" && status.error ? (
          <span className="truncate text-destructive-foreground" title={status.error}>
            {status.error}
          </span>
        ) : status.state === "running" ? (
          <span className="truncate text-muted-foreground">
            {status.stage ? `${analysisTitle(options, status.stage)} · ` : ""}
            {done} of {total}
          </span>
        ) : null}
      </span>
      <span className="shrink-0 text-muted-foreground text-xs tabular-nums">
        {total} {total === 1 ? "analysis" : "analyses"}
      </span>
      <span className="w-12 shrink-0 text-right text-muted-foreground text-xs" title={created ?? undefined}>
        {ago(created)}
      </span>
    </div>
  );
}
