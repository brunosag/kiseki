// What this machine needs to run kiseki, each with its state and the action that fixes it
// (electron/requirements.cjs). In the app at /setup; before the Python environment exists,
// the shell shows it alone (setup.html) with Open kiseki once the required rows are ready.
import { RefreshCwIcon } from "lucide-react";
import { useState } from "react";
import { PageBody, PageHeader } from "@/components/page";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Spinner } from "@/components/ui/spinner";
import { canOpen, check, type Health, rowHealth, run, useSetup } from "@/lib/setup";
import { type Requirement, shell } from "@/lib/shell";
import { cn } from "@/lib/utils";

const DOT: Record<Health, string> = {
  ok: "bg-emerald-500",
  warning: "bg-amber-500",
  missing: "bg-red-500",
  off: "bg-muted-foreground/50",
};

export function SetupView({ standalone = false }: { standalone?: boolean }) {
  const { rows, checking } = useSetup();
  const [opening, setOpening] = useState(false);
  const open = () => {
    setOpening(true);
    shell?.openApp().catch(() => setOpening(false));
  };
  return (
    <>
      <PageHeader
        actions={
          <Button variant="ghost" size="sm" className="text-[13px]" onClick={() => check(true)} disabled={checking}>
            {checking ? <Spinner /> : <RefreshCwIcon />}
            Re-check
          </Button>
        }
      >
        <span className="font-medium">Setup</span>
      </PageHeader>
      <PageBody>
        {!shell ? (
          <Empty className="h-full">
            <EmptyHeader>
              <EmptyTitle className="font-medium text-base">Only in the app</EmptyTitle>
              <EmptyDescription>Setup checks this machine from the desktop app.</EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : !rows ? (
          <Empty className="h-full">
            <Spinner />
          </Empty>
        ) : (
          <div className="mx-auto max-w-4xl py-6">
            {standalone && (
              <p className="px-4 pb-4 text-[13px] text-muted-foreground">
                kiseki needs a Python environment on this machine before it opens. The rest is optional.
              </p>
            )}
            <ul aria-label="Requirements" className="divide-y rounded-xl border">
              {rows.map((row) => (
                <RequirementRow key={row.id} row={row} />
              ))}
            </ul>
            {standalone && (
              <div className="flex items-center justify-end gap-3 px-1 pt-4">
                <Button onClick={open} disabled={!canOpen(rows) || opening}>
                  {opening && <Spinner />}
                  Open kiseki
                </Button>
              </div>
            )}
          </div>
        )}
      </PageBody>
    </>
  );
}

function RequirementRow({ row }: { row: Requirement }) {
  const job = row.job;
  const running = job?.state === "running";
  const failed = job?.state === "failed";
  const label = failed && row.action?.kind === "install" ? "Retry" : row.action?.label;
  return (
    <li className="px-4 py-3 text-[13px]">
      <div className="flex min-h-8 items-center gap-3">
        <span className={cn("size-2 shrink-0 rounded-full", DOT[rowHealth(row)])} title={row.state} />
        <span className="w-48 shrink-0 whitespace-nowrap font-medium">
          {row.title}
          {!row.required && <span className="ms-1.5 font-normal text-muted-foreground text-xs">optional</span>}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-muted-foreground" title={row.detail}>
            {row.detail}
          </span>
          {running && job.lines.length > 0 && (
            <span className="block truncate font-mono text-muted-foreground text-xs" title={job.lines.at(-1)}>
              {job.lines.at(-1)}
            </span>
          )}
          {failed && <span className="block truncate text-destructive-foreground text-xs">{job.error}</span>}
        </span>
        {row.version && <span className="w-44 shrink-0 truncate text-right text-muted-foreground text-xs">{row.version}</span>}
        <span className="flex w-40 shrink-0 justify-end">
          {label && (
            <Button
              variant={row.state === "ok" ? "ghost" : "outline"}
              size="sm"
              className="text-[13px]"
              disabled={running}
              onClick={() => run(row.id)}
            >
              {running && <Spinner />}
              {label}
            </Button>
          )}
        </span>
      </div>
      {job && job.lines.length > 0 && (
        <details className="ms-5 mt-1">
          <summary className="cursor-default text-muted-foreground text-xs">Output</summary>
          <pre className="mt-1 max-h-60 overflow-y-auto whitespace-pre-wrap rounded-lg bg-muted p-2 font-mono text-xs">
            {job.lines.join("\n")}
          </pre>
        </details>
      )}
    </li>
  );
}
