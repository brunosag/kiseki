// Ctrl+K: every action and place in one searchable list, runs and reports included.
import { FileChartColumnIcon, FlaskConicalIcon, PlusIcon, RefreshCwIcon, ServerIcon } from "lucide-react";
import { Fragment } from "react";
import { StatusIcon } from "@/components/status-icon";
import {
  Command,
  CommandCollection,
  CommandDialog,
  CommandDialogPopup,
  CommandEmpty,
  CommandGroup,
  CommandGroupLabel,
  CommandInput,
  CommandItem,
  CommandList,
  CommandPanel,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command";
import { datasetTitle, progress, runName, shortId } from "@/lib/format";
import { probeMachines, useLive } from "@/lib/live";
import { reportTitle } from "@/lib/reports";
import { href, navigate } from "@/lib/router";

interface Entry {
  value: string;
  label: string;
  search: string;
  icon: React.ReactNode;
  hint?: string;
  shortcut?: string;
  act: () => void;
}

export function CommandMenu({
  open,
  onOpenChange,
  onNewRun,
  onNewReport,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNewRun: () => void;
  onNewReport: () => void;
}) {
  const { runs, reports, options } = useLive();
  const entry = (value: string, label: string, icon: React.ReactNode, act: () => void, more: Partial<Entry> = {}): Entry => ({
    value,
    label,
    search: label,
    icon,
    act,
    ...more,
  });
  const groups = [
    {
      value: "Actions",
      items: [
        entry("new-run", "New run", <PlusIcon />, onNewRun, { shortcut: "C" }),
        entry("new-report", "New report", <PlusIcon />, onNewReport),
        entry("probe", "Refresh machines", <RefreshCwIcon />, () => probeMachines().catch(() => {})),
      ],
    },
    {
      value: "Go to",
      items: [
        entry("go-runs", "Runs", <FlaskConicalIcon />, () => navigate("/runs")),
        entry("go-reports", "Reports", <FileChartColumnIcon />, () => navigate("/reports")),
        entry("go-machines", "Machines", <ServerIcon />, () => navigate("/machines")),
      ],
    },
    {
      value: "Runs",
      items: (runs ?? []).map((run) => {
        const label = runName(options, run);
        return entry(`run:${run.id}`, label, <StatusIcon state={run.status.state} progress={progress(run)} />, () => navigate(href.run(run.id)), {
          hint: [datasetTitle(options, run.spec.dataset), run.spec.label].filter(Boolean).join(" · "),
          search: `${label} ${run.id} ${run.spec.label} ${datasetTitle(options, run.spec.dataset)} ${run.status.state}`,
        });
      }),
    },
    {
      value: "Reports",
      items: (reports ?? []).map((report) => {
        const label = reportTitle(report, runs, options);
        return entry(`report:${report.id}`, label, <StatusIcon state={report.status.state} />, () => navigate(href.report(report.id)), {
          hint: shortId(report.id),
          search: `${label} ${report.id} report`,
        });
      }),
    },
  ].filter((group) => group.items.length);

  const run = (item: Entry) => {
    onOpenChange(false);
    item.act();
  };

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandDialogPopup>
        <Command items={groups} itemToStringValue={(item) => (item as Entry).search}>
          <CommandInput placeholder="Search runs, reports, actions…" />
          <CommandPanel>
            <CommandEmpty>Nothing matches.</CommandEmpty>
            <CommandList className="max-h-90 overflow-y-auto">
              {(group: (typeof groups)[number], index: number) => (
                <Fragment key={group.value}>
                  <CommandGroup items={group.items}>
                    <CommandGroupLabel>{group.value}</CommandGroupLabel>
                    <CommandCollection>
                      {(item: Entry) => (
                        <CommandItem key={item.value} value={item} onClick={() => run(item)} className="gap-2.5 text-[13px]">
                          <span className="flex size-4 items-center justify-center text-muted-foreground [&_svg]:size-3.5">
                            {item.icon}
                          </span>
                          <span className="truncate">{item.label}</span>
                          {item.hint && <span className="truncate text-muted-foreground text-xs">{item.hint}</span>}
                          {item.shortcut && <CommandShortcut>{item.shortcut}</CommandShortcut>}
                        </CommandItem>
                      )}
                    </CommandCollection>
                  </CommandGroup>
                  {index < groups.length - 1 && <CommandSeparator />}
                </Fragment>
              )}
            </CommandList>
          </CommandPanel>
        </Command>
      </CommandDialogPopup>
    </CommandDialog>
  );
}
