// Every run, grouped (by state unless chosen otherwise), newest first. Rows open the run;
// a selection compares runs, builds a report from them, or stops, pauses and resumes them.
import { ChartLineIcon, FileChartColumnIcon, ListFilterIcon, ServerIcon, SlidersHorizontalIcon, XIcon } from "lucide-react";
import { memo, useMemo, useState } from "react";
import { PageBody, PageHeader } from "@/components/page";
import { Segmented } from "@/components/segmented";
import { STATE_LABEL, STATE_ORDER, StatusIcon } from "@/components/status-icon";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import {
  Menu,
  MenuCheckboxItem,
  MenuGroup,
  MenuGroupLabel,
  MenuItem,
  MenuPopup,
  MenuRadioGroup,
  MenuRadioItem,
  MenuSeparator,
  MenuTrigger,
} from "@/components/ui/menu";
import { control } from "@/lib/actions";
import {
  ago,
  canResume,
  changedParams,
  datasetTitle,
  duration,
  isActive,
  number,
  optimizerTitle,
  percent,
  progress,
  shortId,
  words,
} from "@/lib/format";
import { useLive } from "@/lib/live";
import { modelRef } from "@/lib/reports";
import { href, navigate } from "@/lib/router";
import { useStored } from "@/lib/stored";
import type { Options, Run } from "@/lib/types";
import { cn } from "@/lib/utils";

type Show = "all" | "active" | "finished";
type Group = "state" | "dataset" | "optimizer" | "none";
type Filters = { dataset: string[]; optimizer: string[]; machine: string[] };

const NO_FILTERS: Filters = { dataset: [], optimizer: [], machine: [] };

export function RunsView({ onNewRun, onNewReport }: { onNewRun: () => void; onNewReport: (refs: string[]) => void }) {
  const { runs, options } = useLive();
  const [show, setShow] = useStored<Show>("runs.show", "all");
  const [group, setGroup] = useStored<Group>("runs.group", "state");
  const [filters, setFilters] = useStored<Filters>("runs.filters", NO_FILTERS);
  const [text, setText] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const visible = useMemo(() => {
    const needle = text.trim().toLowerCase();
    return (runs ?? []).filter((run) => {
      const state = run.status.state;
      if (show === "active" && !isActive(state) && state !== "paused") return false;
      if (show === "finished" && (isActive(state) || state === "paused")) return false;
      if (filters.dataset.length && !filters.dataset.includes(run.spec.dataset)) return false;
      if (filters.optimizer.length && !filters.optimizer.includes(run.spec.optimizer)) return false;
      if (filters.machine.length && !filters.machine.includes(run.status.machine ?? "local")) return false;
      if (needle) {
        const haystack = `${run.id} ${run.spec.label} ${optimizerTitle(options, run.spec.optimizer)}`.toLowerCase();
        if (!haystack.includes(needle)) return false;
      }
      return true;
    });
  }, [runs, options, show, filters, text]);

  const groups = useMemo(() => groupRuns(visible, group, options), [visible, group, options]);

  const toggle = (id: string) =>
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const filterCount = filters.dataset.length + filters.optimizer.length + filters.machine.length;

  return (
    <>
      <PageHeader
        actions={
          <>
            <Input
              size="sm"
              type="search"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Filter by name or label"
              className="w-52 text-[13px] sm:text-[13px]"
              aria-label="Filter runs by name or label"
            />
            <FilterMenu options={options} runs={runs ?? []} filters={filters} onChange={setFilters} count={filterCount} />
            <Menu>
              <MenuTrigger render={<Button variant="ghost" size="sm" className="text-[13px]" />}>
                <SlidersHorizontalIcon />
                Display
              </MenuTrigger>
              <MenuPopup align="end" className="min-w-44">
                <MenuGroup>
                  <MenuGroupLabel>Group by</MenuGroupLabel>
                  <MenuRadioGroup value={group} onValueChange={(value) => setGroup(value as Group)}>
                    <MenuRadioItem value="state">State</MenuRadioItem>
                    <MenuRadioItem value="dataset">Dataset</MenuRadioItem>
                    <MenuRadioItem value="optimizer">Optimizer</MenuRadioItem>
                    <MenuRadioItem value="none">No grouping</MenuRadioItem>
                  </MenuRadioGroup>
                </MenuGroup>
              </MenuPopup>
            </Menu>
          </>
        }
      >
        <span className="me-2 font-medium">Runs</span>
        <Segmented
          label="Which runs"
          value={show}
          onChange={setShow}
          options={[
            { value: "all", label: "All" },
            { value: "active", label: "Active" },
            { value: "finished", label: "Finished" },
          ]}
        />
      </PageHeader>
      <PageBody>
        {runs && !visible.length ? (
          <Empty className="h-full">
            <EmptyHeader>
              <EmptyTitle className="font-medium text-base">{runs.length ? "No runs match" : "No runs yet"}</EmptyTitle>
              <EmptyDescription>
                {runs.length ? "Change the filters to see more." : "Queue one here, or with kiseki run from the CLI."}
              </EmptyDescription>
            </EmptyHeader>
            {!runs.length && (
              <Button size="sm" onClick={onNewRun}>
                New run
              </Button>
            )}
          </Empty>
        ) : (
          <div role="list" aria-label="Runs" onKeyDown={(event) => listKeys(event, toggle, () => setSelected(new Set()))}>
            {groups.map(([key, label, items]) => (
              <section key={key}>
                {group !== "none" && (
                  <div className="sticky top-0 z-10 flex h-9 items-center gap-2 border-b bg-muted/60 px-4 text-[13px] backdrop-blur-sm">
                    {group === "state" && <StatusIcon state={key} progress={0.5} />}
                    <span className="font-medium">{label}</span>
                    <span className="text-muted-foreground tabular-nums">{items.length}</span>
                  </div>
                )}
                {items.map((run) => (
                  <RunRow
                    key={run.id}
                    run={run}
                    options={options}
                    selected={selected.has(run.id)}
                    selecting={selected.size > 0}
                    onToggle={toggle}
                  />
                ))}
              </section>
            ))}
          </div>
        )}
      </PageBody>
      {selected.size > 0 && (
        <SelectionBar
          runs={(runs ?? []).filter((run) => selected.has(run.id))}
          onClear={() => setSelected(new Set())}
          onNewReport={onNewReport}
        />
      )}
    </>
  );
}

function groupRuns(runs: Run[], group: Group, options: Options | null): [string, string, Run[]][] {
  if (group === "none") return [["all", "All", runs]];
  const key = (run: Run) =>
    group === "state" ? run.status.state : group === "dataset" ? run.spec.dataset : run.spec.optimizer;
  const order =
    group === "state"
      ? STATE_ORDER
      : group === "dataset"
        ? Object.keys(options?.datasets ?? {})
        : Object.keys(options?.optimizers ?? {});
  const byKey = new Map<string, Run[]>();
  for (const run of runs) byKey.set(key(run), [...(byKey.get(key(run)) ?? []), run]);
  const rank = (k: string) => (order.includes(k) ? order.indexOf(k) : order.length);
  return [...byKey.entries()]
    .sort(([a], [b]) => rank(a) - rank(b))
    .map(([k, items]) => [
      k,
      group === "state"
        ? (STATE_LABEL[k] ?? k)
        : group === "dataset"
          ? datasetTitle(options, k)
          : optimizerTitle(options, k),
      items,
    ]);
}

/** ↑/↓ or j/k move between rows, x selects, Esc clears the selection. */
function listKeys(event: React.KeyboardEvent, toggle: (id: string) => void, clear: () => void) {
  const rows = [...event.currentTarget.querySelectorAll<HTMLElement>("[data-row]")];
  const index = rows.indexOf(document.activeElement as HTMLElement);
  const move = (step: number) => {
    event.preventDefault();
    rows[Math.max(0, Math.min(rows.length - 1, index + step))]?.focus();
  };
  if (event.key === "ArrowDown" || event.key === "j") move(1);
  else if (event.key === "ArrowUp" || event.key === "k") move(-1);
  else if (event.key === "x" && index >= 0) toggle(rows[index].dataset.row!);
  else if (event.key === "Escape") clear();
}

const RunRow = memo(function RunRow({
  run,
  options,
  selected,
  selecting,
  onToggle,
}: {
  run: Run;
  options: Options | null;
  selected: boolean;
  selecting: boolean;
  onToggle: (id: string) => void;
}) {
  const { status, spec } = run;
  const state = status.state;
  const params = changedParams(run, options);
  const defaultModel = options?.datasets[spec.dataset]?.default_model;
  const open = () => navigate(href.run(run.id));
  return (
    <div
      role="listitem"
      tabIndex={0}
      data-row={run.id}
      aria-selected={selected}
      onClick={open}
      onKeyDown={(event) => event.key === "Enter" && open()}
      className={cn(
        "group flex h-10 cursor-default items-center gap-3 px-4 text-[13px] outline-none [contain-intrinsic-size:auto_2.5rem] [content-visibility:auto] hover:bg-accent focus-visible:bg-accent",
        selected && "bg-info/6 hover:bg-info/10",
      )}
    >
      <span
        className={cn("flex w-4 shrink-0 items-center", !selecting && !selected && "opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100")}
        onClick={(event) => event.stopPropagation()}
      >
        <Checkbox checked={selected} onCheckedChange={() => onToggle(run.id)} aria-label={`Select run ${shortId(run.id)}`} />
      </span>
      <StatusIcon state={state} progress={progress(run)} />
      <span className="w-10 shrink-0 font-mono text-muted-foreground text-xs">{shortId(run.id)}</span>
      <span className="flex min-w-0 flex-1 items-center gap-2">
        <span className="shrink-0 font-medium">{optimizerTitle(options, spec.optimizer)}</span>
        <span className="shrink-0 text-muted-foreground">{datasetTitle(options, spec.dataset)}</span>
        {spec.model && spec.model !== defaultModel && (
          <span className="shrink-0 font-mono text-muted-foreground text-xs">{spec.model}</span>
        )}
        {state === "failed" && status.error ? (
          <span className="truncate text-destructive-foreground" title={status.error}>
            {status.error}
          </span>
        ) : (
          <span className="truncate text-muted-foreground/80">
            {params.map(([key, value]) => `${words(key)} ${number(value)}`).join(" · ")}
          </span>
        )}
      </span>
      {spec.label && (
        <Badge variant="outline" className="max-w-40 shrink truncate font-normal text-muted-foreground">
          <span className="truncate">{spec.label}</span>
        </Badge>
      )}
      {status.machine && status.machine !== "local" && (
        <span className="flex shrink-0 items-center gap-1 text-muted-foreground text-xs">
          <ServerIcon className="size-3" />
          {status.machine}
        </span>
      )}
      <span className="flex w-24 shrink-0 items-center justify-end gap-2 text-muted-foreground text-xs tabular-nums">
        {state === "running" || state === "paused" ? (
          <>
            <span>{Math.round(progress(run) * 100)}%</span>
            <span className="h-1 w-12 overflow-hidden rounded-full bg-muted">
              <span className="block h-full rounded-full bg-amber-500" style={{ width: `${progress(run) * 100}%` }} />
            </span>
          </>
        ) : state === "queued" ? null : (
          duration(status.elapsed_seconds)
        )}
      </span>
      <span
        className="w-14 shrink-0 text-right tabular-nums"
        title={state === "completed" ? "Test accuracy of the best model" : "Best validation accuracy"}
      >
        {state === "completed" ? percent(status.test_accuracy) : (
          <span className="text-muted-foreground">{percent(status.best_accuracy)}</span>
        )}
      </span>
      <span className="w-12 shrink-0 text-right text-muted-foreground text-xs" title={status.created_at}>
        {ago(status.created_at)}
      </span>
    </div>
  );
});

function FilterMenu({
  options,
  runs,
  filters,
  onChange,
  count,
}: {
  options: Options | null;
  runs: Run[];
  filters: Filters;
  onChange: (filters: Filters) => void;
  count: number;
}) {
  const machines = [...new Set(runs.map((run) => run.status.machine ?? "local"))].sort();
  const flip = (axis: keyof Filters, value: string) =>
    onChange({
      ...filters,
      [axis]: filters[axis].includes(value) ? filters[axis].filter((v) => v !== value) : [...filters[axis], value],
    });
  const axes: [keyof Filters, string, [string, string][]][] = [
    ["dataset", "Dataset", Object.entries(options?.datasets ?? {}).map(([key, d]) => [key, d.title])],
    ["optimizer", "Optimizer", Object.entries(options?.optimizers ?? {}).map(([key, o]) => [key, o.title])],
    ["machine", "Machine", machines.map((m) => [m, m])],
  ];
  return (
    <Menu>
      <MenuTrigger render={<Button variant="ghost" size="sm" className="text-[13px]" />}>
        <ListFilterIcon />
        Filter
        {count > 0 && (
          <Badge variant="secondary" size="sm" className="tabular-nums">
            {count}
          </Badge>
        )}
      </MenuTrigger>
      <MenuPopup align="end" className="min-w-48">
        {axes.map(([axis, title, values], index) => (
          <MenuGroup key={axis}>
            {index > 0 && <MenuSeparator />}
            <MenuGroupLabel>{title}</MenuGroupLabel>
            {values.map(([value, label]) => (
              <MenuCheckboxItem
                key={value}
                checked={filters[axis].includes(value)}
                onCheckedChange={() => flip(axis, value)}
                closeOnClick={false}
              >
                {label}
              </MenuCheckboxItem>
            ))}
          </MenuGroup>
        ))}
        {count > 0 && (
          <>
            <MenuSeparator />
            <MenuItem onClick={() => onChange(NO_FILTERS)}>
              <XIcon />
              Clear filters
            </MenuItem>
          </>
        )}
      </MenuPopup>
    </Menu>
  );
}

function SelectionBar({
  runs,
  onClear,
  onNewReport,
}: {
  runs: Run[];
  onClear: () => void;
  onNewReport: (refs: string[]) => void;
}) {
  const active = runs.filter((run) => isActive(run.status.state));
  const resumable = runs.filter((run) => canResume(run.status.state));
  const finished = runs.filter((run) => run.status.state === "completed");
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-5 z-20 flex justify-center">
      <div className="pointer-events-auto flex h-10 items-center gap-1 rounded-xl border bg-popover px-1.5 text-[13px] shadow-lg/10">
        <span className="px-2 tabular-nums">
          {runs.length} selected
        </span>
        <span className="mx-1 h-4 w-px bg-border" />
        <Button variant="ghost" size="sm" disabled={runs.length < 1} onClick={() => navigate(href.compare(runs.map((r) => r.id)))}>
          <ChartLineIcon />
          Compare
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={!finished.length}
          onClick={() => onNewReport(finished.map((run) => modelRef(run.id)))}
        >
          <FileChartColumnIcon />
          Report
        </Button>
        {active.length > 0 && (
          <>
            <Button variant="ghost" size="sm" onClick={() => control(active, "pause")}>
              Pause
            </Button>
            <Button variant="ghost" size="sm" onClick={() => control(active, "stop")}>
              Stop
            </Button>
          </>
        )}
        {resumable.length > 0 && (
          <Button variant="ghost" size="sm" onClick={() => control(resumable, "resume")}>
            Resume
          </Button>
        )}
        <span className="mx-1 h-4 w-px bg-border" />
        <Button variant="ghost" size="icon-sm" aria-label="Clear selection" onClick={onClear}>
          <XIcon />
        </Button>
      </div>
    </div>
  );
}

