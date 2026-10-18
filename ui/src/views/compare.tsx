// Runs side by side: their validation curves on one chart, and a table that keys each
// color to its run with the numbers that tell them apart. Colors follow each run's place
// in the address, so removing one never repaints the others.
import { FileChartColumnIcon, XIcon } from "lucide-react";
import { useMemo } from "react";
import { Chart, type Series, type XAxis, type YAxis } from "@/components/chart";
import { Crumbs, PageBody, PageHeader } from "@/components/page";
import { Segmented } from "@/components/segmented";
import { StatusIcon } from "@/components/status-icon";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { type Columns, crossing } from "@/lib/curves";
import { count, DASH, datasetTitle, duration, optimizerTitle, percent, progress, runName } from "@/lib/format";
import { useLive } from "@/lib/live";
import { modelRef } from "@/lib/reports";
import { href, navigate } from "@/lib/router";
import { useRowsOf } from "@/lib/rows";
import { useStored } from "@/lib/stored";
import { SERIES, useDark } from "@/lib/theme";
import type { Run } from "@/lib/types";

interface Entry {
  run: Run;
  color: string;
  label: string;
}

export function CompareView({ ids, onNewReport }: { ids: string[]; onNewReport: (refs?: string[]) => void }) {
  const { runs, options } = useLive();
  const dark = useDark();
  const palette = SERIES[dark ? "dark" : "light"];
  const [y, setY] = useStored<YAxis>("compare.y", "accuracy");
  // Time by default: a step of SGD and a generation of LEEA cost very different amounts.
  const [x, setX] = useStored<XAxis>("compare.x", "time");
  const [threshold, setThreshold] = useStored<number>("compare.threshold", 0.95);

  const entries = useMemo(() => {
    const out: Entry[] = [];
    ids.slice(0, palette.length).forEach((id, index) => {
      const run = runs?.find((candidate) => candidate.id === id);
      if (run)
        out.push({ run, color: palette[index], label: runName(options, run) });
    });
    return out;
  }, [ids, runs, options, palette]);
  const rows = useRowsOf(
    entries.map(({ run }) => run),
    "val",
  );

  const series = useMemo<Series[]>(
    () =>
      entries.map((entry, index) => ({
        label: entry.label,
        color: entry.color,
        x: (x === "step" ? rows[index]?.step : rows[index]?.elapsed) ?? [],
        y: (y === "accuracy" ? rows[index]?.val_accuracy : rows[index]?.val_loss) ?? [],
      })),
    [entries, rows, x, y],
  );

  const datasets = [...new Set(entries.map(({ run }) => run.spec.dataset))];
  // A report compares models on one dataset: the first run's.
  const reportable = entries
    .filter(({ run }) => run.status.state === "completed" && run.spec.dataset === datasets[0])
    .map(({ run }) => modelRef(run.id));
  const remove = (id: string) => navigate(href.compare(ids.filter((other) => other !== id)));

  if (!runs) {
    return (
      <Frame>
        <Empty className="flex-1">
          <Spinner />
        </Empty>
      </Frame>
    );
  }
  if (!entries.length) {
    return (
      <Frame>
        <Empty className="flex-1">
          <EmptyHeader>
            <EmptyTitle className="font-medium text-base">Nothing to compare</EmptyTitle>
            <EmptyDescription>Select runs in the list, then choose Compare.</EmptyDescription>
          </EmptyHeader>
        </Empty>
      </Frame>
    );
  }

  const titles = [...new Set(entries.map(({ run }) => optimizerTitle(options, run.spec.optimizer)))];
  return (
    <Frame
      actions={
        reportable.length > 0 && (
          <Button variant="outline" size="sm" onClick={() => onNewReport(reportable)}>
            <FileChartColumnIcon />
            Report
          </Button>
        )
      }
    >
      <PageBody>
        <div className="mx-auto max-w-5xl px-8 py-7">
          <h1 className="font-medium text-xl tracking-tight">{titles.join(" vs ")}</h1>
          <p className="mt-1 text-muted-foreground text-xs">
            {entries.length} runs on {datasets.map((key) => datasetTitle(options, key)).join(", ")}
            {ids.length > palette.length && ` · the first ${palette.length} of ${ids.length}`}
          </p>
          <section className="mt-7">
            <div className="mb-3 flex items-center gap-3">
              <h2 className="font-medium text-[13px]">Validation</h2>
              <div className="ms-auto flex items-center gap-2">
                <Segmented
                  label="Measure"
                  value={y}
                  onChange={setY}
                  options={[
                    { value: "accuracy", label: "Accuracy" },
                    { value: "loss", label: "Loss" },
                  ]}
                />
                <Segmented
                  label="Horizontal axis"
                  value={x}
                  onChange={setX}
                  options={[
                    { value: "time", label: "Time" },
                    { value: "step", label: "Step" },
                  ]}
                />
              </div>
            </div>
            <Chart series={series} x={x} y={y} height={340} />
          </section>
          <Legend entries={entries} rows={rows} threshold={threshold} onThreshold={setThreshold} onRemove={remove} />
        </div>
      </PageBody>
    </Frame>
  );
}

function Frame({ children, actions }: { children: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <>
      <PageHeader actions={actions}>
        <Crumbs items={[{ label: "Runs", href: "/runs" }, { label: "Compare" }]} />
      </PageHeader>
      {children}
    </>
  );
}

/** The chart's legend as a table: one row per run, keyed by its line's color. */
function Legend({
  entries,
  rows,
  threshold,
  onThreshold,
  onRemove,
}: {
  entries: Entry[];
  rows: (Columns | null)[];
  threshold: number;
  onThreshold: (value: number) => void;
  onRemove: (id: string) => void;
}) {
  const head = "h-9 px-3 font-normal text-xs first:ps-0";
  const cell = "h-10 px-3 py-0 tabular-nums first:ps-0";
  return (
    <Table className="mt-6 text-[13px]">
      <TableHeader>
        <TableRow className="hover:bg-transparent">
          <TableHead className={head}>Run</TableHead>
          <TableHead className={`${head} text-right`}>Best validation</TableHead>
          <TableHead className={`${head} text-right`}>Test</TableHead>
          <TableHead className={`${head} text-right`}>Steps</TableHead>
          <TableHead className={`${head} text-right`}>Took</TableHead>
          <TableHead className={`${head} text-right`}>
            <label className="inline-flex items-center gap-1.5">
              Reaches
              <Input
                type="number"
                size="sm"
                min={1}
                max={100}
                step={0.5}
                value={Math.round(threshold * 1000) / 10}
                onChange={(event) => {
                  const value = Number(event.currentTarget.value);
                  if (value > 0 && value <= 100) onThreshold(value / 100);
                }}
                className="w-16 text-xs"
                aria-label="Accuracy to reach, in percent"
              />
              %
            </label>
          </TableHead>
          <TableHead className="w-8" />
        </TableRow>
      </TableHeader>
      <TableBody>
        {entries.map(({ run, color, label }, index) => {
          const reached = crossing(rows[index], threshold);
          const { status } = run;
          return (
            <TableRow key={run.id} className="group">
              <TableCell className={cell}>
                <a href={href.run(run.id)} className="flex items-center gap-2.5 hover:underline">
                  <span className="h-0.5 w-3.5 shrink-0 rounded-full" style={{ background: color }} />
                  <StatusIcon state={status.state} progress={progress(run)} />
                  <span className="font-medium">{label}</span>
                  {run.spec.label && <span className="truncate text-muted-foreground">{run.spec.label}</span>}
                </a>
              </TableCell>
              <TableCell className={`${cell} text-right`}>{percent(status.best_accuracy, 2)}</TableCell>
              <TableCell className={`${cell} text-right`}>{percent(status.test_accuracy, 2)}</TableCell>
              <TableCell className={`${cell} text-right text-muted-foreground`}>{count(status.step ?? 0)}</TableCell>
              <TableCell className={`${cell} text-right text-muted-foreground`}>{duration(status.elapsed_seconds)}</TableCell>
              <TableCell className={`${cell} text-right`}>
                {reached ? (
                  <>
                    {duration(reached.elapsed)}
                    <span className="text-muted-foreground"> · step {count(reached.step)}</span>
                  </>
                ) : (
                  <span className="text-muted-foreground">{DASH}</span>
                )}
              </TableCell>
              <TableCell className="h-10 py-0 text-right">
                <Button
                  variant="ghost"
                  size="icon-xs"
                  className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                  onClick={() => onRemove(run.id)}
                  aria-label={`Remove ${label} from the comparison`}
                >
                  <XIcon />
                </Button>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
