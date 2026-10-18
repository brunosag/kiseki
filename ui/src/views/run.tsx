// One run: its numbers and curves, what happened to it, its saved models and log, and its
// full setup in the side panel.
import {
  CopyIcon,
  DownloadIcon,
  FileChartColumnIcon,
  PauseIcon,
  PlayIcon,
  SquareIcon,
  TriangleAlertIcon,
} from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import type { RunDraft } from "@/app";
import { Chart, type Series, type XAxis, type YAxis } from "@/components/chart";
import { Crumbs, PageBody, PageHeader, PanelSection, Property } from "@/components/page";
import { Segmented } from "@/components/segmented";
import { STATE_LABEL, StatusIcon } from "@/components/status-icon";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs";
import { control } from "@/lib/actions";
import { fileURL } from "@/lib/api";
import {
  canResume,
  changedParams,
  count,
  DASH,
  datasetTitle,
  dateTime,
  duration,
  gpuName,
  isActive,
  number,
  optimizerTitle,
  percent,
  progress,
  runName,
  shortId,
  words,
} from "@/lib/format";
import { useLive } from "@/lib/live";
import { modelRef } from "@/lib/reports";
import { useResource, useText } from "@/lib/resource";
import { useRows } from "@/lib/rows";
import { useStored } from "@/lib/stored";
import { SERIES, useDark } from "@/lib/theme";
import type { Json, Options, Run, RunDetail, RunEvent } from "@/lib/types";

export function RunView({
  id,
  onNewRun,
  onNewReport,
}: {
  id: string;
  onNewRun: (draft?: RunDraft) => void;
  onNewReport: (refs?: string[]) => void;
}) {
  const { runs, options } = useLive();
  const listed = runs?.find((run) => run.id === id);
  const checkpoint = Number(listed?.spec.checkpoint_interval ?? 1000);
  // Models and events change when the state does and at checkpoints. A finished run is
  // fetched again on each announcement: a remote run's models arrive after it finishes.
  const version =
    listed && !isActive(listed.status.state)
      ? listed.status
      : `${listed?.status.state}/${Math.floor((listed?.status.step ?? 0) / checkpoint)}`;
  const detail = useResource<RunDetail>(`runs/${encodeURIComponent(id)}`, version);
  const run: Run | undefined = listed ?? detail.data;

  if (!run) {
    return (
      <>
        <PageHeader>
          <Crumbs items={[{ label: "Runs", href: "/runs" }, { label: shortId(id) }]} />
        </PageHeader>
        <Empty className="flex-1">
          {detail.error ? (
            <EmptyHeader>
              <EmptyTitle className="font-medium text-base">No such run</EmptyTitle>
              <EmptyDescription>{detail.error}</EmptyDescription>
            </EmptyHeader>
          ) : (
            <Spinner />
          )}
        </Empty>
      </>
    );
  }

  const { spec, status } = run;
  const state = status.state;
  const title = `${optimizerTitle(options, spec.optimizer)} on ${datasetTitle(options, spec.dataset)}`;
  const models = detail.data?.models ?? [];

  return (
    <>
      <PageHeader
        actions={
          <>
            {isActive(state) && (
              <>
                <Button variant="ghost" size="sm" onClick={() => control([run], "pause")}>
                  <PauseIcon />
                  Pause
                </Button>
                <Button variant="ghost" size="sm" onClick={() => control([run], "stop")}>
                  <SquareIcon />
                  Stop
                </Button>
              </>
            )}
            {canResume(state) && (
              <Button variant="ghost" size="sm" onClick={() => control([run], "resume")}>
                <PlayIcon />
                Resume
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={() => onNewRun(draftFrom(run))}>
              <CopyIcon />
              Run again
            </Button>
            {models.includes("best") && (
              <Button variant="outline" size="sm" onClick={() => onNewReport([modelRef(run.id)])}>
                <FileChartColumnIcon />
                Report
              </Button>
            )}
          </>
        }
      >
        <Crumbs items={[{ label: "Runs", href: "/runs" }, { label: runName(options, run) }]} />
      </PageHeader>
      <div className="flex min-h-0 flex-1">
        <PageBody>
          <div className="mx-auto max-w-5xl px-8 py-7">
            <h1 className="font-medium text-xl tracking-tight">{title}</h1>
            <div className="mt-1 flex items-center gap-2 text-muted-foreground text-xs">
              <span className="font-mono">{run.id}</span>
              {spec.label && (
                <Badge variant="outline" className="font-normal">
                  {spec.label}
                </Badge>
              )}
            </div>
            {state === "failed" && status.error && (
              <Alert variant="error" className="mt-5">
                <TriangleAlertIcon />
                <AlertTitle>Failed at step {count(status.step ?? 0)}</AlertTitle>
                <AlertDescription className="font-mono text-xs">{status.error}</AlertDescription>
              </Alert>
            )}
            <Numbers run={run} />
            <Curves run={run} />
            <Tabs defaultValue="activity" className="mt-8">
              <TabsList variant="underline" className="w-full justify-start border-b">
                <TabsTab value="activity" className="grow-0 px-2 text-[13px] sm:text-[13px]">
                  Activity
                </TabsTab>
                <TabsTab value="models" className="grow-0 px-2 text-[13px] sm:text-[13px]">
                  Models
                  <span className="text-muted-foreground tabular-nums">{models.length}</span>
                </TabsTab>
                <TabsTab value="log" className="grow-0 px-2 text-[13px] sm:text-[13px]">
                  Log
                </TabsTab>
              </TabsList>
              <TabsPanel value="activity" className="pt-3">
                <Activity events={detail.data?.events ?? []} />
              </TabsPanel>
              <TabsPanel value="models" className="pt-3">
                <Models run={run} models={models} onNewReport={onNewReport} />
              </TabsPanel>
              <TabsPanel value="log" className="pt-3">
                <Log run={run} />
              </TabsPanel>
            </Tabs>
          </div>
        </PageBody>
        <Properties run={run} options={options} events={detail.data?.events} />
      </div>
    </>
  );
}

/** What the new-run form starts from to run this again. */
function draftFrom(run: Run): RunDraft {
  const { params, ...rest } = run.spec;
  return { ...rest, ...(params as Record<string, Json>), label: run.spec.label };
}

function Numbers({ run }: { run: Run }) {
  const { status } = run;
  const items: [string, string, string?][] = [
    [
      "Test accuracy",
      percent(status.test_accuracy, 2),
      status.test_accuracy != null ? "best model" : "measured at the end",
    ],
    [
      "Best validation",
      percent(status.best_accuracy, 2),
      status.best_step != null ? `at step ${count(status.best_step)}` : undefined,
    ],
    [
      "Speed",
      status.steps_per_second ? `${number(status.steps_per_second)}` : DASH,
      status.steps_per_second ? "steps/s" : undefined,
    ],
    [
      isActive(status.state) ? "Elapsed" : "Took",
      duration(status.elapsed_seconds),
      status.state === "running" && status.eta_seconds ? `${duration(status.eta_seconds)} left` : undefined,
    ],
  ];
  return (
    <dl className="mt-6 grid grid-cols-4 divide-x rounded-xl border">
      {items.map(([label, value, note]) => (
        <div key={label} className="px-4 py-3">
          <dt className="text-muted-foreground text-xs">{label}</dt>
          <dd className="mt-1 font-medium text-lg tabular-nums tracking-tight">{value}</dd>
          <dd className="h-4 truncate text-muted-foreground text-xs">{note}</dd>
        </div>
      ))}
    </dl>
  );
}

function Curves({ run }: { run: Run }) {
  const [y, setY] = useStored<YAxis>("run.y", "accuracy");
  const [x, setX] = useStored<XAxis>("run.x", "step");
  const val = useRows(run.id, "val", run.status);
  const train = useRows(run.id, "train", run.status);
  const dark = useDark();
  const color = SERIES[dark ? "dark" : "light"][0];
  const series = useMemo(() => {
    const out: Series[] = [];
    // Training rows carry no time, so the time axis shows validation alone.
    if (x === "step" && train?.step)
      out.push({
        label: "Training batch",
        color: "var(--muted-foreground)",
        x: train.step,
        y: (y === "accuracy" ? train.accuracy : train.loss) ?? [],
        faint: true,
      });
    if (val?.step)
      out.push({
        label: "Validation",
        color,
        x: x === "step" ? val.step : (val.elapsed ?? []),
        y: (y === "accuracy" ? val.val_accuracy : val.val_loss) ?? [],
      });
    return out;
  }, [val, train, x, y, color]);
  return (
    <section className="mt-8">
      <div className="mb-3 flex items-center gap-3">
        <h2 className="font-medium text-[13px]">Curves</h2>
        <div className="flex items-center gap-3 text-muted-foreground text-xs">
          {series.map((s) => (
            <span key={s.label} className="flex items-center gap-1.5">
              <span
                className="h-0.5 w-3 rounded-full"
                style={{ background: s.faint ? "color-mix(in oklab, var(--muted-foreground) 50%, transparent)" : s.color }}
              />
              {s.label}
            </span>
          ))}
        </div>
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
              { value: "step", label: "Step" },
              { value: "time", label: "Time" },
            ]}
          />
        </div>
      </div>
      {val?.step?.length || train?.step?.length ? (
        <Chart series={series} x={x} y={y} height={300} />
      ) : (
        <div className="flex h-75 items-center justify-center rounded-xl border border-dashed text-muted-foreground text-sm">
          {isActive(run.status.state) ? "The first validation will draw here." : "No curves recorded."}
        </div>
      )}
    </section>
  );
}

function describe(event: RunEvent): string {
  const step = typeof event.step === "number" ? count(event.step) : null;
  switch (event.kind) {
    case "created":
      return `Queued on ${event.machine ?? "local"}`;
    case "started": {
      const device = event.device as Record<string, Json> | null;
      return `Started on ${device?.device ?? "the CPU"}${device?.jax ? ` · JAX ${device.jax}` : ""}`;
    }
    case "resumed":
      return `Resumed from step ${step}`;
    case "compiled": {
      const chunk = Number(event.chunk_size ?? 1);
      return `Compiled in ${duration(Number(event.seconds))}${chunk > 1 ? ` · ${chunk} networks per chunk` : ""}`;
    }
    case "paused":
    case "stopped":
      return `${STATE_LABEL[event.kind]} at step ${step}`;
    case "requeued":
      return `Queued again after being ${event.previous}`;
    case "completed": {
      const best = (event.summary as Record<string, Record<string, Json>> | null)?.best;
      return `Completed at step ${step}${best ? ` · test ${percent(Number(best.test_accuracy), 2)}` : ""}`;
    }
    case "failed":
      return `Failed${step ? ` at step ${step}` : ""}: ${event.error}`;
    default:
      return words(event.kind);
  }
}

function Activity({ events }: { events: RunEvent[] }) {
  if (!events.length) return <p className="py-2 text-muted-foreground text-sm">Nothing yet.</p>;
  return (
    <ol className="relative">
      {events.map((event, index) => (
        <li key={index} className="relative flex gap-3 pb-3 text-[13px] last:pb-0">
          {index < events.length - 1 && <span className="absolute top-4 bottom-0 left-[5px] w-px bg-border" />}
          <span className="mt-1.5 size-[11px] shrink-0 rounded-full border-2 border-background bg-muted-foreground/40 ring-1 ring-border" />
          <span className={event.kind === "failed" ? "text-destructive-foreground" : undefined}>
            {describe(event)}
          </span>
          <span className="ms-auto shrink-0 text-muted-foreground text-xs">{dateTime(event.time)}</span>
        </li>
      ))}
    </ol>
  );
}

function modelNote(name: string, run: Run): string {
  if (name === "best")
    return run.status.best_step != null ? `Best validation, step ${count(run.status.best_step)}` : "Best validation";
  if (name === "final") return "At the last step";
  const step = /^step-(\d+)$/.exec(name);
  return step ? `Checkpoint at step ${count(Number(step[1]))}` : "";
}

function Models({
  run,
  models,
  onNewReport,
}: {
  run: Run;
  models: string[];
  onNewReport: (refs?: string[]) => void;
}) {
  if (!models.length)
    return (
      <p className="py-2 text-muted-foreground text-sm">
        {isActive(run.status.state) ? "Models are saved at checkpoints and at the end." : "No saved models."}
      </p>
    );
  return (
    <ul className="divide-y rounded-xl border">
      {models.map((name) => (
        <li key={name} className="flex h-11 items-center gap-3 px-4 text-[13px]">
          <span className="w-20 font-mono text-xs">{name}</span>
          <span className="text-muted-foreground">{modelNote(name, run)}</span>
          <div className="ms-auto flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={() => onNewReport([`${run.id}:${name}`])}>
              <FileChartColumnIcon />
              Report
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Download ${name}`}
              render={<a href={fileURL("runs", run.id, `models/${name}.safetensors`)} download={`${run.id}-${name}.safetensors`} />}
            >
              <DownloadIcon />
            </Button>
          </div>
        </li>
      ))}
    </ul>
  );
}

function Log({ run }: { run: Run }) {
  const text = useText(fileURL("runs", run.id, "worker.log"), `${run.status.state}/${run.status.updated_at}`);
  const end = useRef<HTMLPreElement>(null);
  const lines = useMemo(() => (text ?? "").split("\n").slice(-500).join("\n"), [text]);
  useEffect(() => {
    end.current?.scrollTo({ top: end.current.scrollHeight });
  }, [lines]);
  if (text === null) return <Spinner className="m-2" />;
  if (!text.trim()) return <p className="py-2 text-muted-foreground text-sm">The log is empty.</p>;
  return (
    <pre
      ref={end}
      className="max-h-96 overflow-auto rounded-xl border bg-muted/40 p-4 font-mono text-muted-foreground text-xs leading-relaxed"
    >
      {lines}
    </pre>
  );
}

function Properties({ run, options, events }: { run: Run; options: Options | null; events?: RunEvent[] }) {
  const { spec, status } = run;
  const changed = new Set(changedParams(run, options).map(([key]) => key));
  const started = events?.findLast((event) => event.kind === "started" || event.kind === "resumed");
  const info = started?.device as Record<string, Json> | undefined;
  return (
    <aside className="w-80 shrink-0 overflow-y-auto border-l">
      <PanelSection title="Run">
        <Property label="State">
          <span className="flex items-center gap-2">
            <StatusIcon state={status.state} progress={progress(run)} />
            {STATE_LABEL[status.state] ?? status.state}
          </span>
        </Property>
        <Property label="Progress">
          <span className="flex items-center gap-2 tabular-nums">
            <span className="h-1 w-16 overflow-hidden rounded-full bg-muted">
              <span className="block h-full rounded-full bg-foreground/60" style={{ width: `${progress(run) * 100}%` }} />
            </span>
            {count(status.step ?? 0)} / {count(status.iterations ?? spec.iterations)}
          </span>
        </Property>
        <Property label="Machine">{status.machine ?? "local"}</Property>
        <Property label="Device">
          <span title={String(info?.device ?? status.device ?? "")}>
            {gpuName(String(info?.device ?? status.device ?? DASH))}
          </span>
        </Property>
      </PanelSection>
      <PanelSection title="Setup">
        <Property label="Dataset">{datasetTitle(options, spec.dataset)}</Property>
        <Property label="Model">
          <span className="font-mono text-xs">{spec.model || options?.datasets[spec.dataset]?.default_model}</span>
        </Property>
        <Property label="Optimizer">{optimizerTitle(options, spec.optimizer)}</Property>
        <Property label="Steps">{count(spec.iterations)}</Property>
        <Property label="Seed">{spec.seed}</Property>
        <Property label="Batch size">{number(spec.batch_size)}</Property>
        <Property label="Precision">{String(spec.precision ?? DASH)}</Property>
        <Property label="Validate every">{number(spec.validation_interval)} steps</Property>
        <Property label="Checkpoint every">{number(spec.checkpoint_interval)} steps</Property>
        {spec.target_accuracy != null && <Property label="Stop at">{percent(Number(spec.target_accuracy))}</Property>}
      </PanelSection>
      <PanelSection title="Parameters">
        {Object.entries(spec.params).map(([key, value]) => (
          <Property key={key} label={words(key)}>
            <span className={changed.has(key) ? "font-medium" : "text-muted-foreground"}>{number(value)}</span>
          </Property>
        ))}
      </PanelSection>
      <PanelSection title="Times">
        <Property label="Created">{dateTime(status.created_at)}</Property>
        <Property label="Started">{dateTime(status.started_at)}</Property>
        <Property label="Finished">{dateTime(status.finished_at)}</Property>
      </PanelSection>
    </aside>
  );
}
