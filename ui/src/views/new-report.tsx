// The new-report composer: which trained models to compare (each run's best by default),
// on which split, with which analyses. A report compares models of one dataset, so the
// first pick rules out runs on the others.
import { TriangleAlertIcon } from "lucide-react";
import { useRef, useState } from "react";
import { Segmented } from "@/components/segmented";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogPopup, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import { toastManager } from "@/components/ui/toast";
import { api } from "@/lib/api";
import { datasetTitle, optimizerTitle, percent, shortId } from "@/lib/format";
import { addReport, useLive } from "@/lib/live";
import { modelRef, refParts, runOf } from "@/lib/reports";
import { href, navigate } from "@/lib/router";
import type { Options, Report, ReportSpec, Run } from "@/lib/types";
import { cn } from "@/lib/utils";

export function NewReportDialog({ models, onClose }: { models: string[] | undefined; onClose: () => void }) {
  const { options, runs } = useLive();
  const last = useRef<{ models: string[]; session: number }>({ models: [], session: 0 });
  if (models !== undefined && models !== last.current.models) last.current = { models, session: last.current.session + 1 };
  return (
    <Dialog open={models !== undefined} onOpenChange={(open) => !open && onClose()}>
      <DialogPopup className="max-w-xl" showCloseButton={false}>
        {options && runs && (
          <Composer
            key={last.current.session}
            initial={last.current.models}
            options={options}
            runs={runs}
            onDone={onClose}
          />
        )}
      </DialogPopup>
    </Dialog>
  );
}

function Composer({
  initial,
  options,
  runs,
  onDone,
}: {
  initial: string[];
  options: Options;
  runs: Run[];
  onDone: () => void;
}) {
  // Models handed in from other datasets than the first one's stay unpicked.
  const [picked, setPicked] = useState<string[]>(() =>
    initial.filter((ref) => runOf(ref, runs)?.spec.dataset === runOf(initial[0], runs)?.spec.dataset),
  );
  const [split, setSplit] = useState<ReportSpec["split"]>("test");
  const [analyses, setAnalyses] = useState<string[]>(Object.keys(options.analyses));
  const [seed, setSeed] = useState("0");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

    // Each finished run offers its best model; models handed in (another checkpoint, an
  // unfinished run's best so far) join the list.
  const refs = [
    ...new Set([...initial, ...runs.filter((run) => run.status.state === "completed").map((run) => modelRef(run.id))]),
  ];
  const dataset = picked.length ? runOf(picked[0], runs)?.spec.dataset : undefined;
  const validSeed = /^\d+$/.test(seed.trim());
  const ready = picked.length > 0 && analyses.length > 0 && validSeed && !busy;

  const toggle = (ref: string) =>
    setPicked((previous) => (previous.includes(ref) ? previous.filter((other) => other !== ref) : [...previous, ref]));

  const submit = async () => {
    if (!ready) return;
    setBusy(true);
    setFailure(null);
    try {
      const spec: ReportSpec = {
        models: picked,
        labels: [],
        split,
        analyses: Object.fromEntries(analyses.map((name) => [name, {}])),
        seed: Number(seed),
      };
      const report = await api<Report>("reports", spec);
      addReport(report);
      toastManager.add({ type: "success", title: "Building the report" });
      onDone();
      navigate(href.report(report.id));
    } catch (error) {
      setFailure((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      className="flex min-h-0 flex-col"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
          event.preventDefault();
          submit();
        }
      }}
    >
      <div className="px-5 pt-4 pb-3">
        <DialogTitle className="font-medium font-sans text-muted-foreground text-xs">New report</DialogTitle>
        <p className="mt-3 font-medium text-lg">Compare trained models</p>
      </div>
      <div className="min-h-0 overflow-y-auto border-t px-5 py-4">
        <h3 className="mb-2 flex items-baseline gap-2 text-xs">
          <span className="font-medium">Models</span>
          <span className="text-muted-foreground">
            {dataset ? `on ${datasetTitle(options, dataset)}` : "each run's best checkpoint"}
          </span>
        </h3>
        {refs.length ? (
          <ul className="-mx-2 max-h-60 overflow-y-auto">
            {refs.map((ref) => {
              const run = runOf(ref, runs);
              const { run: id, model } = refParts(ref);
              const other = !!dataset && run?.spec.dataset !== dataset;
              return (
                <li key={ref}>
                  <label
                    className={cn(
                      "flex h-8 cursor-default items-center gap-2.5 rounded-md px-2 text-[13px] hover:bg-accent",
                      other && "opacity-50",
                    )}
                  >
                    <Checkbox checked={picked.includes(ref)} disabled={other && !picked.includes(ref)} onCheckedChange={() => toggle(ref)} />
                    <span className="font-medium">{run ? optimizerTitle(options, run.spec.optimizer) : "Run"}</span>
                    <span className="font-mono text-muted-foreground text-xs">{shortId(id)}</span>
                    {model !== "best" && <span className="text-muted-foreground text-xs">{model}</span>}
                    <span className="text-muted-foreground">{run && datasetTitle(options, run.spec.dataset)}</span>
                    {run?.spec.label && <span className="truncate text-muted-foreground">{run.spec.label}</span>}
                    <span className="ms-auto text-muted-foreground text-xs tabular-nums">
                      {percent(run?.status.test_accuracy, 2)}
                    </span>
                  </label>
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="text-muted-foreground text-xs">No finished runs yet.</p>
        )}
        <div className="mt-5 flex items-center gap-6">
          <div className="flex items-center gap-3">
            <span className="font-medium text-xs">Images</span>
            <Segmented
              label="Images to analyse"
              value={split}
              onChange={setSplit}
              options={[
                { value: "test", label: "Test" },
                { value: "val", label: "Validation" },
              ]}
            />
          </div>
          <label className="flex items-center gap-3">
            <span className="font-medium text-xs">Seed</span>
            <Input
              size="sm"
              value={seed}
              onChange={(event) => setSeed(event.target.value)}
              aria-invalid={validSeed ? undefined : true}
              className="w-20 text-[13px] sm:text-[13px]"
            />
          </label>
        </div>
        <h3 className="mt-5 mb-2 font-medium text-xs">Analyses</h3>
        <div className="grid grid-cols-2 gap-x-6 gap-y-0.5">
          {Object.entries(options.analyses).map(([name, item]) => (
            <label key={name} className="flex h-7 cursor-default items-center gap-2.5 text-[13px]" title={item.summary}>
              <Checkbox
                checked={analyses.includes(name)}
                onCheckedChange={(checked) =>
                  setAnalyses((previous) => (checked ? [...previous, name] : previous.filter((other) => other !== name)))
                }
              />
              <span className="truncate">{item.title}</span>
            </label>
          ))}
        </div>
      </div>
      {failure && (
        <div className="flex gap-2 border-t bg-destructive/4 px-5 py-3 text-destructive-foreground text-xs">
          <TriangleAlertIcon className="mt-px size-3.5 shrink-0" />
          <span className="whitespace-pre-wrap font-mono">{failure}</span>
        </div>
      )}
      <div className="flex items-center gap-3 border-t px-5 py-3">
        <span className="min-w-0 flex-1 text-muted-foreground text-xs">
          {picked.length} model{picked.length === 1 ? "" : "s"} · {analyses.length} analys{analyses.length === 1 ? "is" : "es"}
        </span>
        <Button type="button" variant="ghost" size="sm" onClick={onDone}>
          Cancel
        </Button>
        <Button type="submit" size="sm" disabled={!ready} loading={busy}>
          Build report
          <KbdGroup className="-me-1 ms-1 opacity-70">
            <Kbd className="bg-transparent text-current">Ctrl</Kbd>
            <Kbd className="bg-transparent text-current">↵</Kbd>
          </KbdGroup>
        </Button>
      </div>
    </form>
  );
}
