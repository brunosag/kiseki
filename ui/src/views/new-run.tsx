// The new-run composer: what to train (optimizer, dataset, model) and where as chips,
// then the optimizer's parameters and the training settings. Fields start at the
// defaults, the preset's values, or the run being repeated; only what differs is sent.
// Comma-separated values try each one: a grid of runs.
import { CpuIcon, DatabaseIcon, LayersIcon, ServerIcon, SparklesIcon, TriangleAlertIcon } from "lucide-react";
import { useRef, useState } from "react";
import type { RunDraft } from "@/app";
import { type Choice, Pick } from "@/components/pick";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsiblePanel, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Dialog, DialogPopup, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import { Switch } from "@/components/ui/switch";
import { toastManager } from "@/components/ui/toast";
import { Tooltip, TooltipPopup, TooltipTrigger } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import { words } from "@/lib/format";
import { addRuns, useLive } from "@/lib/live";
import { href, navigate } from "@/lib/router";
import { type Field, fields, parse, request, runCount, shown } from "@/lib/schema";
import { useStored } from "@/lib/stored";
import type { Json, Options, Run } from "@/lib/types";
import { cn } from "@/lib/utils";

/** Chosen with the chips, not typed. */
const CHIPS = ["dataset", "model", "optimizer", "params", "label"];
/** Shown up front; the other run settings fold under "More settings". */
const MAIN = ["iterations", "seed"];

export function NewRunDialog({ draft, onClose }: { draft: RunDraft | undefined; onClose: () => void }) {
  const { options } = useLive();
  // The popup animates out after `draft` is cleared: keep showing what was open.
  const last = useRef<{ draft: RunDraft; session: number }>({ draft: null, session: 0 });
  if (draft !== undefined && draft !== last.current.draft)
    last.current = { draft, session: last.current.session + 1 };
  return (
    <Dialog open={draft !== undefined} onOpenChange={(open) => !open && onClose()}>
      <DialogPopup className="max-w-2xl" showCloseButton={false}>
        {options && (
          <Composer key={last.current.session} draft={last.current.draft} options={options} onDone={onClose} />
        )}
      </DialogPopup>
    </Dialog>
  );
}

function Composer({ draft, options, onDone }: { draft: RunDraft; options: Options; onDone: () => void }) {
  const { machines } = useLive();
  const start = draft ?? {};
  const [dataset, setDataset] = useState(String(start.dataset ?? options.defaults.dataset));
  const [optimizer, setOptimizerKey] = useState(String(start.optimizer ?? options.defaults.optimizer));
  const [model, setModel] = useState(String(start.model ?? ""));
  const [machine, setMachine] = useStored("new-run.machine", "local");
  const [preset, setPreset] = useState("");
  const [values, setValues] = useState<Record<string, string>>(() => {
    const out: Record<string, string> = {};
    for (const [key, value] of Object.entries(start)) if (!CHIPS.includes(key) || key === "label") out[key] = shown(value);
    return out;
  });
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  const all = fields(options.run);
  const runFields = all.filter((field) => !CHIPS.includes(field.key));
  const labelField = all.filter((field) => field.key === "label");
  const paramFields = fields(options.optimizers[optimizer]?.params ?? {});
  const baseline: Record<string, Json> = {
    ...Object.fromEntries([...runFields, ...paramFields].map((field) => [field.key, field.default])),
    label: "",
    ...(preset ? options.presets[preset] : {}),
  };
  const known = [...runFields, ...paramFields, ...labelField].map((field) => ({ ...field, default: baseline[field.key] }));
  const text = (key: string) => values[key] ?? shown(baseline[key]);
  const set = (key: string, value: string) => setValues((previous) => ({ ...previous, [key]: value }));

  const errors: Record<string, string> = {};
  for (const field of known) {
    try {
      parse(text(field.key), field);
    } catch (error) {
      errors[field.key] = (error as Error).message;
    }
  }
  const valid = !Object.keys(errors).length;
  const asked = valid ? request(Object.fromEntries(known.map((field) => [field.key, text(field.key)])), known) : null;
  const runs = asked ? runCount(asked.grid) : 0;
  const chosen = machines?.find((candidate) => candidate.name === machine);
  const notice = options.machines.find((candidate) => candidate.name === machine)?.notice;

  const setOptimizer = (next: string) => {
    // Parameters mean different things to each optimizer: start the new one fresh.
    setValues((previous) =>
      Object.fromEntries(Object.entries(previous).filter(([key]) => !paramFields.some((field) => field.key === key))),
    );
    setOptimizerKey(next);
  };

  const submit = async () => {
    if (!asked || busy) return;
    setBusy(true);
    setFailure(null);
    try {
      const created = await api<Run[]>("runs", {
        overrides: { ...asked.overrides, dataset, optimizer, ...(model ? { model } : {}) },
        grid: asked.grid,
        preset: preset || null,
        machine,
      });
      addRuns(created);
      toastManager.add({
        type: "success",
        title: `Queued ${created.length} run${created.length > 1 ? "s" : ""} on ${machine}`,
      });
      onDone();
      navigate(created.length === 1 ? href.run(created[0].id) : "/runs");
    } catch (error) {
      setFailure((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const choices = {
    optimizer: Object.entries(options.optimizers).map(([value, item]) => ({ value, label: item.title })),
    dataset: Object.entries(options.datasets).map(([value, item]) => ({
      value,
      label: item.title,
      note: options.cached[value] ? undefined : "downloads first",
    })),
    model: [
      { value: "", label: options.datasets[dataset]?.default_model ?? "Default", note: "default" },
      ...options.models
        .filter((name) => name !== options.datasets[dataset]?.default_model)
        .map((name) => ({ value: name, label: name })),
    ],
    machine: options.machines.map((item) => ({ value: item.name, label: item.name, note: item.description })),
    preset: [
      { value: "", label: "No preset" },
      ...Object.keys(options.presets).map((name) => ({ value: name, label: `Preset: ${name}` })),
    ],
  } satisfies Record<string, Choice[]>;
  const advanced = runFields.filter((field) => !MAIN.includes(field.key));
  const changed = advanced.filter((field) => text(field.key).trim() !== shown(baseline[field.key])).length;
  const field = (item: Field) => (
    <FieldRow
      key={`${optimizer}/${item.key}`}
      field={item}
      text={text(item.key)}
      changed={text(item.key).trim() !== shown(baseline[item.key])}
      error={errors[item.key]}
      onChange={(value) => set(item.key, value)}
    />
  );

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
      <div className="px-5 pt-4">
        <DialogTitle className="font-medium font-sans text-muted-foreground text-xs">New run</DialogTitle>
        <input
          value={text("label")}
          onChange={(event) => set("label", event.target.value)}
          placeholder="Label (optional)"
          aria-label="Label"
          className="mt-3 w-full bg-transparent font-medium text-lg outline-none placeholder:text-muted-foreground/60"
        />
        <div className="mt-3 flex flex-wrap gap-1.5">
          <Pick label="Optimizer" icon={<SparklesIcon />} value={optimizer} choices={choices.optimizer} onChange={setOptimizer} />
          <Pick
            label="Dataset"
            icon={<DatabaseIcon />}
            value={dataset}
            choices={choices.dataset}
            onChange={(next) => {
              setDataset(next);
              setModel("");
            }}
          />
          <Pick label="Model" icon={<LayersIcon />} value={model} choices={choices.model} onChange={setModel} />
          <Pick label="Machine" icon={<ServerIcon />} value={machine} choices={choices.machine} onChange={setMachine} />
          {Object.keys(options.presets).length > 0 && (
            <Pick label="Preset" icon={<CpuIcon />} value={preset} choices={choices.preset} onChange={setPreset} />
          )}
        </div>
      </div>
      <div className="mt-4 min-h-0 overflow-y-auto border-t px-5 py-4">
        <Group title={options.optimizers[optimizer]?.title ?? optimizer} note={options.optimizers[optimizer]?.summary}>
          {paramFields.map(field)}
        </Group>
        <Group title="Training">
          {MAIN.map((key) => runFields.find((item) => item.key === key))
            .filter((item) => item !== undefined)
            .map(field)}
        </Group>
        <Collapsible>
          <CollapsibleTrigger className="mt-3 flex items-center gap-1.5 text-muted-foreground text-xs outline-none hover:text-foreground focus-visible:text-foreground">
            More settings
            {changed > 0 && <span className="text-foreground">· {changed} changed</span>}
          </CollapsibleTrigger>
          <CollapsiblePanel>
            <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 pt-3">{advanced.map(field)}</div>
          </CollapsiblePanel>
        </Collapsible>
      </div>
      {failure && (
        <div className="flex gap-2 border-t bg-destructive/4 px-5 py-3 text-destructive-foreground text-xs">
          <TriangleAlertIcon className="mt-px size-3.5 shrink-0" />
          <span className="whitespace-pre-wrap font-mono">{failure}</span>
        </div>
      )}
      <div className="flex items-center gap-3 border-t px-5 py-3">
        <div className="min-w-0 flex-1 text-muted-foreground text-xs">
          {chosen?.error || chosen?.gpu_error ? (
            <span className="flex items-center gap-1.5 text-warning-foreground">
              <TriangleAlertIcon className="size-3.5 shrink-0" />
              <span className="truncate">
                {machine}: {chosen.error ?? chosen.gpu_error}
              </span>
            </span>
          ) : notice ? (
            `Queuing here posts your name, email and expected dates to ${notice}.`
          ) : (
            "Commas try several values, one run each."
          )}
        </div>
        <Button type="button" variant="ghost" size="sm" onClick={onDone}>
          Cancel
        </Button>
        <Button type="submit" size="sm" disabled={!valid || busy} loading={busy}>
          {runs > 1 ? `Start ${runs} runs` : "Start run"}
          <KbdGroup className="-me-1 ms-1 opacity-70">
            <Kbd className="bg-transparent text-current">Ctrl</Kbd>
            <Kbd className="bg-transparent text-current">↵</Kbd>
          </KbdGroup>
        </Button>
      </div>
    </form>
  );
}

function Group({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <section className="mb-4">
      <h3 className="mb-2 flex items-baseline gap-2 text-xs">
        <span className="font-medium">{title}</span>
        {note && <span className="truncate text-muted-foreground">{note}</span>}
      </h3>
      <div className="grid grid-cols-2 gap-x-6 gap-y-1.5">{children}</div>
    </section>
  );
}

function FieldRow({
  field,
  text,
  changed,
  error,
  onChange,
}: {
  field: Field;
  text: string;
  changed: boolean;
  error?: string;
  onChange: (value: string) => void;
}) {
  const id = `field-${field.key}`;
  const label = (
    <label htmlFor={id} className={cn("truncate text-[13px]", changed ? "text-foreground" : "text-muted-foreground")}>
      {words(field.key)}
    </label>
  );
  return (
    <div className="grid grid-cols-[9rem_1fr] items-center gap-2">
      {field.description ? (
        <Tooltip>
          <TooltipTrigger render={<span className="min-w-0 truncate" />}>{label}</TooltipTrigger>
          <TooltipPopup>{field.description}</TooltipPopup>
        </Tooltip>
      ) : (
        label
      )}
      {field.kind === "boolean" ? (
        <Switch id={id} checked={text === "true"} onCheckedChange={(checked) => onChange(String(checked))} />
      ) : field.kind === "choice" ? (
        <Pick
          label={words(field.key)}
          value={text}
          choices={field.choices.map((choice) => ({ value: String(choice), label: String(choice) }))}
          onChange={onChange}
          className="w-full justify-start"
        />
      ) : (
        <Input
          id={id}
          size="sm"
          value={text}
          onChange={(event) => onChange(event.target.value)}
          placeholder={field.nullable ? "auto" : undefined}
          aria-invalid={error ? true : undefined}
          title={error}
          className={cn("text-[13px] sm:text-[13px]", changed && "font-medium")}
          spellCheck={false}
        />
      )}
    </div>
  );
}
