import { useMemo, useState } from "react";
import type { Options, Run, Schema } from "../types";
import { defaultFields, properties, runRequest } from "../core";
import { api, useViewTiming } from "../api";
import { ErrorBox, PageHeader } from "../components/Common";
import { SchemaFields } from "../components/SchemaFields";
export function NewRun({
  options,
  created,
}: {
  options: Options;
  created: (runs: Run[]) => void;
}) {
  const [optimizer, setOptimizer] = useState(options.defaults.optimizer),
    [dataset, setDataset] = useState(options.defaults.dataset),
    [model, setModel] = useState(""),
    [machine, setMachine] = useState("local");
  const [preset, setPreset] = useState(""),
    [runFields, setRunFields] = useState(() => defaultFields(options.run)),
    [params, setParams] = useState(() =>
      defaultFields(options.optimizers[optimizer].params),
    );
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const target = options.machines.find((m) => m.name === machine);
  const schema = Object.fromEntries(
    Object.entries(properties(options.run)).filter(
      ([key]) => !["dataset", "model", "optimizer", "params"].includes(key),
    ),
  );
  const paramSchema = properties(options.optimizers[optimizer].params);
  const combined: Record<string, Schema> = {
    ...schema,
    ...Object.fromEntries(
      Object.entries(paramSchema).map(([k, s]) => [`params.${k}`, s]),
    ),
  };
  const fields = {
    ...runFields,
    ...Object.fromEntries(
      Object.entries(params).map(([k, v]) => [`params.${k}`, v]),
    ),
  };
  const preview = useMemo(() => {
    try {
      return { request: runRequest(fields, combined), error: "" };
    } catch (e) {
      return { request: null, error: String((e as Error).message) };
    }
  }, [JSON.stringify(fields), optimizer]);
  function applyPreset(name: string, opt = optimizer) {
    const base = defaultFields(options.run),
      pars = defaultFields(options.optimizers[opt].params);
    if (name)
      for (const [key, value] of Object.entries(options.presets[name])) {
        if (key in pars) pars[key] = String(value);
        else if (key in base) base[key] = String(value);
      }
    setRunFields(base);
    setParams(pars);
    setPreset(name);
  }
  async function submit(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const request = runRequest(fields, combined);
      const result = await api<Run[]>("runs", {
        overrides: { ...request.overrides, dataset, model, optimizer },
        grid: request.grid,
        preset: preset || null,
        machine,
      });
      created(result);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  useViewTiming("new-run", true);
  return (
    <>
      <PageHeader eyebrow="RUNS / NEW" title="New run">
        Start with a baseline. Change one thing. Follow what happens.
      </PageHeader>
      <form onSubmit={submit}>
        <ErrorBox error={error} />
        <div className="two-column">
          <div className="stack">
            <section className="panel padded">
              <div className="section-heading">
                <span className="section-number">01</span>
                <h2>What to train</h2>
              </div>
              <div className="fields">
                <label className="field">
                  <span>Dataset</span>
                  <select
                    value={dataset}
                    onChange={(e) => {
                      setDataset(e.target.value);
                      setModel("");
                    }}
                  >
                    {Object.entries(options.datasets).map(([name, d]) => (
                      <option key={name} value={name}>
                        {d.title}
                        {options.cached[name] ? " · cached" : ""}
                      </option>
                    ))}
                  </select>
                  <small>
                    {options.datasets[dataset].image_shape.join(" × ")} ·{" "}
                    {options.datasets[dataset].class_names.length} classes
                  </small>
                </label>
                <label className="field">
                  <span>Model</span>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                  >
                    <option value="">
                      Dataset default ·{" "}
                      {options.datasets[dataset].default_model}
                    </option>
                    {options.models.map((m) => (
                      <option key={m}>{m}</option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>Optimizer</span>
                  <select
                    value={optimizer}
                    onChange={(e) => {
                      setOptimizer(e.target.value);
                      setParams(
                        defaultFields(
                          options.optimizers[e.target.value].params,
                        ),
                      );
                      if (preset) applyPreset(preset, e.target.value);
                    }}
                  >
                    {Object.keys(options.optimizers).map((o) => (
                      <option key={o}>{o}</option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>Preset</span>
                  <select
                    value={preset}
                    onChange={(e) => applyPreset(e.target.value)}
                  >
                    <option value="">No preset</option>
                    {Object.keys(options.presets).map((p) => (
                      <option key={p}>{p}</option>
                    ))}
                  </select>
                  <small>
                    Applying a preset resets run and optimizer fields.
                  </small>
                </label>
              </div>
            </section>
            <section className="panel padded">
              <div className="section-heading">
                <span className="section-number">02</span>
                <h2>{optimizer} parameters</h2>
              </div>
              <p className="muted">{options.optimizers[optimizer].summary}</p>
              <SchemaFields
                schemas={paramSchema}
                values={params}
                onChange={(k, v) => setParams((p) => ({ ...p, [k]: v }))}
                grid
              />
            </section>
            <section className="panel padded">
              <div className="section-heading">
                <span className="section-number">03</span>
                <h2>Training & reproducibility</h2>
              </div>
              <SchemaFields
                schemas={schema}
                values={runFields}
                onChange={(k, v) => setRunFields((p) => ({ ...p, [k]: v }))}
                grid
              />
            </section>
          </div>
          <aside className="stack sticky">
            <section className="panel padded">
              <div className="eyebrow">LAUNCH SUMMARY</div>
              <h2 className="large-number">
                {preview.request?.count ?? "—"}{" "}
                <span>run{preview.request?.count === 1 ? "" : "s"}</span>
              </h2>
              <p className="muted">
                Comma-separated numeric values create one run for every
                combination.
              </p>
              <dl className="key-values">
                <dt>Dataset</dt>
                <dd>{options.datasets[dataset].title}</dd>
                <dt>Model</dt>
                <dd>{model || options.datasets[dataset].default_model}</dd>
                <dt>Optimizer</dt>
                <dd>{optimizer}</dd>
              </dl>
              <label className="field">
                <span>Target machine</span>
                <select
                  value={machine}
                  onChange={(e) => setMachine(e.target.value)}
                >
                  {options.machines.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                      {m.description ? ` · ${m.description}` : ""}
                    </option>
                  ))}
                </select>
                <small>
                  {target?.ssh
                    ? `Over SSH: deploys the code and makes the dataset available on ${target.name} first (minutes the first time), then queues there.`
                    : "Queues on this laptop."}
                  {target?.notice &&
                    ` While runs are active, ${target.notice} names you with a forecast end, as ${target.name} requires.`}
                </small>
              </label>
              <ErrorBox error={preview.error} />
              <button
                className="primary full"
                disabled={busy || !preview.request}
                type="submit"
              >
                {busy
                  ? target?.ssh
                    ? `Deploying to ${target.name}…`
                    : "Queuing…"
                  : `Queue ${preview.request?.count ?? ""} run${preview.request?.count === 1 ? "" : "s"} →`}
              </button>
            </section>
            <div className="note">
              Specs and bounds come directly from the backend registry. Every
              queued run stores its complete configuration.
            </div>
          </aside>
        </div>
      </form>
    </>
  );
}
