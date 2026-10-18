import { useEffect, useMemo, useState } from "react";
import { api, runFileURL, useResource, useRows, useViewTiming } from "../api";
import { duration, number, percent, runLabel, colors } from "../core";
import type { Run, Value } from "../types";
import {
  active,
  Badge,
  Empty,
  ErrorBox,
  Loading,
  PageHeader,
  Stat,
} from "../components/Common";
import { Curve } from "../components/Curve";
export function RunDetail({
  id,
  statusVersion,
  navigate,
  select,
}: {
  id: string;
  statusVersion: number;
  navigate: (path: string) => void;
  select: (id: string) => void;
}) {
  const [refresh, setRefresh] = useState(0),
    [x, setX] = useState("step"),
    [log, setLog] = useState(false),
    [tab, setTab] = useState("curves"),
    [message, setMessage] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const { data: run, error: loadError } = useResource<Run>(
      `runs/${id}`,
      statusVersion + refresh,
    ),
    live = active(run?.status.state ?? "");
  const val = useRows(id, "val", live),
    train = useRows(id, "train", live);
  useViewTiming("run-detail", !!run && !!val.data && !!train.data);
  const accuracy = useMemo(
    () => [
      {
        label: "Validation accuracy",
        x: val.data?.columns[x] ?? [],
        y: val.data?.columns.val_accuracy ?? [],
        color: colors[0],
      },
    ],
    [val.data, x],
  );
  const losses = useMemo(
    () => [
      {
        label: "Training loss",
        x: train.data?.columns[x] ?? [],
        y: train.data?.columns.loss ?? [],
        color: colors[1],
      },
      {
        label: "Validation loss",
        x: val.data?.columns[x] ?? [],
        y: val.data?.columns.val_loss ?? [],
        color: colors[2],
      },
    ],
    [val.data, train.data, x],
  );
  async function control(action: string) {
    setBusy(true);
    setError("");
    try {
      const result = await api<{ result: string }>(`runs/${id}/${action}`, {});
      setMessage(result.result);
      setRefresh((r) => r + 1);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  if (loadError) return <ErrorBox error={loadError} />;
  if (!run) return <Loading />;
  return (
    <>
      <PageHeader
        eyebrow="RUNS / RUN"
        title={runLabel(run)}
        actions={
          <>
            <Badge state={run.status.state} />
            {active(run.status.state) && (
              <>
                <button disabled={busy} onClick={() => control("pause")}>
                  Ⅱ Pause
                </button>
                <button disabled={busy} onClick={() => control("stop")}>
                  Stop
                </button>
              </>
            )}
            {["paused", "stopped", "failed"].includes(run.status.state) && (
              <button
                title={
                  run.status.machine !== "local"
                    ? `Requeues on ${run.status.machine}`
                    : undefined
                }
                disabled={busy}
                onClick={() => control("resume")}
              >
                ▶ Resume
              </button>
            )}
            <button
              onClick={() => {
                select(id);
                navigate("/compare");
              }}
            >
              Compare ↗
            </button>
          </>
        }
      >
        <span className="mono">{id}</span>
      </PageHeader>
      <ErrorBox error={error || run.status.error || val.error || train.error} />
      {message && (
        <div className="notice" role="status">
          {message}
        </div>
      )}
      <div className="stats">
        <Stat
          label="Best validation"
          value={percent(run.status.best_accuracy)}
          detail={`at step ${number(run.status.best_step)}`}
        />
        <Stat
          label="Test accuracy"
          value={percent(run.status.test_accuracy)}
          detail="held-out test set"
        />
        <Stat
          label="Progress"
          value={`${number(run.status.step)} / ${number(run.spec.iterations)}`}
          detail={`${number(run.status.steps_per_second)} steps/s · ETA ${duration(run.status.eta_seconds)}`}
        />
        <Stat
          label="Elapsed"
          value={duration(run.status.elapsed_seconds)}
          detail={`${run.status.machine} · ${run.status.device ?? "device pending"}`}
        />
      </div>
      <div className="tabs">
        {["curves", "configuration", "events", "models", "log"].map((t) => (
          <button
            key={t}
            className={tab === t ? "current" : ""}
            onClick={() => setTab(t)}
          >
            {t}
            {t === "models"
              ? ` (${run.models?.length ?? 0})`
              : t === "events"
                ? ` (${run.events?.length ?? 0})`
                : ""}
          </button>
        ))}
      </div>
      {tab === "curves" && (
        <>
          <div className="chart-controls">
            <label>
              X axis{" "}
              <select
                aria-label="Curve x axis"
                value={x}
                onChange={(e) => setX(e.target.value)}
              >
                <option value="step">Step</option>
                <option value="elapsed">Elapsed seconds</option>
              </select>
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={log}
                onChange={(e) => setLog(e.target.checked)}
              />
              Log x
            </label>
            <span className="muted">Drag to zoom · double-click to reset</span>
          </div>
          <div className="panel padded">
            <h2>Validation accuracy</h2>
            <Curve
              lines={accuracy}
              accuracy
              log={log}
              xLabel={x === "step" ? "Step" : "Elapsed seconds"}
            />
          </div>
          <div className="panel padded mt">
            <h2>Loss</h2>
            {x === "elapsed" && !train.data?.columns.elapsed && (
              <p className="muted">
                Training rows have no elapsed timestamps; only validation loss
                is shown on this axis.
              </p>
            )}
            <Curve
              lines={losses}
              log={log}
              xLabel={x === "step" ? "Step" : "Elapsed seconds"}
            />
          </div>
        </>
      )}
      {tab === "configuration" && (
        <section className="panel padded">
          <h2>Frozen run specification</h2>
          <ValueTable value={run.spec} />
          <p className="mono muted path">{run.path}</p>
          {run.summary && (
            <>
              <h2>Summary</h2>
              <ValueTable value={run.summary} />
            </>
          )}
        </section>
      )}
      {tab === "events" && (
        <section className="panel">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Event</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {run.events?.map((e, i) => (
                <tr key={i}>
                  <td className="mono">{String(e.time)}</td>
                  <td>{String(e.kind)}</td>
                  <td className="mono">
                    {Object.entries(e)
                      .filter(([k]) => !["time", "kind"].includes(k))
                      .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
                      .join(" · ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!run.events?.length && <Empty title="No events yet" />}
        </section>
      )}
      {tab === "models" && (
        <section className="panel padded">
          <h2>Saved networks</h2>
          <p className="muted">
            Choose a saved network when creating a report.
          </p>
          {run.models?.map((name) => (
            <div className="model-row" key={name}>
              <span className="artifact-symbol">◇</span>
              <div>
                <strong>{name}</strong>
                <small className="block mono">
                  {id}:{name}
                </small>
              </div>
              <a
                className="button"
                href={runFileURL(id, `models/${name}.safetensors`)}
                download={`${id}-${name}.safetensors`}
              >
                Download
              </a>
              <button
                onClick={() => {
                  select(id);
                  navigate("/reports/new");
                }}
              >
                Analyse ↗
              </button>
            </div>
          ))}
          {!run.models?.length && (
            <Empty title="No saved models yet">
              A best or final network will appear after training writes it.
            </Empty>
          )}
        </section>
      )}
      {tab === "log" && (
        <WorkerLog id={id} version={statusVersion} remote={run.status.machine !== "local"} />
      )}
    </>
  );
}
const LOG_LINES = 400;
function WorkerLog({ id, version, remote }: { id: string; version: number; remote: boolean }) {
  const [text, setText] = useState<string | null>(null),
    [missing, setMissing] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    fetch(runFileURL(id, "worker.log"), { signal: controller.signal })
      .then(async (res) => {
        if (res.status === 404) return setMissing(true);
        setMissing(false);
        setText(await res.text());
      })
      .catch(() => {});
    return () => controller.abort();
  }, [id, version]);
  const lines = text?.split("\n") ?? [];
  return (
    <section className="panel padded">
      <h2>Worker log</h2>
      {missing ? (
        <Empty title="No worker log here">
          {remote
            ? "Remote runs keep their log on their machine until it is pulled."
            : "The worker has not started yet."}
        </Empty>
      ) : text === null ? (
        <Loading />
      ) : (
        <>
          {lines.length > LOG_LINES && (
            <p className="muted">
              Last {LOG_LINES} of {lines.length} lines.{" "}
              <a href={runFileURL(id, "worker.log")} target="_blank" rel="noreferrer">
                Whole log
              </a>
            </p>
          )}
          <pre className="log mono">{lines.slice(-LOG_LINES).join("\n")}</pre>
        </>
      )}
    </section>
  );
}
function ValueTable({ value }: { value: Record<string, Value> }) {
  return (
    <dl className="spec-table">
      {Object.entries(value).map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd className="mono">
            {typeof value === "object" ? JSON.stringify(value) : String(value)}
          </dd>
        </div>
      ))}
    </dl>
  );
}
