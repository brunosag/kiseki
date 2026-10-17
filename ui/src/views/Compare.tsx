import { useMemo, useState } from "react";
import { api, useRows, useViewTiming } from "../api";
import {
  colors,
  duration,
  firstReached,
  number,
  percent,
  runLabel,
} from "../core";
import type { Columns, Run } from "../types";
import { useEffect } from "react";
import {
  active,
  Badge,
  Empty,
  ErrorBox,
  PageHeader,
} from "../components/Common";
import { Curve, type Line } from "../components/Curve";
// Each selected run owns an incremental subscription; a stable callback merges its columns.
function Source({
  run,
  onData,
  onError,
}: {
  run: Run;
  onData: (id: string, data: Columns) => void;
  onError: (error: string) => void;
}) {
  const { data, error } = useRows(run.id, "val", active(run.status.state));
  useEffect(() => {
    if (data) onData(run.id, data.columns);
  }, [run.id, data, onData]);
  useEffect(() => {
    if (error) onError(`${run.id}: ${error}`);
  }, [run.id, error, onError]);
  return null;
}
export function Compare({
  runs,
  navigate,
}: {
  runs: Run[];
  navigate: (path: string) => void;
}) {
  const [x, setX] = useState("elapsed"),
    [log, setLog] = useState(false),
    [metric, setMetric] = useState("accuracy"),
    [thresholds, setThresholds] = useState("80, 90, 95, 98");
  const [columns, setColumns] = useState<Record<string, Columns>>({}),
    [error, setError] = useState("");
  const onData = useMemo(
    () => (id: string, data: Columns) =>
      setColumns((p) => ({ ...p, [id]: data })),
    [],
  );
  const lines: Line[] = useMemo(
    () =>
      runs.map((r, i) => ({
        label: runLabel(r),
        x: columns[r.id]?.[x] ?? [],
        y: columns[r.id]?.[`val_${metric}`] ?? [],
        color: colors[i % colors.length],
      })),
    [runs, columns, x, metric],
  );
  const parsed = thresholds
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
      .map(Number),
    valid =
      parsed.length > 0 &&
      parsed.every((n) => Number.isFinite(n) && n > 0 && n <= 100),
    targets = valid ? parsed.slice(0, 12) : [];
  useViewTiming(
    "compare",
    runs.every((r) => !!columns[r.id]),
  );
  async function exportRows() {
    try {
      const rows = await Promise.all(
        runs.map((r) => api(`runs/${r.id}/rows/val`)),
      );
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" }),
      );
      const a = document.createElement("a");
      a.href = url;
      a.download = "kiseki-comparison.json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setError(String(e));
    }
  }
  if (!runs.length)
    return (
      <>
        <PageHeader eyebrow="RUNS / COMPARE" title="Compare runs" />
        <Empty title="Choose runs to compare">
          Select runs in the Runs table, then open Compare.
          <br />
          <button className="primary mt" onClick={() => navigate("/runs")}>
            Choose runs →
          </button>
        </Empty>
      </>
    );
  return (
    <>
      <PageHeader
        eyebrow="RUNS / COMPARE"
        title="Follow the learning"
        actions={
          <>
            <button onClick={exportRows}>Export samples ↓</button>
            <button
              className="primary"
              onClick={() => navigate("/reports/new")}
            >
              Analyse models ↗
            </button>
          </>
        }
      >
        {runs.length} selected runs · validation curves share one axis.
      </PageHeader>
      <ErrorBox error={error} />
      {runs.map((r) => (
        <Source key={r.id} run={r} onData={onData} onError={setError} />
      ))}
      {new Set(runs.map((r) => r.spec.dataset)).size > 1 && (
        <div className="notice">
          These runs use different datasets. Accuracy values measure different
          tasks.
        </div>
      )}
      <div className="panel padded">
        <div className="chart-controls">
          <select
            aria-label="Compare metric"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
          >
            <option value="accuracy">Validation accuracy</option>
            <option value="loss">Validation loss</option>
          </select>
          <select
            aria-label="Compare x axis"
            value={x}
            onChange={(e) => setX(e.target.value)}
          >
            <option value="elapsed">Elapsed seconds</option>
            <option value="step">Step</option>
          </select>
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
        <Curve
          lines={lines}
          accuracy={metric === "accuracy"}
          log={log}
          xLabel={x === "elapsed" ? "Elapsed seconds" : "Step"}
          height={390}
        />
      </div>
      <section className="panel mt">
        <div className="panel-heading">
          <div>
            <h2>Time to accuracy</h2>
            <p className="muted">
              First observed validation sample at or above each threshold.
            </p>
          </div>
          <label>
            Thresholds (%){" "}
            <input
              aria-label="Accuracy thresholds"
              value={thresholds}
              onChange={(e) => setThresholds(e.target.value)}
            />
          </label>
        </div>
        {!valid && (
          <ErrorBox error="Enter comma-separated percentages greater than 0 and at most 100." />
        )}
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Best validation</th>
                {targets.map((t, i) => (
                  <th key={i} className="numeric">
                    {t}%
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((r, i) => (
                <tr key={r.id}>
                  <td>
                    <span
                      className="line-dot"
                      style={{ background: colors[i % colors.length] }}
                    />
                    {runLabel(r)}
                    <small className="block mono">{r.id}</small>
                  </td>
                  <td>
                    <Badge state={r.status.state} />
                  </td>
                  <td className="accent">{percent(r.status.best_accuracy)}</td>
                  {targets.map((t, i) => {
                    const reached =
                      columns[r.id] && firstReached(columns[r.id], t / 100);
                    return (
                      <td key={i} className="numeric mono">
                        {reached ? (
                          <>
                            {duration(reached.elapsed)}
                            <small className="block">
                              step {number(reached.step)}
                            </small>
                          </>
                        ) : columns[r.id] ? (
                          "Not reached"
                        ) : (
                          "Loading…"
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
