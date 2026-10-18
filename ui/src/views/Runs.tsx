import { useMemo, useState } from "react";
import type { Run } from "../types";
import { duration, number, percent, runLabel } from "../core";
import { useViewTiming } from "../api";
import { Badge, Empty, PageHeader, Stat } from "../components/Common";
export function Runs({
  runs,
  selected,
  toggle,
  selectMany,
  navigate,
}: {
  runs: Run[];
  selected: Set<string>;
  toggle: (id: string) => void;
  selectMany: (ids: string[], checked: boolean) => void;
  navigate: (path: string) => void;
}) {
  const [search, setSearch] = useState(""),
    [state, setState] = useState("all"),
    [optimizer, setOptimizer] = useState("all"),
    [dataset, setDataset] = useState("all");
  const [sort, setSort] = useState("newest"),
    [limit, setLimit] = useState(100);
  const visible = useMemo(
    () =>
      runs
        .filter(
          (r) =>
            (state === "all" || r.status.state === state) &&
            (optimizer === "all" || r.spec.optimizer === optimizer) &&
            (dataset === "all" || r.spec.dataset === dataset) &&
            `${r.id} ${r.spec.label} ${r.spec.model} ${r.status.machine}`
              .toLowerCase()
              .includes(search.toLowerCase()),
        )
        .sort((a, b) =>
          sort === "accuracy"
            ? (b.status.best_accuracy ?? -1) - (a.status.best_accuracy ?? -1)
            : b.id.localeCompare(a.id),
        ),
    [runs, state, optimizer, dataset, search, sort],
  );
  const shown = visible.slice(0, limit),
    all = shown.length > 0 && shown.every((r) => selected.has(r.id));
  const best = runs.reduce(
    (v, r) => Math.max(v, r.status.best_accuracy ?? 0),
    0,
  );
  useViewTiming("runs", true);
  return (
    <>
      <PageHeader
        eyebrow="EXPERIMENT WORKSPACE"
        title="Runs"
        actions={
          <button className="primary" onClick={() => navigate("/new")}>
            ＋ New run
          </button>
        }
      >
        Every optimizer. One place to follow the evidence.
      </PageHeader>
      <div className="stats">
        <Stat label="Total runs" value={runs.length} detail="in this store" />
        <Stat
          label="In progress"
          value={
            runs.filter((r) => ["running", "queued"].includes(r.status.state))
              .length
          }
          detail="running or queued"
        />
        <Stat
          label="Completed"
          value={runs.filter((r) => r.status.state === "completed").length}
          detail="ready to analyse"
        />
        <Stat
          label="Best validation"
          value={best ? percent(best) : "—"}
          detail="across all runs"
        />
      </div>
      <div className="panel">
        <div className="toolbar">
          <input
            aria-label="Search runs"
            className="search"
            placeholder="Search runs, labels, machines…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select
            aria-label="Filter status"
            value={state}
            onChange={(e) => setState(e.target.value)}
          >
            {[
              "all",
              "running",
              "queued",
              "paused",
              "completed",
              "stopped",
              "failed",
            ].map((s) => (
              <option value={s} key={s}>
                {s === "all" ? "All statuses" : s}
              </option>
            ))}
          </select>
          <select
            aria-label="Filter optimizer"
            value={optimizer}
            onChange={(e) => setOptimizer(e.target.value)}
          >
            <option value="all">All optimizers</option>
            {[...new Set(runs.map((r) => r.spec.optimizer))].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
          <select
            aria-label="Filter dataset"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
          >
            <option value="all">All datasets</option>
            {[...new Set(runs.map((r) => r.spec.dataset))].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
          <select
            aria-label="Sort runs"
            value={sort}
            onChange={(e) => setSort(e.target.value)}
          >
            <option value="newest">Newest first</option>
            <option value="accuracy">Best accuracy</option>
          </select>
        </div>
        <div className="table-wrap">
          <table className="run-table">
            <thead>
              <tr>
                <th>
                  <input
                    type="checkbox"
                    aria-label="Select visible runs"
                    checked={all}
                    onChange={(e) =>
                      selectMany(
                        shown.map((r) => r.id),
                        e.target.checked,
                      )
                    }
                  />
                </th>
                <th>Run</th>
                <th>Status</th>
                <th>Optimizer / model</th>
                <th>Progress</th>
                <th className="numeric">Best validation</th>
                <th className="numeric">Elapsed</th>
                <th>Machine</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.id} className={selected.has(r.id) ? "selected" : ""}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`Select ${r.id}`}
                      checked={selected.has(r.id)}
                      onChange={() => toggle(r.id)}
                    />
                  </td>
                  <td>
                    <button
                      className="text-link run-link"
                      onClick={() => navigate(`/runs/${r.id}`)}
                    >
                      {runLabel(r)}
                    </button>
                    <small className="block mono">{r.id}</small>
                    <span className="dataset-tag">{r.spec.dataset}</span>
                  </td>
                  <td>
                    <Badge state={r.status.state} />
                  </td>
                  <td>
                    {r.spec.optimizer}
                    <small className="block">
                      {r.spec.model} · seed {r.spec.seed}
                    </small>
                  </td>
                  <td>
                    <span className="mono">
                      {number(r.status.step)}{" "}
                      <span className="muted">
                        / {number(r.spec.iterations)}
                      </span>
                    </span>
                    <div className="progress">
                      <i
                        style={{
                          width: `${Math.min(100, (100 * (r.status.step ?? 0)) / r.spec.iterations)}%`,
                        }}
                      />
                    </div>
                  </td>
                  <td className="numeric accent">
                    {percent(r.status.best_accuracy)}
                  </td>
                  <td className="numeric mono">
                    {duration(r.status.elapsed_seconds)}
                  </td>
                  <td>
                    <span className="machine-dot" />
                    {r.status.machine}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!shown.length && (
          <Empty title="No matching runs">
            Queue a run or adjust your filters.
          </Empty>
        )}
        <footer className="table-footer">
          <span>
            {visible.length} runs · {selected.size} selected
          </span>
          {visible.length > limit && (
            <button onClick={() => setLimit((l) => l + 100)}>
              Show 100 more
            </button>
          )}
          <span>Live status updates</span>
        </footer>
      </div>
      {selected.size > 0 && (
        <div className="selection-bar">
          <span>
            <strong>{selected.size}</strong> runs selected
          </span>
          <button onClick={() => selectMany([...selected], false)}>
            Clear
          </button>
          <button onClick={() => navigate("/compare")}>Compare curves ↗</button>
          <button className="primary" onClick={() => navigate("/reports/new")}>
            Create report ↗
          </button>
        </div>
      )}
    </>
  );
}
