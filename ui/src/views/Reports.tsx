import { Fragment, lazy, Suspense, useEffect, useState } from "react";
import type {
  Options,
  ReportRecord,
  ReportSpec,
  Run,
  Schema,
  Value,
} from "../types";
import { api, fileURL, useResource, useViewTiming } from "../api";
import { defaultFields, parseField, properties, runLabel } from "../core";
import {
  Badge,
  Empty,
  ErrorBox,
  Loading,
  PageHeader,
} from "../components/Common";
import { SchemaFields } from "../components/SchemaFields";
const Embedding = lazy(() =>
  import("../components/InteractiveReport").then((m) => ({
    default: m.Embedding,
  })),
);
const LrpGallery = lazy(() =>
  import("../components/InteractiveReport").then((m) => ({
    default: m.LrpGallery,
  })),
);
export function Reports({
  reports,
  navigate,
}: {
  reports: ReportRecord[];
  navigate: (path: string) => void;
}) {
  useViewTiming("reports", true);
  return (
    <>
      <PageHeader
        eyebrow="ANALYSIS WORKSPACE"
        title="Reports"
        actions={
          <button className="primary" onClick={() => navigate("/reports/new")}>
            ＋ Create report
          </button>
        }
      >
        Look beyond accuracy. Understand what the networks learned.
      </PageHeader>
      {!reports.length ? (
        <Empty title="No reports yet">
          Choose saved models from your runs and build an analysis.
        </Empty>
      ) : (
        <div className="report-cards">
          {[...reports].reverse().map((r) => (
            <button
              className="report-card panel"
              key={r.id}
              onClick={() => navigate(`/reports/${r.id}`)}
            >
              <div className="flex justify-between items-center">
                <span className="artifact-symbol">▧</span>
                <Badge state={r.status.state} />
              </div>
              <h2>
                {r.spec.models.length} networks · {r.spec.split} split
              </h2>
              <p className="mono muted">{r.id}</p>
              <div className="chips">
                {Object.keys(r.spec.analyses).map((a) => (
                  <span key={a}>{a.replaceAll("_", " ")}</span>
                ))}
              </div>
              <footer>
                <span>{r.status.stage || "Open report"}</span>
                <span>↗</span>
              </footer>
            </button>
          ))}
        </div>
      )}
    </>
  );
}
export function NewReport({
  runs,
  options,
  created,
  navigate,
}: {
  runs: Run[];
  options: Options;
  created: (r: ReportRecord) => void;
  navigate: (path: string) => void;
}) {
  const [details, setDetails] = useState<Run[]>([]),
    [chosen, setChosen] = useState<Set<string>>(new Set()),
    [analyses, setAnalyses] = useState<Set<string>>(
      new Set(["metrics", "calibration", "agreement", "weights"]),
    ),
    [params, setParams] = useState<Record<string, Record<string, string>>>(() =>
      Object.fromEntries(
        Object.entries(options.analyses).map(([name, a]) => [
          name,
          defaultFields(a.params),
        ]),
      ),
    ),
    [split, setSplit] = useState<"test" | "val">("test"),
    [seed, setSeed] = useState("0"),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [loaded, setLoaded] = useState(false);
  const ids = runs.map((r) => r.id).join("|");
  useEffect(() => {
    let live = true;
    setLoaded(false);
    Promise.all(runs.map((r) => api<Run>(`runs/${r.id}`))).then(
      (list) => {
        if (live) {
          setDetails(list);
          setChosen(
            new Set(
              list.flatMap((r) =>
                r.models?.length
                  ? [
                      `${r.id}:${r.models.includes("best") ? "best" : r.models[0]}`,
                    ]
                  : [],
              ),
            ),
          );
          setLoaded(true);
        }
      },
      (e) => {
        if (live) setError(String(e));
      },
    );
    return () => {
      live = false;
    };
  }, [ids]);
  useViewTiming("new-report", loaded);
  function toggle(
    set: React.Dispatch<React.SetStateAction<Set<string>>>,
    key: string,
  ) {
    set((previous) => {
      const next = new Set(previous);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }
  async function submit(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (!chosen.size) throw new Error("Choose at least one saved model.");
      if (!analyses.size) throw new Error("Choose at least one analysis.");
      const datasets = new Set(
        details
          .filter((r) => [...chosen].some((ref) => ref.startsWith(r.id + ":")))
          .map((r) => r.spec.dataset),
      );
      if (datasets.size !== 1)
        throw new Error("All models must share one dataset.");
      if (!Number.isInteger(Number(seed)))
        throw new Error("Seed must be an integer.");
      const values: ReportSpec["analyses"] = {};
      for (const name of analyses) {
        values[name] = {};
        for (const [key, schema] of Object.entries(
          properties(options.analyses[name].params),
        ))
          values[name][key] = parseField(params[name][key], schema)[0];
      }
      created(
        await api<ReportRecord>("reports", {
          models: [...chosen],
          analyses: values,
          split,
          seed: Number(seed),
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  if (!runs.length)
    return (
      <>
        <PageHeader eyebrow="ANALYSIS / CREATE" title="Create report" />
        <Empty title="Select runs first">
          Pick runs in the Runs table to choose their saved models.
          <br />
          <button className="primary mt" onClick={() => navigate("/runs")}>
            Choose runs →
          </button>
        </Empty>
      </>
    );
  return (
    <>
      <PageHeader eyebrow="ANALYSIS / CONFIGURE" title="Create report">
        Compare trained networks on one dataset. Analysis runs locally.
      </PageHeader>
      <ErrorBox error={error} />
      <form onSubmit={submit}>
        <div className="two-column">
          <div className="stack">
            <section className="panel padded">
              <h2>Saved models</h2>
              {!loaded && !error && <Loading />}
              {details.map((r) => (
                <div className="model-choice" key={r.id}>
                  <h3>
                    {runLabel(r)} <small>{r.spec.dataset}</small>
                  </h3>
                  <div className="chips">
                    {r.models?.map((m) => (
                      <label className="checkbox" key={m}>
                        <input
                          type="checkbox"
                          checked={chosen.has(`${r.id}:${m}`)}
                          onChange={() => toggle(setChosen, `${r.id}:${m}`)}
                        />
                        {m}
                      </label>
                    ))}
                  </div>
                  {!r.models?.length && (
                    <small className="muted">No saved models yet.</small>
                  )}
                </div>
              ))}
            </section>
            <section className="panel padded">
              <h2>Analyses</h2>
              <p className="muted">
                Expand a selected analysis to adjust its parameters.
              </p>
              {Object.entries(options.analyses).map(([name, a]) => (
                <div key={name} className="analysis-choice">
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={analyses.has(name)}
                      onChange={() => toggle(setAnalyses, name)}
                    />
                    <strong>{a.title || name}</strong>
                  </label>
                  <p>{a.summary}</p>
                  {analyses.has(name) &&
                    Object.keys(properties(a.params)).length > 0 && (
                      <details>
                        <summary>Parameters</summary>
                        <SchemaFields
                          schemas={properties(a.params)}
                          values={params[name]}
                          onChange={(k, v) =>
                            setParams((p) => ({
                              ...p,
                              [name]: { ...p[name], [k]: v },
                            }))
                          }
                        />
                      </details>
                    )}
                </div>
              ))}
            </section>
          </div>
          <aside className="panel padded sticky">
            <div className="eyebrow">REPORT SUMMARY</div>
            <h2 className="large-number">
              {chosen.size} <span>models</span>
            </h2>
            <p className="muted">{analyses.size} analyses selected</p>
            <label className="field">
              <span>Evaluation split</span>
              <select
                value={split}
                onChange={(e) => setSplit(e.target.value as "test" | "val")}
              >
                <option value="test">Held-out test set</option>
                <option value="val">Validation set</option>
              </select>
            </label>
            <label className="field">
              <span>Random seed</span>
              <input value={seed} onChange={(e) => setSeed(e.target.value)} />
            </label>
            <p className="note">
              All models must share one dataset. A report stores its full spec,
              tables, figures and interactive arrays.
            </p>
            <button
              type="submit"
              className="primary full"
              disabled={busy || !loaded || !chosen.size || !analyses.size}
            >
              {busy ? "Starting analysis…" : "Build report →"}
            </button>
          </aside>
        </div>
      </form>
    </>
  );
}
function inline(text: string) {
  return text
    .split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
    .map((part, i) =>
      part.startsWith("**") ? (
        <strong key={i}>{part.slice(2, -2)}</strong>
      ) : part.startsWith("`") ? (
        <code key={i}>{part.slice(1, -1)}</code>
      ) : (
        part
      ),
    );
}
export function ReportMarkdown({ text }: { text: string }) {
  const blocks = text.split(/\n\s*\n/);
  return (
    <div className="report-markdown">
      {blocks.map((block, index) => {
        const lines = block.trim().split("\n");
        if (lines[0]?.startsWith("|")) {
          const cells = (line: string) =>
            line
              .split("|")
              .slice(1, -1)
              .map((x) => x.trim());
          const head = cells(lines[0]);
          return (
            <div className="table-wrap" key={index}>
              <table>
                <thead>
                  <tr>
                    {head.map((v, i) => (
                      <th key={i}>{inline(v)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {lines
                    .slice(2)
                    .filter((l) => l.startsWith("|"))
                    .map((line, i) => (
                      <tr key={i}>
                        {cells(line).map((v, j) => (
                          <td key={j}>{inline(v)}</td>
                        ))}
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          );
        }
        if (block.startsWith("#"))
          return <h3 key={index}>{inline(block.replace(/^#+\s*/, ""))}</h3>;
        return <p key={index}>{inline(block)}</p>;
      })}
    </div>
  );
}
export function ReportDetail({ id, version }: { id: string; version: number }) {
  const { data: record, error } = useResource<ReportRecord>(
      `reports/${id}`,
      version,
    ),
    [sectionName, setSectionName] = useState(""),
    [interactive, setInteractive] = useState(true);
  const report = record?.report,
    section =
      report?.sections.find((s) => s.name === sectionName) ||
      report?.sections[0];
  useViewTiming("report-detail", !!record);
  if (error) return <ErrorBox error={error} />;
  if (!record) return <Loading />;
  return (
    <>
      <PageHeader
        eyebrow="ANALYSIS / REPORT"
        title={
          report
            ? `${report.dataset.toUpperCase()} · network analysis`
            : "Building report"
        }
        actions={<Badge state={record.status.state} />}
      >
        <span className="mono">{id}</span>
        {report && (
          <>
            {" "}
            · {report.subjects.length} models · {report.split} split
          </>
        )}
      </PageHeader>
      <ErrorBox error={record.status.error || undefined} />
      {!report ? (
        <div className="panel padded">
          <h2>{record.status.stage || record.status.state}</h2>
          <p className="muted">This view updates as the report progresses.</p>
        </div>
      ) : (
        <>
          <div className="subject-strip">
            {report.subjects.map((s) => (
              <span key={s.ref}>
                <i style={{ background: s.color }} />
                {s.label}
                <small className="mono">{s.ref.split(":")[1]}</small>
              </span>
            ))}
          </div>
          <div className="report-layout">
            <nav className="report-nav" aria-label="Report sections">
              {report.sections.map((s, i) => (
                <button
                  key={s.name}
                  className={section?.name === s.name ? "current" : ""}
                  onClick={() => {
                    setSectionName(s.name);
                    setInteractive(true);
                  }}
                >
                  <span>{String(i + 1).padStart(2, "0")}</span>
                  {s.title}
                </button>
              ))}
            </nav>
            {section && (
              <section className="panel report-section" key={section.name}>
                <div className="panel-heading">
                  <div>
                    <div className="eyebrow">
                      {section.name.replaceAll("_", " ")}
                    </div>
                    <h2>{section.title}</h2>
                  </div>
                  {["embedding", "lrp"].includes(section.name) && (
                    <div className="segmented">
                      <button
                        className={interactive ? "current" : ""}
                        onClick={() => setInteractive(true)}
                      >
                        Interactive
                      </button>
                      <button
                        className={!interactive ? "current" : ""}
                        onClick={() => setInteractive(false)}
                      >
                        Figures
                      </button>
                    </div>
                  )}
                </div>
                {interactive && section.name === "embedding" && (
                  <Suspense fallback={<Loading />}>
                    <Embedding id={id} section={section} report={report} />
                  </Suspense>
                )}
                {interactive && section.name === "lrp" && (
                  <Suspense fallback={<Loading />}>
                    <LrpGallery id={id} section={section} report={report} />
                  </Suspense>
                )}
                <ReportMarkdown text={section.markdown} />
                {!(
                  ["embedding", "lrp"].includes(section.name) && interactive
                ) &&
                  section.figures.map((file) => (
                    <figure key={file}>
                      <a
                        href={fileURL(id, file)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        <img
                          src={fileURL(id, file)}
                          alt={`${section.title}: ${file.split("/").at(-1)}`}
                          loading="lazy"
                        />
                      </a>
                      <figcaption>
                        {file.split("/").at(-1)} · click to open full size
                      </figcaption>
                    </figure>
                  ))}
                {["embedding", "lrp"].includes(section.name) && interactive && (
                  <details className="padded">
                    <summary>Saved figures</summary>
                    {section.figures.map((file) => (
                      <img
                        key={file}
                        src={fileURL(id, file)}
                        alt={section.title}
                        loading="lazy"
                        className="report-image"
                      />
                    ))}
                  </details>
                )}
                <details className="padded raw-data">
                  <summary>Structured data & array downloads</summary>
                  <pre>
                    {JSON.stringify(
                      { summary: section.summary, data: section.data },
                      null,
                      2,
                    )}
                  </pre>
                  {section.arrays.map((file) => (
                    <a
                      className="block text-link"
                      key={file}
                      href={fileURL(id, file)}
                      download
                    >
                      {file.split("/").at(-1)} ↓
                    </a>
                  ))}
                </details>
              </section>
            )}
          </div>
        </>
      )}
    </>
  );
}
