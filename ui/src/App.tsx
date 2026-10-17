import { Component, useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api, eventStream, observeRunStatus, resetRunRows } from "./api";
import type { Options, ReportRecord, Run } from "./types";
import { ErrorBox, Loading } from "./components/Common";
import { Runs } from "./views/Runs";
import { NewRun } from "./views/NewRun";
import { RunDetail } from "./views/RunDetail";
import { Compare } from "./views/Compare";
import { NewReport, ReportDetail, Reports } from "./views/Reports";
import { Machines } from "./views/Machines";
function readRoute() {
  return location.hash.slice(1) || "/runs";
}
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: string }
> {
  state = { error: "" };
  static getDerivedStateFromError(e: Error) {
    return { error: e.message };
  }
  render() {
    return this.state.error ? (
      <div className="padded">
        <h1>This view could not render</h1>
        <ErrorBox error={this.state.error} />
        <button onClick={() => location.reload()}>Reload workspace</button>
      </div>
    ) : (
      this.props.children
    );
  }
}
export default function App() {
  const [route, setRoute] = useState(readRoute),
    [options, setOptions] = useState<Options>(),
    [runs, setRuns] = useState<Run[]>([]),
    [reports, setReports] = useState<ReportRecord[]>([]),
    [selected, setSelected] = useState<Set<string>>(new Set()),
    [error, setError] = useState(""),
    [ready, setReady] = useState(false),
    [connected, setConnected] = useState(false),
    [version, setVersion] = useState(0);
  const navigate = useCallback((path: string) => {
    location.hash = path;
  }, []);
  useEffect(() => {
    const listener = () => {
      setRoute(readRoute());
      document.querySelector("main")?.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", listener);
    return () => window.removeEventListener("hashchange", listener);
  }, []);
  useEffect(() => {
    let mounted = true;
    const known = new Map<string, Run["status"]>();
    const track = (run: Pick<Run, "id" | "status">) => {
      observeRunStatus(run.id, known.get(run.id), run.status);
      known.set(run.id, run.status);
    };
    const refresh = async () => {
      try {
        const [o, r, p] = await Promise.all([
          api<Options>("options"),
          api<Run[]>("runs"),
          api<ReportRecord[]>("reports"),
        ]);
        if (mounted) {
          setOptions(o);
          r.forEach(track);
          setRuns(r);
          setReports(p);
          setReady(true);
          setError("");
        }
      } catch (e) {
        if (mounted) setError((e as Error).message);
      }
    };
    void refresh();
    const stream = eventStream();
    stream.onopen = () => {
      setConnected(true);
      for (const [id, status] of known)
        if (status.state !== "completed") resetRunRows(id);
      void refresh();
      setVersion((v) => v + 1);
    };
    stream.onerror = () => setConnected(false);
    const onRun = (event: MessageEvent) => {
      const incoming = JSON.parse(event.data);
      const exists = known.has(incoming.id);
      track(incoming);
      if (exists)
        setRuns((previous) =>
          previous.map((r) =>
            r.id === incoming.id ? { ...r, status: incoming.status } : r,
          ),
        );
      else void refresh();
      setVersion((v) => v + 1);
    };
    const onReport = (event: MessageEvent) => {
      const incoming = JSON.parse(event.data);
      setReports((previous) =>
        previous.map((r) =>
          r.id === incoming.id ? { ...r, status: incoming.status } : r,
        ),
      );
      void api<ReportRecord[]>("reports")
        .then((p) => {
          if (mounted) setReports(p);
        })
        .catch(() => {});
      setVersion((v) => v + 1);
    };
    const onMachines = (event: MessageEvent) =>
      window.dispatchEvent(
        new CustomEvent("kiseki:machines", { detail: JSON.parse(event.data) }),
      );
    stream.addEventListener("run", onRun);
    stream.addEventListener("report", onReport);
    stream.addEventListener("machines", onMachines);
    return () => {
      mounted = false;
      stream.close();
    };
  }, []);
  const toggle = useCallback(
    (id: string) =>
      setSelected((previous) => {
        const next = new Set(previous);
        next.has(id) ? next.delete(id) : next.add(id);
        return next;
      }),
    [],
  );
  const selectMany = useCallback(
    (ids: string[], checked: boolean) =>
      setSelected((previous) => {
        const next = new Set(previous);
        ids.forEach((id) => (checked ? next.add(id) : next.delete(id)));
        return next;
      }),
    [],
  );
  const select = useCallback(
    (id: string) => setSelected((previous) => new Set([...previous, id])),
    [],
  );
  const selectedRuns = runs.filter((r) => selected.has(r.id));
  const nav = [
    { path: "/runs", icon: "▤", label: "Runs", count: runs.length },
    { path: "/compare", icon: "⌁", label: "Compare", count: selected.size },
    { path: "/reports", icon: "▧", label: "Reports", count: reports.length },
    { path: "/machines", icon: "▣", label: "Machines" },
  ];
  let view: ReactNode;
  if (!ready || !options)
    view = (
      <>
        <ErrorBox error={error} />
        {!error && <Loading />}
        {error && (
          <button onClick={() => location.reload()}>Retry connection</button>
        )}
      </>
    );
  else if (route === "/new")
    view = (
      <NewRun
        options={options}
        created={(created) => {
          setRuns((previous) => [...previous, ...created]);
          setSelected(new Set(created.map((r) => r.id)));
          navigate("/runs");
        }}
      />
    );
  else if (route.startsWith("/runs/"))
    view = (
      <RunDetail
        key={route}
        id={decodeURIComponent(route.slice(6))}
        statusVersion={version}
        navigate={navigate}
        select={select}
      />
    );
  else if (route === "/compare")
    view = <Compare runs={selectedRuns} navigate={navigate} />;
  else if (route === "/reports/new")
    view = (
      <NewReport
        runs={selectedRuns}
        options={options}
        created={(r) => {
          setReports((previous) => [...previous, r]);
          navigate(`/reports/${r.id}`);
        }}
        navigate={navigate}
      />
    );
  else if (route.startsWith("/reports/"))
    view = (
      <ReportDetail
        key={route}
        id={decodeURIComponent(route.slice(9))}
        version={version}
      />
    );
  else if (route === "/reports")
    view = <Reports reports={reports} navigate={navigate} />;
  else if (route === "/machines") view = <Machines />;
  else
    view = (
      <Runs
        runs={runs}
        selected={selected}
        toggle={toggle}
        selectMany={selectMany}
        navigate={navigate}
      />
    );
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => navigate("/runs")}>
          <span className="brand-mark">奇</span>
          <span>
            kiseki<small>OPTIMIZATION LAB</small>
          </span>
        </button>
        <div className="sidebar-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {nav.map((item) => (
            <button
              key={item.path}
              className={route.startsWith(item.path) ? "current" : ""}
              onClick={() => navigate(item.path)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
              {item.count !== undefined && <small>{item.count}</small>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="lab-mark">SGD × EVOLUTION</div>
          <p>
            Small networks.
            <br />
            Different ways to learn.
          </p>
          <div className={`connection ${connected ? "online" : "offline"}`}>
            <i />
            {connected ? "Live · local store" : "Reconnecting…"}
          </div>
          <span className="version">kiseki / desktop</span>
        </div>
      </aside>
      <div className="workspace">
        <div className="topbar">
          <span>
            Workspace <span className="crumb">/</span>{" "}
            {route === "/new"
              ? "New run"
              : route.startsWith("/reports")
                ? "Reports"
                : route.startsWith("/runs/")
                  ? "Run detail"
                  : route.slice(1).replace(/^./, (c) => c.toUpperCase())}
          </span>
          <span className="topbar-right">
            <i className="machine-dot" />
            Local compute <span className="keycap">GPU</span>
          </span>
        </div>
        <main>
          <ErrorBoundary key={route}>{view}</ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
