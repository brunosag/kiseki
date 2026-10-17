import { useEffect, useState } from "react";
import { useResource, useViewTiming } from "../api";
import type { Machine } from "../types";
import { ErrorBox, Loading, PageHeader } from "../components/Common";
export function Machines() {
  const [version, setVersion] = useState(0),
    { data, error } = useResource<Machine[]>(
      `machines${version ? "?fresh=1" : ""}`,
      version,
    );
  // a probe started by any request arrives as a `machines` event (App re-dispatches it)
  const [pushed, setPushed] = useState<Machine[] | null>(null);
  useEffect(() => {
    const onMachines = (event: Event) =>
      setPushed((event as CustomEvent<Machine[]>).detail);
    window.addEventListener("kiseki:machines", onMachines);
    return () => window.removeEventListener("kiseki:machines", onMachines);
  }, []);
  const machines = pushed ?? data;
  useViewTiming("machines", !!machines);
  return (
    <>
      <PageHeader
        eyebrow="COMPUTE / INVENTORY"
        title="Machines"
        actions={
          <button onClick={() => setVersion((v) => v + 1)}>↻ Probe now</button>
        }
      >
        This laptop and the remotes in machines.toml. Probes older than 30
        seconds refresh in the background.
      </PageHeader>
      <ErrorBox error={error} />
      {!machines && !error ? (
        <Loading />
      ) : (
        <div className="machine-cards">
          {machines?.map((m) => (
            <section className="panel machine-card" key={m.name}>
              <div className="flex items-center justify-between">
                <span className="machine-icon">▣</span>
                <span
                  className={`connection ${m.error ? "offline" : "online"}`}
                >
                  <i />
                  {m.error ? "Unreachable" : m.ssh ? "Remote" : "This machine"}
                </span>
              </div>
              <h2>{m.name}</h2>
              <p className="muted">
                {m.description || m.ssh || "Local workstation"}
              </p>
              <dl className="key-values">
                <dt>Connection</dt>
                <dd className="mono">{m.ssh || "local"}</dd>
                <dt>Runner</dt>
                <dd>{m.runner || m.runner_pid ? "Running" : "Idle"}</dd>
                {m.gpu_awake !== undefined && (
                  <>
                    <dt>Discrete GPU</dt>
                    <dd>
                      {m.gpu_awake === null
                        ? "Unknown"
                        : m.gpu_awake
                          ? "Awake"
                          : "Sleeping"}
                    </dd>
                  </>
                )}
              </dl>
              {m.gpus?.map((gpu, i) => (
                <div className="gpu" key={i}>
                  <strong>{gpu.name}</strong>
                  <p>
                    {gpu.memory_used_mib} / {gpu.memory_total_mib} MiB
                  </p>
                  <div className="progress">
                    <i
                      style={{
                        width: `${(100 * gpu.memory_used_mib) / gpu.memory_total_mib}%`,
                      }}
                    />
                  </div>
                  <small>{gpu.utilization}% GPU utilization</small>
                </div>
              ))}
              {m.name === "local" && !m.gpus && (
                <p className="note">
                  Memory and load show while the GPU is awake; reading them
                  would wake it otherwise.
                </p>
              )}
              {m.gpu_error && <ErrorBox error={`GPU: ${m.gpu_error}`} />}
              {m.error && <ErrorBox error={m.error} />}
              {m.notice && (
                <p className="note">
                  Shared: runs here keep a usage notice in {m.notice}.
                </p>
              )}
            </section>
          ))}
        </div>
      )}
    </>
  );
}
