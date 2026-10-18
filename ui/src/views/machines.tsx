// The machines runs can go to (from ~/.config/kiseki/machines.toml): whether each is
// reachable, its GPU, and its queue. Probing a remote takes an SSH round trip, so the
// list shows the last probe until Refresh asks again.
import { RefreshCwIcon } from "lucide-react";
import { useState } from "react";
import { PageBody, PageHeader } from "@/components/page";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Spinner } from "@/components/ui/spinner";
import { toastManager } from "@/components/ui/toast";
import { gpuName, isActive } from "@/lib/format";
import { probeMachines, useLive } from "@/lib/live";
import type { Machine, Run } from "@/lib/types";
import { cn } from "@/lib/utils";

export function MachinesView() {
  const { machines, runs } = useLive();
  const [probing, setProbing] = useState(false);
  const refresh = () => {
    setProbing(true);
    probeMachines()
      .catch((error: Error) => toastManager.add({ type: "error", title: "Could not probe machines", description: error.message }))
      .finally(() => setProbing(false));
  };
  return (
    <>
      <PageHeader
        actions={
          <Button variant="ghost" size="sm" className="text-[13px]" onClick={refresh} disabled={probing}>
            {probing ? <Spinner /> : <RefreshCwIcon />}
            Refresh
          </Button>
        }
      >
        <span className="font-medium">Machines</span>
      </PageHeader>
      <PageBody>
        {!machines ? (
          <Empty className="h-full">
            <Spinner />
          </Empty>
        ) : !machines.length ? (
          <Empty className="h-full">
            <EmptyHeader>
              <EmptyTitle className="font-medium text-base">No machines</EmptyTitle>
              <EmptyDescription>Add remotes in ~/.config/kiseki/machines.toml.</EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <ul aria-label="Machines" className="py-1">
            {machines.map((machine) => (
              <MachineRow key={machine.name} machine={machine} runs={runs ?? []} />
            ))}
          </ul>
        )}
      </PageBody>
    </>
  );
}

type Health = "ok" | "idle" | "warning" | "down";

const DOT: Record<Health, string> = {
  ok: "bg-emerald-500",
  idle: "bg-muted-foreground/50",
  warning: "bg-amber-500",
  down: "bg-red-500",
};

function health(machine: Machine): [Health, string] {
  if (machine.error) return ["down", "Unreachable"];
  if (machine.gpu_error) return ["warning", "GPU unavailable"];
  if (machine.gpu_awake === false) return ["idle", "GPU asleep"];
  return ["ok", "Ready"];
}

const gib = (mib: number) => `${(mib / 1024).toFixed(mib < 10240 ? 1 : 0)} GB`;

function gpuText(machine: Machine): string {
  if (machine.gpus?.length)
    return machine.gpus
      .map((gpu) => {
        const name = gpuName(gpu.name);
        return `${name} · ${gib(gpu.memory_used_mib)} of ${gib(gpu.memory_total_mib)} · ${gpu.utilization}%`;
      })
      .join(", ");
  if (machine.gpu_awake === false) return "GPU asleep";
  // A remote without nvidia-smi has no GPU; errors show under the description.
  return machine.ssh && !machine.error && !machine.gpu_error ? "No GPU" : "";
}

function MachineRow({ machine, runs }: { machine: Machine; runs: Run[] }) {
  const [state, label] = health(machine);
  const active = runs.filter((run) => isActive(run.status.state) && (run.status.machine ?? "local") === machine.name);
  const local = !machine.ssh;
  const queue = local ? machine.runner : machine.runner_pid != null;
  return (
    <li className="flex min-h-12 items-center gap-3 px-4 py-2 text-[13px]">
      <span className={cn("size-2 shrink-0 rounded-full", DOT[state])} title={label} />
      <span className="w-20 shrink-0 font-medium">{machine.name}</span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-muted-foreground">
          {machine.description || (local ? "This machine" : machine.ssh)}
        </span>
        {(machine.error || machine.gpu_error) && (
          <span className="block truncate text-destructive-foreground text-xs" title={machine.error ?? machine.gpu_error ?? ""}>
            {machine.error ?? machine.gpu_error}
          </span>
        )}
      </span>
      <span className="w-56 shrink-0 truncate text-muted-foreground text-xs">
        {gpuText(machine)}
      </span>
      <span className="w-28 shrink-0 text-muted-foreground text-xs">
        {machine.error ? "" : queue ? "Queue running" : "Queue idle"}
        {!local && !machine.error && !machine.deployed && " · not set up"}
      </span>
      <span className="w-20 shrink-0 text-right text-xs tabular-nums">
        {active.length ? `${active.length} active` : <span className="text-muted-foreground">No runs</span>}
      </span>
    </li>
  );
}
