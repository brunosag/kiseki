// Actions on runs that views share, reported as toasts.
import { toastManager } from "@/components/ui/toast";
import { api } from "./api";
import type { Run } from "./types";

const VERB = { stop: "Stop", pause: "Pause", resume: "Resume" };

/** Stop, pause or resume runs, wherever each lives; the server answers per run. */
export async function control(runs: Run[], action: "stop" | "pause" | "resume"): Promise<void> {
  try {
    const results = await Promise.all(
      runs.map((run) => api<{ id: string; result: string }>(`runs/${encodeURIComponent(run.id)}/${action}`, {})),
    );
    toastManager.add({
      type: "success",
      title: `${VERB[action]}: ${results.length} run${results.length > 1 ? "s" : ""}`,
      description: [...new Set(results.map((result) => result.result))].join(", "),
    });
  } catch (error) {
    toastManager.add({ type: "error", title: `Could not ${action}`, description: (error as Error).message });
  }
}
