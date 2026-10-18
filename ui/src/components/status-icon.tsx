// A run's or report's state as one small glyph, in the manner of Linear's issue states:
// a dashed ring while queued, a ring filling with progress while running, a filled
// check when completed.
import { cn } from "@/lib/utils";

export const STATE_ORDER = ["running", "queued", "paused", "failed", "stopped", "completed"];

export const STATE_LABEL: Record<string, string> = {
  queued: "Queued",
  running: "Running",
  paused: "Paused",
  stopped: "Stopped",
  failed: "Failed",
  completed: "Completed",
};

const COLOR: Record<string, string> = {
  queued: "text-muted-foreground",
  running: "text-amber-500",
  paused: "text-muted-foreground",
  stopped: "text-muted-foreground",
  failed: "text-red-500",
  completed: "text-indigo-500 dark:text-indigo-400",
};

export function StatusIcon({
  state,
  progress = 0,
  className,
}: {
  state: string;
  progress?: number;
  className?: string;
}) {
  const ring = <circle cx="7" cy="7" r="6" fill="none" stroke="currentColor" strokeWidth="1.5" />;
  let glyph: React.ReactNode;
  switch (state) {
    case "queued":
      glyph = (
        <circle cx="7" cy="7" r="6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="2.4 1.8" />
      );
      break;
    case "running": {
      // A pie from 12 o'clock, at least a sliver so a fresh run reads as started.
      const angle = Math.max(0.06, Math.min(progress, 0.999)) * 2 * Math.PI;
      const x = 7 + 3.5 * Math.sin(angle);
      const y = 7 - 3.5 * Math.cos(angle);
      glyph = (
        <>
          {ring}
          <path d={`M7 7 L7 3.5 A3.5 3.5 0 ${angle > Math.PI ? 1 : 0} 1 ${x} ${y} Z`} fill="currentColor" />
        </>
      );
      break;
    }
    case "paused":
      glyph = (
        <>
          {ring}
          <path d="M5.5 4.75v4.5M8.5 4.75v4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </>
      );
      break;
    case "completed":
      glyph = (
        <>
          <circle cx="7" cy="7" r="6.75" fill="currentColor" />
          <path
            d="M4.4 7.2l1.8 1.8 3.5-3.6"
            fill="none"
            stroke="var(--background)"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </>
      );
      break;
    default: // failed, stopped
      glyph = (
        <>
          <circle cx="7" cy="7" r="6.75" fill="currentColor" />
          <path
            d="M5 5l4 4M9 5l-4 4"
            stroke="var(--background)"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </>
      );
  }
  return (
    <svg
      viewBox="0 0 14 14"
      className={cn("size-3.5 shrink-0", COLOR[state] ?? "text-muted-foreground", className)}
      aria-label={STATE_LABEL[state] ?? state}
      role="img"
    >
      {glyph}
    </svg>
  );
}
