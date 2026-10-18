// The window: title bar across the top, the sidebar, and the current view in an inset
// panel. Dialogs (new run, new report) and the command menu open from anywhere.
import { useEffect, useRef, useState } from "react";
import { CommandMenu } from "@/components/command-menu";
import { Nav } from "@/components/nav";
import { TitleBar } from "@/components/title-bar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { ToastProvider } from "@/components/ui/toast";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useLive } from "@/lib/live";
import { navigate, type Route, useRoute } from "@/lib/router";
import { shell } from "@/lib/shell";
import { trayItems } from "@/lib/tray";
import type { Json } from "@/lib/types";
import { CompareView } from "@/views/compare";
import { MachinesView } from "@/views/machines";
import { NewReportDialog } from "@/views/new-report";
import { NewRunDialog } from "@/views/new-run";
import { ReportView } from "@/views/report";
import { ReportsView } from "@/views/reports";
import { RunView } from "@/views/run";
import { RunsView } from "@/views/runs";

/** Values a new run starts from: another run's spec, or nothing for the defaults. */
export type RunDraft = Record<string, Json> | null;

function typing(target: EventTarget | null) {
  const element = target as HTMLElement | null;
  return !!element?.closest("input, textarea, select, [contenteditable], [role=dialog]");
}

export function App() {
  const route = useRoute();
  const [runDraft, setRunDraft] = useState<RunDraft | undefined>(undefined);
  const [reportModels, setReportModels] = useState<string[] | undefined>(undefined);
  const [palette, setPalette] = useState(false);
  const newRun = (draft: RunDraft = null) => setRunDraft(draft);
  const newReport = (refs: string[] = []) => setReportModels(refs);

  useEffect(() => {
    const keys = (event: KeyboardEvent) => {
      if (event.key === "k" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        setPalette((open) => !open);
      } else if (event.key === "c" && !event.ctrlKey && !event.metaKey && !event.altKey && !typing(event.target)) {
        event.preventDefault();
        newRun();
      }
    };
    window.addEventListener("keydown", keys);
    return () => window.removeEventListener("keydown", keys);
  }, []);

  return (
    <ToastProvider>
      <TooltipProvider delay={400}>
        <div className="flex h-svh flex-col">
          <TitleBar onSearch={() => setPalette(true)} />
          <SidebarProvider className="min-h-0 flex-1">
            <Nav route={route} onNewRun={() => newRun()} />
            <main className="relative me-2 mb-2 flex min-w-0 flex-1 flex-col overflow-hidden rounded-xl border bg-background shadow-xs/5">
              <View route={route} onNewRun={newRun} onNewReport={newReport} />
            </main>
          </SidebarProvider>
        </div>
        <NewRunDialog draft={runDraft} onClose={() => setRunDraft(undefined)} />
        <NewReportDialog models={reportModels} onClose={() => setReportModels(undefined)} />
        <CommandMenu open={palette} onOpenChange={setPalette} onNewRun={() => newRun()} onNewReport={() => newReport()} />
        {shell && <Tray onNewRun={() => newRun()} />}
      </TooltipProvider>
    </ToastProvider>
  );
}

/** Keeps the shell's tray menu current and acts on its items. Its own component, so live
 * updates re-render only it. */
function Tray({ onNewRun }: { onNewRun: () => void }) {
  const { runs, reports, options } = useLive();
  const sent = useRef("");
  const newRun = useRef(onNewRun);
  newRun.current = onNewRun;
  useEffect(() => {
    const items = trayItems(runs, reports, options);
    const text = JSON.stringify(items);
    if (text === sent.current) return;
    sent.current = text;
    shell?.setTray(items);
  }, [runs, reports, options]);
  useEffect(
    () => shell?.onTray((action) => ("open" in action ? navigate(action.open) : newRun.current())),
    [],
  );
  return null;
}

function View({
  route,
  onNewRun,
  onNewReport,
}: {
  route: Route;
  onNewRun: (draft?: RunDraft) => void;
  onNewReport: (refs?: string[]) => void;
}) {
  switch (route.name) {
    case "run":
      return <RunView key={route.id} id={route.id} onNewRun={onNewRun} onNewReport={onNewReport} />;
    case "compare":
      return <CompareView ids={route.ids} onNewReport={onNewReport} />;
    case "reports":
      return <ReportsView onNewReport={onNewReport} />;
    case "report":
      return <ReportView key={route.id} id={route.id} />;
    case "machines":
      return <MachinesView />;
    default:
      return <RunsView onNewRun={() => onNewRun()} onNewReport={onNewReport} />;
  }
}
