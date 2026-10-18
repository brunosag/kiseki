// The sidebar: the one primary action, and the three places there are.
import { FileChartColumnIcon, FlaskConicalIcon, PlusIcon, ServerIcon } from "lucide-react";
import { Kbd } from "@/components/ui/kbd";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { Spinner } from "@/components/ui/spinner";
import { isActive } from "@/lib/format";
import { useLive } from "@/lib/live";
import type { Route } from "@/lib/router";

export function Nav({ route, onNewRun }: { route: Route; onNewRun: () => void }) {
  const live = useLive();
  const activeRuns = live.runs?.filter((run) => isActive(run.status.state)).length ?? 0;
  const building = live.reports?.filter((report) => isActive(report.status.state)).length ?? 0;
  const unreachable = live.machines?.filter((machine) => machine.error || machine.gpu_error).length ?? 0;
  const items = [
    {
      href: "/runs",
      label: "Runs",
      icon: FlaskConicalIcon,
      on: route.name === "runs" || route.name === "run" || route.name === "compare",
      badge: activeRuns || null,
    },
    {
      href: "/reports",
      label: "Reports",
      icon: FileChartColumnIcon,
      on: route.name === "reports" || route.name === "report",
      badge: building || null,
    },
    {
      href: "/machines",
      label: "Machines",
      icon: ServerIcon,
      on: route.name === "machines",
      badge: unreachable ? <span className="size-1.5 rounded-full bg-red-500" aria-label="A machine has a problem" /> : null,
    },
  ];
  return (
    <Sidebar collapsible="none" className="w-60 bg-transparent">
      <SidebarHeader className="px-3 pt-1 pb-2">
        <button
          type="button"
          onClick={onNewRun}
          className="flex h-8 items-center gap-2 rounded-lg border bg-background px-2.5 text-[13px] text-foreground shadow-xs/5 transition-colors hover:bg-accent dark:bg-input/32"
        >
          <PlusIcon className="size-4 opacity-80" />
          <span className="font-medium">New run</span>
          <Kbd className="ms-auto">C</Kbd>
        </button>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup className="px-3 py-0">
          <SidebarMenu className="gap-0.5">
            {items.map((item) => (
              <SidebarMenuItem key={item.href}>
                <SidebarMenuButton
                  isActive={item.on}
                  render={<a href={item.href} />}
                  className="h-8 text-[13px] text-foreground/80 data-[active=true]:font-medium"
                >
                  <item.icon />
                  <span>{item.label}</span>
                </SidebarMenuButton>
                {item.badge != null && <SidebarMenuBadge className="top-1.5">{item.badge}</SidebarMenuBadge>}
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>
      {!live.connected && (
        <SidebarFooter className="px-4 pb-3">
          <div className="flex items-center gap-2 text-muted-foreground text-xs">
            <Spinner className="size-3" />
            Reconnecting to the server…
          </div>
        </SidebarFooter>
      )}
    </Sidebar>
  );
}
