// The window's own title bar (the Electron window is frameless): history, the command
// menu, and the window buttons GNOME's button-layout asks for, on the side it puts them.
import { ChevronLeftIcon, ChevronRightIcon, MinusIcon, SearchIcon, SquareIcon, XIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import { useHistory } from "@/lib/router";
import { shell, type WindowButton, type WindowState } from "@/lib/shell";
import { cn } from "@/lib/utils";

function useWindow() {
  const [layout, setLayout] = useState<{ left: WindowButton[]; right: WindowButton[] }>({ left: [], right: [] });
  const [state, setState] = useState<WindowState>({ maximized: false, focused: true });
  useEffect(() => {
    if (!shell) return;
    shell.layout().then(setLayout);
    shell.state().then(setState);
    return shell.onState(setState);
  }, []);
  return { layout, state };
}

function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" className={cn("size-4", className)} aria-hidden>
      <rect x="0.5" y="0.5" width="15" height="15" rx="4" className="fill-foreground" />
      <path
        d="M3.5 11.5c2.5 0 3.5-6.5 9-7"
        fill="none"
        className="stroke-background"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <circle cx="12.5" cy="4.5" r="1.4" className="fill-background" />
    </svg>
  );
}

function WindowButtons({ buttons, maximized }: { buttons: WindowButton[]; maximized: boolean }) {
  if (!shell || !buttons.length) return null;
  const icon = {
    minimize: <MinusIcon className="size-3.5" />,
    maximize: <SquareIcon className={cn("size-3", maximized && "size-2.5")} />,
    close: <XIcon className="size-3.5" />,
  };
  const label = { minimize: "Minimize", maximize: maximized ? "Restore" : "Maximize", close: "Close" };
  return (
    <div className="no-drag flex items-center gap-2 px-2">
      {buttons.map((button) => (
        <button
          key={button}
          type="button"
          aria-label={label[button]}
          onClick={() => shell?.[button]()}
          className="flex size-6 items-center justify-center rounded-full bg-accent text-foreground/80 transition-colors hover:bg-foreground/12 hover:text-foreground"
        >
          {icon[button]}
        </button>
      ))}
    </div>
  );
}

/** The setup page has no history or search: without `onSearch` the bar shows neither. */
export function TitleBar({ onSearch }: { onSearch?: () => void }) {
  const { layout, state } = useWindow();
  const history = useHistory();
  return (
    <header
      className={cn(
        "drag relative flex h-11 shrink-0 items-center transition-opacity",
        !state.focused && "[&_[data-dim]]:opacity-55",
      )}
    >
      <WindowButtons buttons={layout.left} maximized={state.maximized} />
      <div data-dim className="flex w-60 shrink-0 items-center gap-2 ps-4">
        <Logo />
        <span className="font-medium text-[13px] tracking-tight">kiseki</span>
        <div className={cn("no-drag ms-auto me-2 flex items-center", !onSearch && "hidden")}>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Back"
            disabled={!history.back}
            onClick={() => navigation.back()}
          >
            <ChevronLeftIcon />
          </Button>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Forward"
            disabled={!history.forward}
            onClick={() => navigation.forward()}
          >
            <ChevronRightIcon />
          </Button>
        </div>
      </div>
      <div className={cn("pointer-events-none absolute inset-x-0 flex justify-center", !onSearch && "hidden")}>
        <button
          type="button"
          data-dim
          onClick={onSearch}
          className="no-drag pointer-events-auto flex h-7 w-full max-w-96 items-center gap-2 rounded-lg border border-transparent bg-accent px-2.5 text-[13px] text-muted-foreground transition-colors hover:border-border hover:text-foreground"
        >
          <SearchIcon className="size-3.5" />
          <span>Search runs, reports, actions…</span>
          <KbdGroup className="ms-auto">
            <Kbd>Ctrl</Kbd>
            <Kbd>K</Kbd>
          </KbdGroup>
        </button>
      </div>
      <div className="flex-1" />
      <WindowButtons buttons={layout.right} maximized={state.maximized} />
    </header>
  );
}
