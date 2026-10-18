// The Electron shell's bridge (electron/preload.cjs): the window's buttons and state for
// the title bar, the tray menu, and the requirements the setup view lists
// (electron/requirements.cjs). Undefined in a plain browser.
import type { TrayAction, TrayItem } from "./tray.ts";

export type WindowButton = "minimize" | "maximize" | "close";
export interface WindowState {
  maximized: boolean;
  focused: boolean;
}
export type RequirementState = "ok" | "missing" | "warning";
export interface SetupJob {
  state: "running" | "done" | "failed";
  lines: string[];
  error: string | null;
}
export interface Requirement {
  id: string;
  title: string;
  required: boolean;
  state: RequirementState;
  version?: string;
  detail: string;
  action?: { label: string; kind: "install" | "open" };
  job: SetupJob | null;
}

interface KisekiWindow {
  layout(): Promise<{ left: WindowButton[]; right: WindowButton[] }>;
  state(): Promise<WindowState>;
  onState(listener: (state: WindowState) => void): () => void;
  minimize(): void;
  maximize(): void;
  close(): void;
  /** Replaces the tray menu's live part. */
  setTray(items: TrayItem[]): void;
  /** A tray item was chosen; the shell has already shown the window. */
  onTray(listener: (action: TrayAction) => void): () => void;
  /** Every requirement; `fresh` probes the Python environment again (seconds). */
  checkSetup(fresh?: boolean): Promise<Requirement[]>;
  /** Runs a requirement's action; its progress arrives through onSetupJob. */
  runSetup(id: string): Promise<void>;
  /** From the setup page, once the environment is ready: starts the server and opens it. */
  openApp(): Promise<void>;
  onSetupJob(listener: (job: SetupJob & { id: string }) => void): () => void;
}
declare global {
  interface Window {
    kisekiWindow?: KisekiWindow;
  }
}

export const shell = globalThis.window?.kisekiWindow;
