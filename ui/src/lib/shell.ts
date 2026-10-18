// The Electron shell's bridge (electron/preload.cjs): the window's buttons and state for
// the title bar, and the tray menu. Undefined in a plain browser.
import type { TrayAction, TrayItem } from "./tray.ts";

export type WindowButton = "minimize" | "maximize" | "close";
export interface WindowState {
  maximized: boolean;
  focused: boolean;
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
}
declare global {
  interface Window {
    kisekiWindow?: KisekiWindow;
  }
}

export const shell = window.kisekiWindow;
