// Light or dark, as the system says (main.tsx keeps the `dark` class in step).
import { useSyncExternalStore } from "react";

const dark = matchMedia("(prefers-color-scheme: dark)");

export function useDark(): boolean {
  return useSyncExternalStore(
    (listener) => {
      dark.addEventListener("change", listener);
      return () => dark.removeEventListener("change", listener);
    },
    () => dark.matches,
  );
}

/** A CSS color expression (tokens included) resolved to what canvas can draw. */
export function resolveColor(css: string): string {
  const probe = document.createElement("span");
  probe.style.color = css;
  probe.style.display = "none";
  document.body.append(probe);
  const color = getComputedStyle(probe).color;
  probe.remove();
  return color;
}

/** Categorical series colors in fixed order (validated: dataviz palette, both modes). */
export const SERIES = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
};
