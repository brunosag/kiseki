// The setup page alone (setup.html), which the shell shows over kiseki://app before the
// Python environment, and so the server, exists.
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { TitleBar } from "./components/title-bar";
import { SetupView } from "./views/setup";

const dark = matchMedia("(prefers-color-scheme: dark)");
const theme = () => document.documentElement.classList.toggle("dark", dark.matches);
theme();
dark.addEventListener("change", theme);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <div className="flex h-svh flex-col">
      <TitleBar />
      <main className="relative mx-2 mb-2 flex min-w-0 flex-1 flex-col overflow-hidden rounded-xl border bg-background shadow-xs/5">
        <SetupView standalone />
      </main>
    </div>
  </StrictMode>,
);
