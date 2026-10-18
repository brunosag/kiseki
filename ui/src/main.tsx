import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { App } from "./app";
import { start } from "./lib/live";
import { startRouter } from "./lib/router";

// Light or dark as the system is (GNOME's appearance setting), switching live.
const dark = matchMedia("(prefers-color-scheme: dark)");
const theme = () => document.documentElement.classList.toggle("dark", dark.matches);
theme();
dark.addEventListener("change", theme);

startRouter();
start();
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
