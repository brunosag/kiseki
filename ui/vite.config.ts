import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  // One bundle, loaded from localhost: splitting would only add round trips. Most of it is
  // Base UI and React DOM.
  build: { target: "es2023", chunkSizeWarningLimit: 1000 },
});
