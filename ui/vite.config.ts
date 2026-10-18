import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  // One bundle per page, loaded from localhost: splitting would only add round trips. Most
  // of it is Base UI and React DOM. setup.html is the setup view alone, for the shell to show
  // before a server exists.
  build: {
    target: "es2023",
    chunkSizeWarningLimit: 1000,
    rollupOptions: { input: { main: "index.html", setup: "setup.html" } },
  },
});
