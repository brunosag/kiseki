// The AppImage, built by scripts/appimage.sh: the shell in app.asar, and in resources the
// Python payload with the built UI (kiseki/, from build/), a pinned uv (bin/uv) and the icon.
// Electron is the version nixpkgs ships, which the NixOS app runs.
module.exports = {
  appId: "io.github.brunosag.kiseki",
  productName: "kiseki",
  executableName: "kiseki",
  electronVersion: "41.9.1",
  // The static AppImage runtime: no libfuse2 needed (Ubuntu 24.04 ships without it).
  toolsets: { appimage: "1.0.3" },
  directories: { output: "release" },
  files: ["package.json", "electron/**/*", "!electron/*.test.cjs", "!electron/*.svg", "!node_modules/**"],
  extraResources: [
    { from: "build/kiseki", to: "kiseki" },
    { from: "build/uv", to: "bin/uv" },
    { from: "electron/icon.svg", to: "icon.svg" },
  ],
  linux: {
    target: "AppImage",
    category: "Science",
    icon: "electron/icon.png",
    artifactName: "kiseki-${version}-x86_64.AppImage",
  },
  publish: null,
};
