// What the page may ask of the window: its title bar buttons and state, the tray menu's
// live part, and the setup page's requirements (src/lib/shell.ts). Nothing else of
// Electron or Node reaches the page.
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("kisekiWindow", {
  layout: () => ipcRenderer.invoke("window:layout"),
  state: () => ipcRenderer.invoke("window:state"),
  onState: (listener) => {
    const handler = (_event, state) => listener(state);
    ipcRenderer.on("window:state", handler);
    return () => ipcRenderer.removeListener("window:state", handler);
  },
  minimize: () => ipcRenderer.send("window:minimize"),
  maximize: () => ipcRenderer.send("window:maximize"),
  close: () => ipcRenderer.send("window:close"),
  setTray: (items) => ipcRenderer.send("tray:items", items),
  onTray: (listener) => {
    const handler = (_event, action) => listener(action);
    ipcRenderer.on("tray:action", handler);
    return () => ipcRenderer.removeListener("tray:action", handler);
  },
  checkSetup: (fresh) => ipcRenderer.invoke("setup:check", fresh),
  runSetup: (id) => ipcRenderer.invoke("setup:act", id),
  openApp: () => ipcRenderer.invoke("setup:open"),
  onSetupJob: (listener) => {
    const handler = (_event, job) => listener(job);
    ipcRenderer.on("setup:job", handler);
    return () => ipcRenderer.removeListener("setup:job", handler);
  },
});
