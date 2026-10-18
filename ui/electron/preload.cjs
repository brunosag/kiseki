// What the page may ask of the window: its title bar buttons and state. Nothing else of
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
});
