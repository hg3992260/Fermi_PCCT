// 预加载：只暴露窗口控制与主题状态，保持 contextIsolation
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("shell", {
  command: (cmd) => ipcRenderer.invoke("shell:command", cmd),
  onState: (cb) => ipcRenderer.on("shell:state", (_e, payload) => cb(payload)),
});
