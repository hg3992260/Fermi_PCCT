/* test-ui.mjs — 真机层：Electron 里用真实鼠标事件点界面（--probe-ui） */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const require = createRequire(import.meta.url);

function electronBin() {
  // electron 包在 node 环境里 require 出来就是可执行文件路径
  try {
    const p = require("electron");
    if (typeof p === "string" && existsSync(p)) return p;
  } catch { /* ignore */ }
  const win = path.join(ROOT, "node_modules", "electron", "dist", "electron.exe");
  if (existsSync(win)) return win;
  const nix = path.join(ROOT, "node_modules", ".bin", "electron");
  if (existsSync(nix)) return nix;
  return null;
}

const bin = electronBin();
if (!bin) {
  console.error("找不到 electron，可先 pnpm add -D electron");
  process.exit(2);
}

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;   // 关键：否则 electron 会当 node 跑

const out = await new Promise((resolve, reject) => {
  const p = spawn(bin, [".", "--probe-ui"], { cwd: ROOT, env, stdio: ["ignore", "pipe", "pipe"] });
  let buf = "";
  p.stdout.on("data", (d) => (buf += d));
  p.stderr.on("data", (d) => { if (/PROBE_UI/.test(d)) buf += d; });
  const timer = setTimeout(() => { p.kill(); reject(new Error("Electron 探针超时")); }, 120000);
  p.on("exit", () => { clearTimeout(timer); resolve(buf); });
  p.on("error", reject);
});

const m = out.match(/PROBE_UI (\{.*\})/);
if (!m) {
  console.error("✗ 真机探针没有输出结果");
  process.exit(1);
}

const r = JSON.parse(m[1]);
console.log(`  真机交互：${r.total - r.fail}/${r.total} 通过`);
if (r.fail) {
  for (const b of r.bad) console.error("  ✗ " + b);
  process.exit(1);
}
