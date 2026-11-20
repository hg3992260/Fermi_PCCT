/* test-mcp.mjs — MCP 层：拉起（或复用）应用 → 通过 stdio 的 MCP 服务器调工具 → 断言 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const require = createRequire(import.meta.url);
const BRIDGE = process.env.ATLAS_BRIDGE || "http://127.0.0.1:8701";
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function health() {
  try {
    const r = await fetch(BRIDGE + "/health", { signal: AbortSignal.timeout(1500) });
    return r.ok;
  } catch {
    return false;
  }
}

function electronBin() {
  try {
    const p = require("electron");
    if (typeof p === "string" && existsSync(p)) return p;
  } catch { /* ignore */ }
  const win = path.join(ROOT, "node_modules", "electron", "dist", "electron.exe");
  return existsSync(win) ? win : null;
}

let appProc = null;
if (!(await health())) {
  const bin = electronBin();
  if (!bin) {
    console.error("✗ 找不到 electron，无法启动应用");
    process.exit(2);
  }
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  appProc = spawn(bin, ["."], { cwd: ROOT, env, stdio: "ignore" });
  for (let i = 0; i < 40 && !(await health()); i++) await wait(500);
  if (!(await health())) {
    console.error("✗ 应用启动后桥仍不可用（127.0.0.1:8701）");
    appProc?.kill();
    process.exit(1);
  }
}

/* ── MCP 客户端（stdio，逐行 JSON-RPC） ── */
const mcp = spawn(process.execPath, [path.join(ROOT, "mcp", "atlas-mcp.mjs")], {
  cwd: ROOT,
  env: { ...process.env, ATLAS_BRIDGE: BRIDGE },
  stdio: ["pipe", "pipe", "pipe"],
});

const pending = new Map();
let seq = 0;
let buf = "";
mcp.stdout.on("data", (d) => {
  buf += d;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (!line) continue;
    try {
      const msg = JSON.parse(line);
      const p = pending.get(msg.id);
      if (p) { pending.delete(msg.id); p(msg); }
    } catch { /* ignore */ }
  }
});
mcp.stderr.on("data", () => {});

function rpc(method, params) {
  const id = ++seq;
  mcp.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
  return new Promise((resolve, reject) => {
    pending.set(id, resolve);
    setTimeout(() => reject(new Error(method + " 超时")), 20000);
  });
}

const call = async (name, args = {}) => {
  const r = await rpc("tools/call", { name, arguments: args });
  const c = r.result?.content?.[0];
  if (r.result?.isError) throw new Error(c?.text || "tool error");
  if (c?.type === "image") return { image: c.data };
  return JSON.parse(c?.text || "null");
};

const results = [];
const check = async (name, fn) => {
  try {
    const r = await fn();
    results.push({ name, ok: !!r.ok, detail: r.detail || "" });
  } catch (e) {
    results.push({ name, ok: false, detail: e.message });
  }
};

try {
  const init = await rpc("initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "test", version: "1" },
  });
  await check("mcp:initialize", async () => ({
    ok: init.result?.protocolVersion === "2024-11-05",
    detail: init.result?.serverInfo?.name,
  }));

  const list = await rpc("tools/list", {});
  await check("mcp:tools/list", async () => ({
    ok: (list.result?.tools?.length || 0) >= 8,
    detail: `${list.result?.tools?.length || 0} 个工具`,
  }));

  await check("tool:layers", async () => {
    const l = await call("atlas_layers");
    return { ok: Array.isArray(l) && l.length === 10, detail: `${l.length} 层` };
  });

  await check("tool:scene", async () => {
    const s = await call("atlas_scene", { id: "bands" });
    return { ok: s.scene === "bands" && s.readout.length > 0, detail: `scene=${s.scene} readout=${s.readout.length}` };
  });

  await check("tool:params", async () => {
    const s = await call("atlas_set_params", { params: { nBandsShown: 6 } });
    return { ok: s.params.nBandsShown === 6, detail: `nBandsShown=${s.params.nBandsShown}` };
  });

  await check("tool:material", async () => {
    const s = await call("atlas_material", { id: "Ge" });
    const gap = s.readout.find((r) => r.k === "带隙");
    return { ok: s.material === "Ge" && /0\.6\d\d/.test(gap?.v || ""), detail: `gap=${gap?.v}` };
  });

  await check("tool:time", async () => {
    const s = await call("atlas_time", { value: 0.7 });
    return { ok: s.time && Math.abs(s.time.value - (s.time.min + 0.7 * (s.time.max - s.time.min))) < 1e-6, detail: `t=${s.time?.value?.toFixed?.(2)}` };
  });

  await check("tool:readout", async () => {
    const r = await call("atlas_readout");
    return { ok: Array.isArray(r) && r.length > 0, detail: `${r.length} 行` };
  });

  await check("tool:export", async () => {
    const r = await call("atlas_export", { format: "csv", save: true });
    return { ok: typeof r.csv === "string" && r.csv.includes("dataset:") && r.bytes > 2000, detail: r.bytes + " bytes" };
  });

  await check("tool:cases", async () => {
    const cs = await call("atlas_cases");
    return { ok: Array.isArray(cs) && cs.length >= 12 && !!cs[0].apply, detail: cs.length + " 个算例" };
  });

  await check("tool:solve", async () => {
    const kinds = await call("atlas_solver_kinds");
    const okKinds = Array.isArray(kinds) && kinds.length === 8;
    const sol = await call("atlas_solve", { kind: "bands", spec: { material: "Si" }, params: { nG: 46, mesh: 6 } });
    const okBands = sol.error === null && Number.isFinite(sol.data?.gap) && Math.abs(sol.data.gap - 1.1347) < 1e-3;
    const spin = await call("atlas_solve", { kind: "spin", spec: { kind: "FM", S: 1, J1: 1, D: 0 } });
    const okSpin = spin.error === null && spin.data.gap === 0;
    return { ok: okKinds && okBands && okSpin, detail: `${kinds.length} 域 · bands gap=${sol.data?.gap?.toFixed(3)}` };
  });

  await check("tool:open_case", async () => {
    const r = await call("atlas_open_case", { index: 3 });
    const st = await call("atlas_state");
    return { ok: r.ok && st.scene === "spin", detail: "scene=" + st.scene };
  });

  await check("tool:casebook", async () => {
    const r = await call("atlas_casebook");
    return { ok: r.ok === true };
  });

  await check("tool:hud", async () => {
    const r = await call("atlas_hud", { x: 400, y: 300 });
    return { ok: r.ok === true };
  });

  await check("tool:screenshot", async () => {
    const r = await call("atlas_screenshot", { save: true });
    return { ok: r.image && r.image.length > 20000, detail: `${Math.round((r.image?.length || 0) / 1024)} KB` };
  });
} finally {
  mcp.kill();
  await wait(200);
  if (appProc) appProc.kill();
}

const bad = results.filter((r) => !r.ok);
console.log(`  MCP 服务：${results.length - bad.length}/${results.length} 通过`);
if (bad.length) {
  for (const b of bad) console.error(`  ✗ ${b.name}  ${b.detail}`);
  process.exit(1);
}
