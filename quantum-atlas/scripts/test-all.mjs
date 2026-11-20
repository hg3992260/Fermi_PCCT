/* test-all.mjs — 三层验证一条命令跑完：物理内核 → 浏览器控件 → 真机交互 */
import { spawn } from "node:child_process";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const LAYERS = [
  { name: "物理内核", script: "scripts/test-physics.mjs" },
  { name: "求解器信封", script: "scripts/test-solver.mjs" },
  { name: "收敛性", script: "scripts/test-convergence.mjs" },
  { name: "浏览器控件", script: "scripts/test-audit.mjs" },
  { name: "真机交互", script: "scripts/test-ui.mjs" },
  { name: "MCP 服务", script: "scripts/test-mcp.mjs" },
];

const run = (script) =>
  new Promise((resolve) => {
    const p = spawn(process.execPath, [path.join(ROOT, script)], { cwd: ROOT, stdio: "inherit" });
    p.on("exit", (code) => resolve(code ?? 1));
  });

const results = [];
for (const L of LAYERS) {
  console.log(`\n▸ ${L.name}  (${L.script})`);
  const code = await run(L.script);
  results.push({ ...L, ok: code === 0 });
}

console.log("\n" + "─".repeat(46));
for (const r of results) console.log(`${r.ok ? "✓" : "✗"} ${r.name.padEnd(10)} ${r.script}`);

const failed = results.filter((r) => !r.ok);
if (failed.length) {
  console.error(`\n${failed.length} 层未通过`);
  process.exit(1);
}
console.log("\n六层全部通过");
