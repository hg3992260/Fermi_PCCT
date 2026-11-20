/* test-convergence.mjs — 收敛性与标定契约
 *
 * 这个测试的立场：
 *   1. 承认 EPM 的形状因子是"在 nG=46、按实验标定"的 → 断言 46 处的带隙落在实验值 ±30%
 *   2. 承认结果对截断未收敛 → 断言 46→89 的漂移不超过 30%，超了就是回归
 *   3. ε₁(0) 是**标定输入**不是预测 → 断言它严格等于查表 ε∞，防止有人偷偷改掉却不更新文档
 *   4. ε₂ 峰位/峰值对 k 网格敏感 → 断言 mesh 8→12 的峰值变化 ≤ 15%
 *
 * 失败时退出码 1，并打印是哪条契约破了。
 */
import { writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { structure } from "../src/phys/epm.js";
import { MATERIALS, materialOf } from "../src/phys/materials.js";
import { dielectric } from "../src/phys/spectra.js";

const ROOT = path.resolve(import.meta.dirname, "..");
const fail = [];
const rows = [];

const ok = (cond, msg) => {
  if (!cond) fail.push(msg);
  console.log(`  ${cond ? "ok  " : "FAIL"}  ${msg}`);
};

const EXP = { Si: 1.12, Ge: 0.66, C: 5.47 };
const NGS = [46, 65, 89];

console.log("[带隙 / 平面波截断]");
for (const id of ["Si", "Ge", "C"]) {
  const mat = materialOf(id);
  const rec = { id };
  for (const nG of NGS) {
    const t0 = performance.now();
    const res = structure(mat, { nG, mesh: 6 });
    rec[`nG${nG}`] = res.gap;
    rec[`t${nG}`] = Math.round(performance.now() - t0);
    ok(Number.isFinite(res.gap) && res.gap > 0.2, `${id} nG=${nG}: 带隙有限且非简并 (${res.gap.toFixed(4)} eV)`);
  }
  const drift = Math.abs(rec.nG89 - rec.nG46) / rec.nG46;
  rec.drift = drift;
  rec.err46 = Math.abs(rec.nG46 - EXP[id]) / EXP[id];
  ok(rec.err46 <= 0.3, `${id} nG=46 与实验 ${EXP[id]} eV 差 ${(rec.err46 * 100).toFixed(1)}% (≤30%)`);
  ok(drift <= 0.3, `${id} nG 46→89 漂移 ${(drift * 100).toFixed(1)}% (≤30%)`);
  rows.push(rec);
}

console.log("\n[介电函数 / k 网格 与 标定契约]");
const si = materialOf("Si");
const d8 = dielectric(si, { nG: 26, mesh: 8 });
const d12 = dielectric(si, { nG: 26, mesh: 12 });
const peak = (d) => Math.max(...Array.from(d.eps2));
const argPeak = (d) => {
  const arr = Array.from(d.eps2);
  return Array.from(d.w)[arr.indexOf(Math.max(...arr))];
};
const p8 = peak(d8);
const p12 = peak(d12);
const dPeak = Math.abs(p12 - p8) / p8;
ok(dPeak <= 0.15, `Si ε₂ 峰值 mesh 8→12 变化 ${(dPeak * 100).toFixed(1)}% (≤15%)`);
ok(Math.abs(argPeak(d8) - 4.3) < 1.0, `Si ε₂ 峰位 ${argPeak(d8).toFixed(2)} eV 落在 3.3–5.3（E₂ 峰区）`);

const EPS0 = { Si: 11.7, Ge: 16.0, C: 5.7 };
for (const id of ["Si", "Ge", "C"]) {
  const d = dielectric(materialOf(id), { nG: 26, mesh: 6 });
  const e0 = d.eps1[0];
  const rel = Math.abs(e0 - EPS0[id]) / EPS0[id];
  ok(rel < 5e-3, `${id} ε₁(0)=${Number(e0).toFixed(4)} 与标定 ε∞=${EPS0[id]} 差 ${(rel * 100).toFixed(3)}%（契约：锚定量，非预测）`);
}

/* ── 报告 ── */
const lines = [];
lines.push("# 收敛性报告（自动生成）");
lines.push("");
lines.push(`生成时间：${new Date().toISOString()}`);
lines.push("");
lines.push("## 带隙 vs 平面波截断");
lines.push("");
lines.push("| 材料 | nG=46 | nG=65 | nG=89 | 46→89 漂移 | 与实验差 | 实验 |");
lines.push("|---|---|---|---|---|---|---|");
for (const r of rows) {
  lines.push(
    `| ${r.id} | ${r.nG46.toFixed(4)} | ${r.nG65.toFixed(4)} | ${r.nG89.toFixed(4)} | ` +
      `${(r.drift * 100).toFixed(1)}% | ${(r.err46 * 100).toFixed(1)}% | ${EXP[r.id]} |`
  );
}
lines.push("");
lines.push("## 介电函数（Si，nG=26）");
lines.push("");
lines.push(`- ε₂ 峰值：mesh 8 → ${p8.toFixed(2)}，mesh 12 → ${p12.toFixed(2)}（Δ ${(dPeak * 100).toFixed(1)}%）`);
lines.push(`- ε₂ 峰位：${argPeak(d8).toFixed(2)} eV`);
lines.push("- ε₁(0) 贴住查表 ε∞（差 <0.5%，5 点平滑带来的偏差） —— 这是标定输入，任何「ε∞ 算对了」的说法都不成立");
lines.push("");
lines.push(fail.length ? `**结论：${fail.length} 条契约未通过**` : "**结论：全部契约通过**");

mkdirSync(path.join(ROOT, "docs"), { recursive: true });
writeFileSync(path.join(ROOT, "docs", "convergence.md"), lines.join("\n") + "\n", "utf8");

console.log(`\n收敛性：${fail.length ? fail.length + " 项失败" : "全部通过"}（报告 → docs/convergence.md）`);
if (fail.length) {
  for (const f of fail) console.error("  ✗ " + f);
  process.exit(1);
}
