/* magnon-thermal.js — 有限温度磁振子（Phase 4 第一步）
 *
 * 两个教科书级的量：
 *   1. Bose 占据 n(ω) = 1/(e^{ħω/kT} − 1)  —— 量子统计，ω 以 meV、T 以 K 计，kT = 0.08617·T meV
 *   2. 平均场磁化 m(T) —— 自洽布里渊函数（复用 spin.js 的 brillouin）
 *
 * 边界：n(ω) 是"给定频率的占据数"，不是"自洽修正后的色散"；m(T) 是平均场不是自旋波自洽。
 * 这两个量把"有限温度"从"唯象展宽"升到"量子统计"，但都不是自洽的有限温自旋波理论。
 */

import { brillouin } from "./spin.js";

const KT = 0.086173;   // meV/K（玻尔兹曼常数）

export function bose(omegaMeV, TK) {
  if (!(omegaMeV > 0) || !(TK > 0)) return 0;   // ω=0 的 Goldstone 零点模式排除（测度为零，避免 ∞）
  const kT = KT * TK;
  const x = omegaMeV / kT;
  if (x < 1e-6) return kT / omegaMeV - 0.5;   // 经典极限（等分定理），防 0/0
  return 1 / (Math.exp(x) - 1);
}

/* 沿一条色散曲线，返回每个频率的 Bose 占据 + 平均占据（跳过 ω=0 零点模式） */
export function occupations(omegas, TK) {
  const valid = omegas.filter((w) => w > 1e-9);
  const n = valid.map((w) => bose(w, TK));
  const avg = valid.length ? n.reduce((a, b) => a + b, 0) / valid.length : 0;
  return { n, avg, skipped: omegas.length - valid.length };
}

/* 平均场磁化（0..1），复用 spin.js 的自洽布里渊函数 */
export function magnetizationT(TK, { S = 1, Tc = 300 } = {}) {
  if (TK <= 0) return 1;
  return brillouin(TK, { S, Tc, g: 2 });
}
