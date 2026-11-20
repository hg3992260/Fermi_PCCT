/* exciton.js — Wannier–Mott 激子：从能带曲率算有效质量，再解氢原子型方程
 *
 * 链条（每一环都标注来源）：
 *   能带曲率  → m*_e / m*_h          [计算：EPM 对角化]
 *   约化质量  μ = m_e m_h /(m_e+m_h) [计算]
 *   静态介电  ε                       [输入 / 标定：查表 ε∞]
 *   束缚能    E_b = 13.6·μ/(m₀ε²) eV  [计算（依赖上面的 ε 输入）]
 *   玻尔半径  a_B = 0.529·ε m₀/μ  Å   [计算]
 *
 * 这是一个「部分计算」模型：激子波函数用氢原子型（各向同性、单带、无简并），
 * 不含电子-空穴交换、不含各向异性、不含声子展宽。量级可信，绝对值别当真。
 */

import { structure } from "./epm.js";

const M0 = 9.1093837015e-31;   // kg
const HBAR2_2M = 3.80998;      // eV·Å²  (ħ²/2m₀)
const RY_EV = 13.605693;       // eV
const A0 = 0.529177;           // Å

/* 沿 k 路径在极值附近做抛物线拟合，得到曲率 → 有效质量（单位 m₀） */
function massFromCurve(kdist, E, i0, sign) {
  // i0: 极值所在下标；取左右各两个点做最小二乘抛物线
  // 极值可能落在路径端点（如 Γ），此时用单侧点
  const idx = [];
  for (const d of [-2, -1, 1, 2, 3, 4]) {
    const j = i0 + d;
    if (j >= 0 && j < kdist.length) idx.push(j);
  }
  if (idx.length < 3) return null;
  // 以 i0 为原点，E ≈ E0 + a·k²  → a = Σ(k²E)/Σ(k⁴)，m* = ħ²/(2a) 相对 m₀
  let s4 = 0;
  let s2e = 0;
  const k0 = kdist[i0];
  for (const j of idx) {
    const dk = kdist[j] - k0;
    const dE = E[j] - E[i0];
    s4 += dk ** 4;
    s2e += dk * dk * dE;
  }
  if (s4 < 1e-12) return null;
  const a = s2e / s4; // eV·Å²
  if (Math.abs(a) < 1e-6) return null;
  // |a| = ħ²/(2m*) → m*/m0 = HBAR2_2M / |a|
  return HBAR2_2M / Math.abs(a);
}

export function effectiveMasses(mat, { nG = 46, mesh = 6 } = {}) {
  const res = structure(mat, { nG, mesh });
  const E = res.E;
  if (!E || !E.length) return null;
  const nOcc = res.nOcc ?? 0;
  const kd = res.kdist;

  // 价带顶：第 nOcc-1 支的最大值；导带底：第 nOcc 支的最小值
  const vb = nOcc - 1;
  const cb = nOcc;
  if (vb < 0 || cb >= (E[0]?.length ?? 0)) return null;

  let iV = 0;
  let iC = 0;
  for (let i = 0; i < E.length; i++) {
    if (E[i][vb] > E[iV][vb]) iV = i;
    if (E[i][cb] < E[iC][cb]) iC = i;
  }
  const mh = massFromCurve(kd, E.map((r) => r[vb]), iV, +1);
  const me = massFromCurve(kd, E.map((r) => r[cb]), iC, -1);
  return { me, mh, iV, iC, gap: res.gap, gapKind: res.gapKind };
}

/* Wannier–Mott 氢原子型解 */
export function wannier(mat, { eps, nG = 46, mesh = 6 } = {}) {
  const m = effectiveMasses(mat, { nG, mesh });
  if (!m || !m.me || !m.mh) return null;
  const mu = (m.me * m.mh) / (m.me + m.mh); // 单位 m₀
  const eb = (RY_EV * mu) / (eps * eps);    // eV
  const aB = A0 * eps / mu;                 // Å
  const series = [1, 2, 3].map((n) => ({ n, E_meV: (-eb * 1000) / (n * n) }));
  return { me: m.me, mh: m.mh, mu, eps, eb, aB, series, gap: m.gap, gapKind: m.gapKind };
}

/* 吸收边位置：Eg − E_b（1s） */
export function excitonEdge(mat, opts) {
  const w = wannier(mat, opts);
  if (!w) return null;
  return { edge: w.gap - w.eb, eb: w.eb, series: w.series };
}
