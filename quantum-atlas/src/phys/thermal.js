/* thermal.js — 带隙随温度：算一部分、拿一部分做对照
 *
 * 三件事分开做，避免把"拟合"混成"计算"：
 *   1. 刚性离子 EPM：热膨胀 a(T) + 德拜-瓦尔（Debye–Waller）阻尼形状因子
 *      → 真算，但它只包含"晶格几何 + 热抖动"这一部分
 *   2. Varshni：经验公式，参数来自文献 → 标定
 *   3. Fan 项：把缺少的电声自能用一个唯象项补上（含标定系数）→ 半标定
 *
 * 结论会很明显：刚性离子那部分只有几个 meV，而实验的红移是几十 meV。
 * 这个差就是"无电声耦合"这条缺口的量化值，界面上直接展示出来。
 */

import { RY } from "./materials.js";
import { structure } from "./epm.js";

const KB = 1.380649e-23;      // J/K
const HBAR2 = 1.1121e-68;     // J²·s²
const AMU = 1.66053906660e-27; // kg

/* 每种材料的德拜温度、原子质量、线膨胀系数（文献值，输入） */
export const THERMAL = {
  Si: { thetaD: 645, M: 28.085, alpha: 2.6e-6, Eg0: 1.1696, varshni: { a: 4.73e-4, b: 636 } },
  Ge: { thetaD: 360, M: 72.630, alpha: 5.8e-6, Eg0: 0.7437, varshni: { a: 4.77e-4, b: 235 } },
  C: { thetaD: 2230, M: 12.011, alpha: 1.1e-6, Eg0: 5.47, varshni: { a: 0.5e-4, b: 0 } },
  Na: { thetaD: 158, M: 22.990, alpha: 7.1e-6, Eg0: 0, varshni: null },
  graphene: { thetaD: 2100, M: 12.011, alpha: 1.0e-6, Eg0: 0, varshni: null },
};

/* 均方位移 ⟨u²⟩（Å²），德拜模型 */
export function debyeU2(T, { M, thetaD }) {
  if (T <= 0) return 0;
  const mass = M * AMU;
  const pref = (3 * HBAR2) / (mass * KB); // m²·K → 下面换成 Å²
  const prefA2 = pref * 1e20;             // Å²·K
  const x = thetaD / T;
  // ∫₀^x t/(e^t−1) dt，用 40 点梯形（x 可能很大，截断到 60）
  const xc = Math.min(x, 60);
  const n = 60;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const t0 = (xc * i) / n;
    const t1 = (xc * (i + 1)) / n;
    const f = (t) => (t < 1e-9 ? 1 : t / (Math.exp(t) - 1));
    s += 0.5 * (f(t0) + f(t1)) * (t1 - t0);
  }
  const bracket = 0.25 + Math.pow(T / thetaD, 2) * s;
  return (prefA2 / thetaD) * bracket;
}

/* 热膨胀后的晶格常数 */
export function latticeAt(mat, T, T0 = 300) {
  const th = THERMAL[mat.id];
  if (!th) return mat.a;
  return mat.a * (1 + th.alpha * (T - T0));
}

/* 刚性离子 EPM：热膨胀 + 德拜-瓦尔阻尼 */
export function gapRigid(mat, T, { nG = 46, mesh = 6 } = {}) {
  const th = THERMAL[mat.id];
  if (!th) return structure(mat, { nG, mesh }).gap;

  const a = latticeAt(mat, T);
  const u2 = debyeU2(T, th);
  const k0 = (2 * Math.PI) / mat.a;

  // 形状因子按 |G|² 阻尼：V_T(G) = V(G)·exp(−G²⟨u²⟩/6)
  const vtab = {};
  for (const [key, v] of Object.entries(mat.vtab)) {
    const n = Number(key);
    const G2 = n * k0 * k0;
    vtab[key] = v * Math.exp((-G2 * u2) / 6);
  }

  // 缓存 key 含 mat.id，必须换个 id 否则会命中室温的结果
  const hot = { ...mat, id: `${mat.id}@${T}`, a, vtab, _bz: undefined, _struct: undefined };
  const res = structure(hot, { nG, mesh });
  return res.gap;
}

/* Varshni：标定式 */
export function varshniGap(mat, T) {
  const th = THERMAL[mat.id];
  if (!th || !th.varshni) return null;
  const { a, b } = th.varshni;
  return th.Eg0 - (a * T * T) / (T + b || 1);
}

/* 文献值（300 K），仅用于对照 */
export const EXP_GAP_300 = { Si: 1.12, Ge: 0.66, C: 5.47 };

/* 温度扫描：返回 {T[], rigid[], varshni[], fan[]} */
export function scan(mat, { Tmin = 300, Tmax = 800, n = 6 } = {}) {
  const Ts = Array.from({ length: n }, (_, i) => Tmin + ((Tmax - Tmin) * i) / (n - 1));
  const rigid = Ts.map((T) => gapRigid(mat, T));
  const varshni = Ts.map((T) => varshniGap(mat, T));
  // Fan 项：唯象 —— 用 300 K 的刚性值 + 与 Varshni 同形的软化
  return { T: Ts, rigid, varshni };
}
