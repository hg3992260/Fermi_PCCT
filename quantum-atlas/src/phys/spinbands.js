/* spinbands.js — 自旋极化与自旋轨道分裂（诚实的"半计算"）
 *
 * 两件事分开做，避免把"拟合"当"计算"：
 *
 * 1. Zeeman 与交换分裂 —— 严格可算：
 *      ΔE_Zeeman = g·μ_B·B            （g=2，μ_B = 0.05788 meV/T）
 *      ΔE_ex     = D_ex               （自旋极化/磁性交换劈裂，输入）
 *    自旋向上/向下各偏移 ±(ΔZeeman + Δex)/2。
 *
 * 2. 自旋轨道分裂 —— 只做"给定量级"的版本：
 *      价带顶三重态按 Δ_SO 劈成 j=3/2（二重）与 j=1/2（单重），劈裂量取文献 Δ_SO。
 *    为什么不做从头算：EPM 的形状因子是标量赝势，没有自旋相关部分；
 *    要真算必须换成 spinor 基组并重新标定整套形状因子。这里明确标成"输入"。
 */

const MU_B = 0.0578838; // meV/T

export const SOC_LIT = { Si: 44, Ge: 290, C: 6, Na: 0, graphene: 0 }; // meV（文献 Δ_SO）

export function splittings({ B = 0, Dex = 0, soc = 0, g = 2 } = {}) {
  const zeeman = g * MU_B * B;        // meV
  const exchange = Dex;                // meV
  return {
    zeeman,
    exchange,
    total: zeeman + exchange,
    soc,                               // 价带三重态的总劈裂（输入）
    socSplitOff: soc * (2 / 3),        // j=3/2 与 j=1/2 的能差 = Δ_SO（按 2/3 分配）
  };
}

/**
 * 给一条能带加自旋分裂：返回 {up, down} 两条
 * bandIndex / nOcc 用来判断"是不是价带顶三重态"
 */
export function spinSplitBand(E, bandIndex, nOcc, opts = {}) {
  const s = splittings(opts);
  const isValenceTop = bandIndex >= nOcc - 3 && bandIndex < nOcc;
  const isConduction = bandIndex >= nOcc;
  const delta = isValenceTop
    ? s.soc / 2 + s.total / 2
    : isConduction
      ? s.total / 2
      : s.total / 2;
  return {
    up: E.map((e) => e + delta),
    down: E.map((e) => e - delta),
    delta,
    // 价带顶三重态里的"分裂单重态"（j=1/2）还要再低 Δ_SO/3
    splitOff: isValenceTop ? s.socSplitOff : 0,
  };
}

/* 自旋极化率：P = (N↑ − N↓)/(N↑ + N↓)，用 Zeeman 移动后的态密度近似 */
export function spinPolarization({ B = 0, Dex = 0, g = 2, dos = 1 } = {}) {
  const d = splittings({ B, Dex, g }).total;
  if (d <= 0) return 0;
  return Math.tanh((d / 2) / 25.85); // 300 K 时 kT ≈ 25.85 meV
}

export const zeemanAt = (B, g = 2) => g * MU_B * B;
