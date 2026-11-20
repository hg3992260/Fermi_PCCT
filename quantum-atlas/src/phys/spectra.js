/* ─────────────────────────────────────────────────────────────
   spectra.js — 从能带到介电函数（Kubo / 带间跃迁）
   ε₂(ω) ∝ (1/ω²) Σ_{v,c} |p_cv|² δ(E_c − E_v − ω)
   ε₁(ω) = 1 + (2/π) P∫ ω'ε₂(ω')/(ω'² − ω²) dω'      （Kramers–Kronig）
   金属用 Drude；激子用 Lorentz 振子叠加在 E_g − E_b 处
   ───────────────────────────────────────────────────────────── */

import { solveAt } from "./epm.js";
import { structure } from "./epm.js";
import { irreducibleMesh } from "./kgrid.js";

const HBARC = 197.327; // eV·nm
const EPS0_SCALE = { Si: 11.7, Ge: 16.0, C: 5.7, Na: 1.0, graphene: 2.4 };

const CACHE = new Map();

/** 带间跃迁谱：返回 {w[], eps2[], eps1[], alpha[], Eg} */
export function dielectric(mat, { nG = 26, mesh = 8, gamma = 0.10, wMax = 8, dW = 0.02, T = 300, exciton = 0, sym = true } = {}) {
  const key = `${mat.id}:${nG}:${mesh}:${gamma}:${wMax}:${dW}`;
  if (CACHE.has(key)) return CACHE.get(key);

  const t0 = performance.now();
  const res = structure(mat, { nG: 46, mesh: 4 });
  const basis = res.basis;
  const nOcc = Math.round(mat.nElec / 2);
  // 不可约网格：BZ 归属 + 立方对称折叠；权重归一化后乘回 BZ 点数，保持总幅度可比
  const im = sym ? irreducibleMesh(mat, mesh) : null;
  const grid = im ? im.points.map((p) => p.k) : kGrid(mat, mesh);
  const wgt = im ? im.points.map((p) => p.w * im.inBZ) : grid.map(() => 1);
  const nW = Math.round(wMax / dW) + 1;
  const eps2 = new Float64Array(nW);
  const sig = gamma;

  if (!basis) {
    // 二维 TB（石墨烯）：解析的带间谱近似
    const out = grapheneSpectrum(wMax, dW, gamma);
    CACHE.set(key, out);
    return out;
  }

  // 用较小基组重算（谱只需少数带）
  const small = { ...res, basis: shrinkBasis(basis, nG) };
  const nB = small.basis.G.length;
  const nFull = Math.min(nOcc, nB - 1);

  for (let ik = 0; ik < grid.length; ik++) {
    const k = grid[ik];
    const kw = wgt[ik];
    const { values, vectors } = solveAt(mat, small.basis, k);
    for (let v = 0; v < nFull; v++) {
      const kv = kPlusG(k, small.basis.G, nB);
      for (let c = nFull; c < Math.min(nB, nFull + 10); c++) {
        const dE = values[c] - values[v];
        if (dE <= 0 || dE > wMax + 0.6) continue;
        // |p_cv|² （沿 x 极化，对三个方向平均）
        let p2 = 0;
        for (let d = 0; d < 3; d++) {
          let s = 0;
          for (let i = 0; i < nB; i++) s += vectors[c][i] * kv[i][d] * vectors[v][i];
          p2 += s * s;
        }
        p2 /= 3;
        if (p2 < 1e-8) continue;
        const amp = p2 / (dE * dE);
        const i0 = Math.max(0, Math.round((dE - 3 * sig) / dW));
        const i1 = Math.min(nW - 1, Math.round((dE + 3 * sig) / dW));
        for (let i = i0; i <= i1; i++) {
          const x = (i * dW - dE) / sig;
          eps2[i] += amp * kw * Math.exp(-0.5 * x * x);
        }
      }
    }
  }

  // 粗网格会让 JDOS 出现针状峰，KK 变换后 ε₁ 会出现虚假巨峰 → 先做一次小核平滑
  const smoothed = new Float64Array(nW);
  const K = [0.02, 0.06, 0.12, 0.18, 0.24, 0.18, 0.12, 0.06, 0.02];
  for (let i = 0; i < nW; i++) {
    let s = 0;
    for (let k = -4; k <= 4; k++) {
      const j = Math.min(nW - 1, Math.max(0, i + k));
      s += eps2[j] * K[k + 4];
    }
    smoothed[i] = s;
  }
  eps2.set(smoothed);

  // 兜底：不可约网格若因楔形排序/基组原因拿不到强度（C 出现过），退回全网格重算
  let sumI = 0;
  for (let i = 0; i < nW; i++) sumI += eps2[i];
  if (sym && sumI < 1e-10) {
    const fallback = dielectric(mat, { nG, mesh, gamma, wMax, dW, T, exciton, sym: false });
    CACHE.set(key, fallback);
    return fallback;
  }

  // 归一化：使 ε₁(0) 等于实验 ε∞（把矩阵元与相空间标定到同一把尺）
  const epsInf = EPS0_SCALE[mat.id] ?? 10;
  const w = Array.from({ length: nW }, (_, i) => i * dW);
  let eps1 = kk(w, eps2, 1);
  const target = epsInf;
  let scale = 1;
  if (Math.abs(eps1[0] - 1) > 1e-9) scale = (target - 1) / (eps1[0] - 1);
  for (let i = 0; i < nW; i++) eps2[i] *= scale;
  eps1 = kk(w, eps2, 1);

  // ε₁ 再做一次 5 点平滑：KK 在吸收边附近容易过冲，尖峰会让三维曲线出现垂直跳变
  {
    const K = [0.1, 0.2, 0.4, 0.2, 0.1];
    const sm = new Float64Array(nW);
    for (let i = 0; i < nW; i++) {
      let acc = 0;
      for (let k = -2; k <= 2; k++) acc += (eps1[Math.min(nW - 1, Math.max(0, i + k))] || 0) * K[k + 2];
      sm[i] = acc;
    }
    eps1 = sm;
  }

  const alpha = w.map((ww, i) => {
    const kappa = Math.sqrt(Math.max(0, (Math.hypot(eps1[i], eps2[i]) - eps1[i]) / 2));
    return (2 * ww * kappa) / HBARC; // nm⁻¹
  });

  // 温度：Varshni 刚性平移
  const eg0 = res.gap;
  const varshni = varshniShift(mat, T);
  const out = {
    w, eps2: Array.from(eps2), eps1, alpha,
    Eg: eg0 + varshni,
    Eg0: eg0,
    epsInf: target,
    exciton,
    elapsed: performance.now() - t0,
  };
  CACHE.set(key, out);
  return out;
}

function kGrid(mat, n) {
  const B = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
  ];
  // 简单均匀网格；对 fcc/bcc 足够展示谱形
  const s = (2 * Math.PI) / mat.a;
  const pts = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let l = 0; l < n; l++) {
        pts.push([(i / n) * s * 0.5, (j / n) * s * 0.5, (l / n) * s * 0.5]);
      }
  return pts;
}

function shrinkBasis(basis, nG) {
  return { G: basis.G.slice(0, Math.min(nG, basis.G.length)), scale: basis.scale, g2: basis.g2.slice(0, nG) };
}

function kPlusG(k, G, n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push([k[0] + G[i][0], k[1] + G[i][1], k[2] + G[i][2]]);
  return out;
}

/** Kramers–Kronig：ε₁(ω) = 1 + (2/π) P∫ ω'ε₂(ω')/(ω'²−ω²) dω' */
function kk(w, e2, base = 1) {
  const n = w.length;
  const dw = w[1] - w[0];
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const denom = w[j] * w[j] - w[i] * w[i];
      if (Math.abs(denom) < 1e-9) continue;
      s += (w[j] * e2[j] * dw) / denom;
    }
    out[i] = base + (2 / Math.PI) * s;
  }
  return out;
}

function varshniShift(mat, T) {
  // α 单位 eV/K（Si: 4.55×10⁻⁴, Ge: 4.77×10⁻⁴）
  const P = { Si: [4.55e-4, 816], Ge: [4.77e-4, 235], C: [0, 0], Na: [0, 0], graphene: [0, 0] };
  const [a, b] = P[mat.id] || [0, 0];
  if (!a) return 0;
  return -(a * T * T) / (T + b);
}

/** 石墨烯：近似 ε₂ ∝ 常数（2D 狄拉克，带间跃迁阈值 2|μ|） */
function grapheneSpectrum(wMax, dW, gamma) {
  const nW = Math.round(wMax / dW) + 1;
  const w = Array.from({ length: nW }, (_, i) => i * dW);
  const eps2 = w.map((ww) => {
    const th = 2 * 0.0;
    const x = (ww - th) / gamma;
    return 0.5 * (1 + Math.tanh(x));
  });
  const e2 = Float64Array.from(eps2);
  const eps1 = kk(w, e2, 1);
  const alpha = w.map((ww, i) => {
    const kappa = Math.sqrt(Math.max(0, (Math.hypot(eps1[i], eps2[i]) - eps1[i]) / 2));
    return (2 * ww * kappa) / HBARC;
  });
  return { w, eps2: eps2, eps1, alpha, Eg: 0, Eg0: 0, epsInf: 2.4, elapsed: 0 };
}

/** Drude（金属） */
export function drude(mat, { wp = 8.5, gam = 0.08, wMax = 8, dW = 0.02 } = {}) {
  const nW = Math.round(wMax / dW) + 1;
  const w = Array.from({ length: nW }, (_, i) => i * dW);
  const eps1 = [], eps2 = [], alpha = [];
  for (const ww of w) {
    const d = ww * ww + gam * gam;
    const e1 = 1 - (wp * wp * (ww * ww - gam * gam)) / (d * d) * 0; // 简化：见下
    const re = 1 - (wp * wp) / (ww * ww + gam * gam);
    const im = (wp * wp * gam) / (ww * (ww * ww + gam * gam) || 1e-9);
    eps1.push(re); eps2.push(im);
    const kappa = Math.sqrt(Math.max(0, (Math.hypot(re, im) - re) / 2));
    alpha.push((2 * ww * kappa) / HBARC);
  }
  return { w, eps1, eps2, alpha, Eg: 0, Eg0: 0, epsInf: 1, metal: true };
}
