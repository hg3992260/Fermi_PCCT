/* ═══════════════════════════════════════════════════════════════
   device.js — 半导体探测器物理内核
   移植自 Fermi (PyQt) 的 physics.py / simulator.py / monte_carlo.py /
   field_solver.py，公式与量纲保持一致（CGS 混合单位，与原文相同）。
   ═══════════════════════════════════════════════════════════════ */

export const KB = 8.617e-5;   // eV/K
export const QE = 1.602e-19;  // C

/** K 边能量 (keV)，与原文 K_EDGE_ENERGIES 一致 */
export const K_EDGES = {
  Si: 1.84, Ge: 11.1, Ga: 10.37, As: 11.87, Cd: 26.71, Te: 31.81,
  Zn: 9.66, Se: 12.66, Hg: 83.10, I: 33.17, Tl: 85.53, Pb: 88.0, Br: 13.47,
};

/** 探测器材料库：与 data/materials.json 对齐，并补上极化/噪声所需的参数 */
export const DETECTORS = {
  Si: { id: "Si", name: "Si", Eg: 1.12, mu_e: 1400, mu_h: 450, tau_e: 1e-3, tau_h: 1e-3, W: 3.6, rho_m: 2.33, comp: { Si: 1 }, pol: 0.02, tau_pol: 6 },
  Ge: { id: "Ge", name: "Ge", Eg: 0.66, mu_e: 3900, mu_h: 1900, tau_e: 1e-3, tau_h: 1e-3, W: 2.96, rho_m: 5.32, comp: { Ge: 1 }, pol: 0.03, tau_pol: 6 },
  CdTe: { id: "CdTe", name: "CdTe", Eg: 1.44, mu_e: 1100, mu_h: 100, tau_e: 3e-6, tau_h: 1e-6, W: 4.43, rho_m: 5.85, comp: { Cd: 0.5, Te: 0.5 }, pol: 0.18, tau_pol: 14 },
  CZT: { id: "CZT", name: "CZT", Eg: 1.57, mu_e: 1000, mu_h: 80, tau_e: 5e-6, tau_h: 5e-7, W: 4.6, rho_m: 5.78, comp: { Cd: 0.45, Zn: 0.05, Te: 0.5 }, pol: 0.10, tau_pol: 10 },
  TlBr: { id: "TlBr", name: "TlBr", Eg: 2.68, mu_e: 30, mu_h: 4, tau_e: 1e-5, tau_h: 1e-6, W: 6.5, rho_m: 7.56, comp: { Tl: 0.5, Br: 0.5 }, pol: 0.40, tau_pol: 20 },
  HgI2: { id: "HgI2", name: "HgI₂", Eg: 2.13, mu_e: 100, mu_h: 4, tau_e: 1e-6, tau_h: 1e-5, W: 4.2, rho_m: 6.4, comp: { Hg: 0.33, I: 0.67 }, pol: 0.35, tau_pol: 18 },
  GaAs: { id: "GaAs", name: "GaAs", Eg: 1.42, mu_e: 8000, mu_h: 400, tau_e: 1e-8, tau_h: 1e-8, W: 4.2, rho_m: 5.32, comp: { Ga: 0.5, As: 0.5 }, pol: 0.06, tau_pol: 9 },
  Perovskite: { id: "Perovskite", name: "钙钛矿", Eg: 1.5, mu_e: 100, mu_h: 100, tau_e: 1e-5, tau_h: 1e-5, W: 4.5, rho_m: 4.2, comp: { Cs: 0.33, Pb: 0.33, I: 0.34 }, pol: 0.12, tau_pol: 12 },
  "4H-SiC": { id: "4H-SiC", name: "4H-SiC", Eg: 3.26, mu_e: 800, mu_h: 40, tau_e: 1e-8, tau_h: 1e-8, W: 7.6, rho_m: 3.21, comp: { Si: 0.5, C: 0.5 }, pol: 0.02, tau_pol: 7 },
};

/* ── 自定义材料：写 localStorage，等价原 RepositoryDialog 的"手工成分" ── */
const CUSTOM_KEY = "qa:detectors";

export function loadCustomMaterials() {
  try {
    const arr = JSON.parse(localStorage.getItem(CUSTOM_KEY) || "[]");
    for (const d of arr) {
      if (!d || !d.id) continue;
      DETECTORS[d.id] = {
        ...d,
        custom: true,
        tau_e: d.tau_e ?? 1e-6,
        tau_h: d.tau_h ?? 1e-6,
        W: d.W ?? 4.5,
        pol: d.pol ?? 0.1,
        tau_pol: d.tau_pol ?? 12,
        name: d.name || d.id,
      };
    }
  } catch { /* 存储不可用时忽略 */ }
}

export function customMaterials() {
  return Object.values(DETECTORS).filter((d) => d.custom);
}

export function saveCustomMaterial(def) {
  const arr = customMaterials().filter((d) => d.id !== def.id);
  arr.push({ ...def, custom: true });
  localStorage.setItem(CUSTOM_KEY, JSON.stringify(arr));
  DETECTORS[def.id] = { ...def, custom: true };
}

export function removeCustomMaterial(id) {
  const arr = customMaterials().filter((d) => d.id !== id);
  localStorage.setItem(CUSTOM_KEY, JSON.stringify(arr));
  delete DETECTORS[id];
}

loadCustomMaterials();

/* ── 基础半导体量（physics.py） ─────────────────────────── */
export function intrinsicCarriers(Eg, T, Nc300 = 2.2e17, Nv300 = 1.8e18) {
  const g = Math.pow(T / 300, 1.5);
  const Nc = Nc300 * g, Nv = Nv300 * g;
  return Math.sqrt(Nc * Nv) * Math.exp(-Eg / (2 * KB * T));
}

export function fermiLevel(Eg, T, Nc300 = 2.2e17, Nv300 = 1.8e18) {
  return Eg / 2 + (KB * T) / 2 * Math.log(Nv300 / Nc300);
}

export function resistivity(ni, muE, muH) {
  const s = QE * ni * (muE + muH);
  return s > 0 ? 1 / s : Infinity;
}

/** Hecht 方程：z 是距阴极的深度，返回 CCE（0..1） */
export function hecht(z, d, lamE, lamH) {
  const le = Math.max(lamE / d, 1e-9);
  const lh = Math.max(lamH / d, 1e-9);
  const cceE = le * (1 - Math.exp(-(d - z) / (lamE + 1e-12)));
  const cceH = lh * (1 - Math.exp(-z / (lamH + 1e-12)));
  return cceE + cceH;
}

/* ── 衰减系数：两套实现（NIST 表格 / 解析近似） ──────────
   · exact：NIST XCOM 质量衰减系数表（14 种元素，4–600 keV），按 mass fraction 加权，
            组分间做 log–log 插值。这就是 xraylib 的数据来源本身。
   · mock ：原 Fermi 的解析近似（Z⁴·⁵/E³ + 康普顿常数），用来做"快速档"对照。
   说明：浏览器端没有可用的 xraylib WASM 构建，所以精确路径直接用 NIST 原始表；
   若将来有 WASM 版，只需在 providers 里加一个分支，调用点不用改。
   ──────────────────────────────────────────────────────── */
import { NIST } from "./atten-nist.js";

const ATOMIC_WEIGHT = {
  C: 12.011, Si: 28.085, Zn: 65.38, Ga: 69.723, Ge: 72.63, As: 74.922, Br: 79.904,
  Cd: 112.414, Te: 127.6, I: 126.904, Cs: 132.905, Hg: 200.59, Tl: 204.38, Pb: 207.2,
};

/** 原子分数 → 质量分数（NIST 的表是 μ/ρ，必须按质量加权） */
export function massFractions(comp) {
  let tot = 0;
  for (const [el, a] of Object.entries(comp)) tot += a * (ATOMIC_WEIGHT[el] || 100);
  const w = {};
  for (const [el, a] of Object.entries(comp)) w[el] = (a * (ATOMIC_WEIGHT[el] || 100)) / tot;
  return w;
}

/** 单元素 μ/ρ 插值（log E – log μ 线性） */
function muRho(sym, eKeV) {
  const t = NIST[sym];
  if (!t || !t.length) return null;
  const E = Math.max(t[0][0], Math.min(t[t.length - 1][0], eKeV));
  let i = 0;
  while (i < t.length - 1 && t[i + 1][0] < E) i++;
  const [e0, m0] = t[i], [e1, m1] = t[i + 1] || t[i];
  if (e1 === e0) return m0;
  const f = (Math.log(E) - Math.log(e0)) / (Math.log(e1) - Math.log(e0));
  return Math.exp(Math.log(m0) + f * (Math.log(m1) - Math.log(m0)));
}

/** 精确档：NIST 表格 + 质量分数加权，返回线衰减系数 (1/cm) */
export function nistAttenuation(mat, energyKeV) {
  const E = Array.isArray(energyKeV) ? energyKeV : [energyKeV];
  const w = massFractions(mat.comp);
  const out = E.map((e) => {
    let mur = 0, known = 0;
    for (const [el, wi] of Object.entries(w)) {
      const v = muRho(el, e);
      if (v === null) continue;
      mur += wi * v;
      known += wi;
    }
    if (known < 0.999) {
      // 有元素缺表（例如自定义材料里的陌生元素）→ 用近似补齐
      const rest = 1 - known;
      const approx = mockMuRho(mat, e);
      mur += rest * approx;
    }
    return mur * mat.rho_m;
  });
  return Array.isArray(energyKeV) ? out : out[0];
}

/** 近似档：原 Fermi 的 MockXrayLib（Z⁴·⁵/E³ + 康普顿） */
function mockMuRho(mat, e) {
  let mu = 0;
  for (const [el, frac] of Object.entries(mat.comp)) {
    const kEdge = K_EDGES[el] || 0;
    const amp = kEdge > 0 ? 5e3 * Math.pow(kEdge, 2.5) : 1e5;
    let photo = amp * Math.pow(e, -3);
    if (e < kEdge) photo /= 8;
    mu += frac * (photo + 0.15);
  }
  const tot = Object.values(mat.comp).reduce((a, b) => a + b, 0) || 1;
  return mu / tot;
}

/**
 * 统一入口。
 * @param {string} source  "nist"（默认，精确）| "mock"（快速近似）
 */
export function attenuation(mat, energyKeV, source = "nist") {
  if (source === "mock") {
    const E = Array.isArray(energyKeV) ? energyKeV : [energyKeV];
    const out = E.map((e) => mockMuRho(mat, e) * mat.rho_m);
    return Array.isArray(energyKeV) ? out : out[0];
  }
  return nistAttenuation(mat, energyKeV);
}

/* ── 一维泊松场求解（field_solver.py） ─────────────────── */
export function fieldProfile(d, V, opts = {}) {
  const N = opts.N || 120;
  const na = opts.trapped ?? 0;   // 归一化俘获电荷密度（0 → 线性场）
  const eps = 11.0;
  const eps0 = 8.854e-14;
  const z = Array.from({ length: N }, (_, i) => (i / (N - 1)) * d);
  const rho = na > 0
    ? z.map((zz) => -na * 1e-9 * Math.exp(-Math.pow((zz - 0.25 * d) / (0.22 * d), 2)))
    : z.map(() => 0);
  // 解 -d²V/dz² = rho/eps 的离散三对角系统，边界 V(0)=0, V(d)=Vb
  const h = z[1] - z[0];
  const n = N - 2;
  const a = new Float64Array(n), b = new Float64Array(n), c = new Float64Array(n), r = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    a[i] = 1; b[i] = -2; c[i] = 1;
    r[i] = (rho[i + 1] / (eps * eps0)) * h * h;
  }
  r[0] -= 0;              // V(0) = 0
  r[n - 1] -= V;          // V(d) = Vb
  // Thomas 算法
  for (let i = 1; i < n; i++) {
    const w = a[i] / b[i - 1];
    b[i] -= w * c[i - 1];
    r[i] -= w * r[i - 1];
  }
  const Vg = new Float64Array(N);
  Vg[N - 1] = V;
  const x = new Float64Array(n);
  x[n - 1] = r[n - 1] / b[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = (r[i] - c[i] * x[i + 1]) / b[i];
  for (let i = 0; i < n; i++) Vg[i + 1] = x[i];
  const Ef = z.map((_, i) => {
    const i0 = Math.max(0, i - 1), i1 = Math.min(N - 1, i + 1);
    return -(Vg[i1] - Vg[i0]) / ((z[i1] - z[i0]) || h);
  });
  return { z, V: Array.from(Vg), E: Ef };
}

/* ── 器件模型：一次算完所有页需要的量 ──────────────────── */
export function deviceModel(matId, opts = {}) {
  const mat = typeof matId === "object" ? matId : (DETECTORS[matId] || DETECTORS.CZT);
  const thick = opts.thickness ?? 1.0;      // mm
  const V = opts.voltage ?? 300;            // V
  const T = opts.temp ?? 300;               // K
  const Einc = opts.energy ?? 60;           // keV
  const attnSrc = opts.attn ?? "nist";      // "nist" | "mock"
  const d = thick * 0.1;                    // cm
  const E = V / d;                          // V/cm

  const ni = intrinsicCarriers(mat.Eg, T);
  const rho = resistivity(ni, mat.mu_e, mat.mu_h);
  const Ef = fermiLevel(mat.Eg, T);

  const lamE = mat.mu_e * mat.tau_e * E;
  const lamH = mat.mu_h * mat.tau_h * E;

  // 漏电流 / 电子学噪声（1 mm² 像素）
  const area = 0.01;
  const current = (E / rho) * area;             // A
  const encE = Math.sqrt(2 * QE * current * 1e-6) / QE;
  const fwhmElec = (2.355 * encE * mat.W) / 1000; // keV

  // 平均 CCE（沿厚度积分）
  let sum = 0;
  const N = 60;
  for (let i = 0; i < N; i++) sum += hecht(((i + 0.5) / N) * d, d, lamE, lamH);
  const cceAvg = sum / N;

  // 吸收效率（Beer–Lambert）
  const mu = attenuation(mat, Einc, attnSrc);
  const eff = 1 - Math.exp(-mu * d);

  // 能量分辨率：Fano + 电子学
  const sigFano = Math.sqrt(0.11 * Einc * (mat.W / 1000));
  const fwhmTotal = Math.sqrt(Math.pow(2.355 * sigFano, 2) + fwhmElec * fwhmElec);

  const field = fieldProfile(d, V, { trapped: mat.pol > 0.15 ? mat.pol : 0 });

  return { mat, thick, V, T, Einc, d, E, ni, rho, Ef, lamE, lamH, current, encE, fwhmElec, cceAvg, mu, eff, fwhmTotal, field, sigmaFano: sigFano, attnSrc };
}

/* ── 蒙特卡洛：光电吸收 + 康普顿散射（monte_carlo.py 简化版） ── */
export function monteCarlo(matId, opts = {}) {
  const mat = typeof matId === "object" ? matId : (DETECTORS[matId] || DETECTORS.CZT);
  const thick = opts.thickness ?? 1.0;
  const V = opts.voltage ?? 300;
  const Einc = opts.energy ?? 60;
  const n = opts.n ?? 4000;
  const bins = opts.bins ?? 100;
  const d = thick * 0.1;
  const E = V / d;
  const lamE = mat.mu_e * mat.tau_e * E;
  const lamH = mat.mu_h * mat.tau_h * E;
  const mu = attenuation(mat, Einc, opts.attn ?? "nist");

  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  const hist = new Float64Array(bins);
  let interacted = 0;
  const tracks = [];
  for (let i = 0; i < n; i++) {
    // 沿深度抽样相互作用位置
    const z = -Math.log(1 - rnd()) / mu;
    if (z > d) continue;
    interacted++;
    let dep = Einc;
    // 康普顿分支：E_Compton > 100 keV 时概率上升（简化的竞争比）
    const pCompton = Math.min(0.85, Math.max(0.05, (Einc - 30) / 200));
    if (rnd() < pCompton) {
      const cosT = 2 * rnd() - 1;
      const g = Einc / 511;
      const dep2 = Einc / (1 + g * (1 - cosT));
      dep = dep2;                     // 只计首次散射的沉积
    } else {
      // 光电：扣除可能的 K 荧光逃逸
      const edges = Object.keys(mat.comp).map((el) => K_EDGES[el] || 0).filter((e) => e > 0 && e < Einc);
      if (edges.length && rnd() < 0.25) dep -= Math.max(...edges);
    }
    const cce = hecht(z, d, lamE, lamH);
    const meas = Math.max(0, dep * cce);
    const bi = Math.min(bins - 1, Math.floor((meas / (Einc * 1.1)) * bins));
    hist[bi] += 1;
    if (tracks.length < 26) tracks.push({ z, dep, cce, meas });
  }

  // 高斯展宽（Fano + 电子学）
  const model = deviceModel(matId, opts);
  const sig = Math.max(1e-4, Math.sqrt(Math.pow(model.sigmaFano, 2) + Math.pow(model.fwhmElec / 2.355, 2))) / (Einc * 1.1) * bins;
  const out = new Float64Array(bins);
  const rad = Math.max(1, Math.ceil(3 * sig));
  for (let i = 0; i < bins; i++) {
    if (!hist[i]) continue;
    for (let k = -rad; k <= rad; k++) {
      const j = i + k;
      if (j < 0 || j >= bins) continue;
      out[j] += hist[i] * Math.exp(-0.5 * Math.pow(k / sig, 2));
    }
  }
  const mx = Math.max(...out, 1e-9);
  return {
    w: Array.from({ length: bins }, (_, i) => ((i + 0.5) / bins) * Einc * 1.1),
    counts: Array.from(out, (v) => v / mx),
    efficiency: interacted / n,
    tracks,
    photopeak: countPeak(out, bins, Einc, Einc * 1.1),
  };
}

function countPeak(hist, bins, Einc, span) {
  const center = Math.floor((Einc / span) * bins);
  let lo = Math.max(0, center - Math.floor(bins * 0.08));
  let hi = Math.min(bins - 1, center + Math.floor(bins * 0.08));
  let best = lo;
  for (let i = lo; i <= hi; i++) if (hist[i] > hist[best]) best = i;
  return { bin: best, frac: hist[best], energy: ((best + 0.5) / bins) * span };
}

/* ── 温度扫描（t_scan 页） ─────────────────────────────── */
export function tempScan(matId, opts = {}) {
  const out = [];
  for (let T = 200; T <= 500; T += 10) {
    const m = deviceModel(matId, { ...opts, temp: T });
    out.push({ T, fwhm: m.fwhmTotal, dark: m.current * 1e9 });
  }
  return out;
}

/* ── 通量扫描（flux 页） ───────────────────────────────── */
export function fluxScan(matId, opts = {}) {
  const m = deviceModel(matId, opts);
  const out = [];
  for (const f of [0.1, 0.3, 1, 3, 10, 30, 100, 200]) {
    // CCE 展宽随通量增长（陷阱填充）
    const spread = m.mat.pol * Math.min(1, f / 60) * 0.25;
    const fwhmCce = 2.355 * spread * 60;
    const fwhm = Math.sqrt(Math.pow(fwhmCce, 2) + Math.pow(m.fwhmTotal, 2));
    // 视差/堆积：每像素输入率
    const rate = (f * 1e6) * 1e-4;         // Mcps/mm² → Hz/pixel(0.1mm)
    const tau = Math.max(20e-9, (m.d / (Math.min(m.mat.mu_e, m.mat.mu_h) * m.E)) * 0.5);
    const throughput = rate * Math.exp(-rate * tau) / 1e6 * 1; // Mcps/mm²
    out.push({ flux: f, fwhm, throughput });
  }
  return out;
}

/* ── 极化（polarization 页） ───────────────────────────── */
export function polarization(matId, opts = {}) {
  const m = deviceModel(matId, opts);
  const out = [];
  for (let t = 0; t <= 100; t += 2) {
    out.push({ t, f: 1 - m.mat.pol * (1 - Math.exp(-t / m.mat.tau_pol)) });
  }
  return out;
}

/* ── PCCT 综合评分（summary 页） ───────────────────────── */
export function score(matId, opts = {}) {
  const m = deviceModel(matId, opts);
  const eff = m.eff;                                  // 吸收效率
  const res = Math.min(1, 2.5 / Math.max(m.fwhmTotal, 0.05)); // 分辨率（越窄越好）
  const speed = Math.min(1, m.cceAvg * 1.1);          // 收集速度
  const fluxScore = Math.min(1, 1 / (1 + m.mat.pol * 4));
  const s = 0.34 * eff + 0.28 * res + 0.22 * speed + 0.16 * fluxScore;
  return { mat: m.mat, eff, res, speed, fluxScore, score: s, fwhm: m.fwhmTotal, cce: m.cceAvg };
}
