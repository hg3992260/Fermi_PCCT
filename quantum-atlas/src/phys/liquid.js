/* ─────────────────────────────────────────────────────────────
   liquid.js — 液态 / 无序体系中的原子自旋
   · 构型：周期边界下的随机顺序吸附（RSA），给出真实 g(r)
   · 结构因子：S(q) = 1 + 4πρ/q ∫ r²(g−1) sin(qr)/(qr) dr
   · 自旋：每个原子带一个旋量 → 箭头（长度=|幅度|，色相=相位）
   · 动力学：S(q,ω) = S(q) · Lorentzian(ω; D q²)（自旋扩散）
   ───────────────────────────────────────────────────────────── */

const { sin, cos, sqrt, exp, PI, atan2, hypot } = Math;

/* ── 构型 ───────────────────────────────────────────────── */
export function liquidConfig({ N = 150, seed = 7, sigma = 3.3, rho = 0.0164 } = {}) {
  let s = seed >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  // 液态 ³He 的数密度 ≈ 1.64×10⁻² Å⁻³（原子体积 46 Å³）
  const L = Math.cbrt(N / rho);
  const dmin = sigma * 0.92;
  const pos = [];
  let guard = 0;
  while (pos.length < N && guard < 600000) {
    guard++;
    const p = [rnd() * L, rnd() * L, rnd() * L];
    let good = true;
    for (const q of pos) {
      let dx = Math.abs(p[0] - q[0]); dx = Math.min(dx, L - dx);
      let dy = Math.abs(p[1] - q[1]); dy = Math.min(dy, L - dy);
      let dz = Math.abs(p[2] - q[2]); dz = Math.min(dz, L - dz);
      if (dx * dx + dy * dy + dz * dz < dmin * dmin) { good = false; break; }
    }
    if (good) pos.push(p);
  }
  // 自旋旋量：随机取向 + 缓慢的空间关联（短程有序）
  const spins = pos.map((p, i) => {
    const ph = 0.8 * (p[0] + p[2]) + rnd() * 2.2;
    const th = 0.9 + rnd() * 0.5;
    return { amp: 0.55 + 0.45 * rnd(), phase: ph, theta: th, i };
  });
  return { pos, spins, L, N: pos.length, seed, rho };
}

/* ── 径向分布函数 ──────────────────────────────────────── */
export function pairCorrelation(cfg, { bins = 90, rMax = 6 } = {}) {
  const { pos, L, N } = cfg;
  const dr = rMax / bins;
  const hist = new Float64Array(bins);
  let pairs = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      let dx = Math.abs(pos[i][0] - pos[j][0]); dx = Math.min(dx, L - dx);
      let dy = Math.abs(pos[i][1] - pos[j][1]); dy = Math.min(dy, L - dy);
      let dz = Math.abs(pos[i][2] - pos[j][2]); dz = Math.min(dz, L - dz);
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const b = Math.floor(r / dr);
      if (b >= 0 && b < bins) { hist[b] += 2; pairs++; }
    }
  }
  const rho = N / (L * L * L);
  const r = [], g = [];
  for (let b = 0; b < bins; b++) {
    const rr = (b + 0.5) * dr;
    const shell = 4 * PI * rr * rr * dr;
    r.push(rr);
    g.push(hist[b] / (N * rho * shell));
  }
  return { r, g, rho, pairs };
}

/* ── 静态结构因子 S(q) ─────────────────────────────────── */
export function structureFactor(pc, { qMax = 6, n = 140 } = {}) {
  const { r, g, rho } = pc;
  const dr = r[1] - r[0];
  const qs = [], S = [];
  for (let i = 1; i <= n; i++) {
    const q = (i / n) * qMax;
    let integ = 0;
    for (let k = 0; k < r.length; k++) {
      const qr = q * r[k];
      const sinc = qr < 1e-6 ? 1 : sin(qr) / qr;
      integ += r[k] * r[k] * (g[k] - 1) * sinc * dr;
    }
    qs.push(q);
    S.push(1 + (4 * PI * rho * integ) / q);
  }
  return { q: qs, S };
}

/* ── 动态结构因子 S(q,ω)：扩散型 ───────────────────────── */
export function dynamicFactor(sf, { D = 1.6, wMax = 3, nw = 70 } = {}) {
  const ws = Array.from({ length: nw }, (_, i) => -wMax + (2 * wMax * i) / (nw - 1));
  const grid = [];
  for (const q of sf.q) {
    const gam = D * q * q;
    const row = ws.map((w) => {
      const L = (1 / PI) * (gam / (w * w + gam * gam));
      return L * (sf.S[sf.q.indexOf(q)] || 1);
    });
    grid.push(row);
  }
  return { ws, grid, D };
}

/* ── 自旋泡利/居里磁化率（演示费米液体 vs 局域矩） ────── */
export function susceptibility(T, { TF = 1.8, theta = -0.5, kind = "fermi" } = {}) {
  if (kind === "fermi") return 1 - (PI * PI / 12) * (T / TF) ** 2; // 归一化 Pauli
  return 1 / Math.max(0.05, T - theta); // Curie–Weiss
}
