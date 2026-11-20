/* ─────────────────────────────────────────────────────────────
   spin.js — 自旋：单原子 → 局域点阵 → 宏观磁化
   · 单原子：Zeeman + 单轴各向异性 → 能级 E(m) 与进动
   · 点阵：经典海森堡基态（FM/AF/条纹/螺旋/Skyrmion）+ 线性自旋波色散
   · 宏观：Brillouin 函数 M(T)，以及双稳模型给出的磁滞回线
   ───────────────────────────────────────────────────────────── */

import { V3 } from "../engine.js";

const { sin, cos, tanh, atan, exp, sqrt, PI } = Math;

/* ── 单原子能级：E(m) = D m² − g μ_B B m ────────────────── */
export function atomicLevels({ S = 1, D = 0.3, g = 2.0, B = 1.0 }) {
  const muB = 5.7883818e-5; // eV/T
  const out = [];
  for (let m = -S; m <= S; m += 1) out.push({ m, E: D * m * m - g * muB * B * m });
  const E0 = Math.min(...out.map((o) => o.E));
  return out.map((o) => ({ ...o, dE: (o.E - E0) * 1e3 })); // meV
}

/** 经典进动：ω = γB，返回单位矢量在 t 时刻的位置 */
export function precess(theta0, omega, t) {
  const th = theta0;
  const ph = omega * t;
  return [sin(th) * cos(ph), cos(th), sin(th) * sin(ph)];
}

/* ── 点阵基态 ───────────────────────────────────────────── */
/**
 * @param kind "FM" | "AF" | "stripe" | "spiral" | "skyrmion"
 * @returns [{p:[x,y,z], m:[mx,my,mz]}] 归一化磁矩
 */
export function spinLattice({ kind = "FM", N = 9, a = 1, q = 0.55, R = 2.2, t = 0 }) {
  const out = [];
  const c = (N - 1) / 2;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x = (i - c) * a, z = (j - c) * a;
      let m = [0, 1, 0];
      if (kind === "FM") m = [0, 1, 0];
      else if (kind === "AF") m = [(i + j) % 2 ? 0.0 : 0.0, (i + j) % 2 ? -1 : 1, 0];
      else if (kind === "stripe") m = [0, j % 2 ? -1 : 1, 0];
      else if (kind === "spiral") {
        const ph = q * (i * a * 1.0 + j * a * 0.55) + t * 1.2;
        m = [0, cos(ph), sin(ph)];
      } else if (kind === "skyrmion") {
        const r = Math.hypot(x, z);
        const th = 4 * atan(exp(-(r - R) / 0.55)) + t * 0.4;
        const ph = atan(z, x);
        m = [sin(th) * cos(ph), cos(th), sin(th) * sin(ph)];
      }
      out.push({ p: [x, 0, z], m: V3.norm(m) });
    }
  }
  return out;
}

/* ── 线性自旋波 ─────────────────────────────────────────── */
/**
 * FM: ω(q) = 2JSz(1−γ_q) + Δ
 * AF: ω(q) = 2JSz√(1−γ_q²) + Δ ,  γ_q = (cos qx + cos qy)/2
 */
export function magnon({ kind = "FM", J = 1, S = 1, D = 0.05, qMax = PI, n = 120 }) {
  const z = 4;
  const qs = [];
  // 路径 Γ(0,0) → X(π,0) → M(π,π) → Γ
  const nodes = [[0, 0], [PI, 0], [PI, PI], [0, 0]];
  for (let s = 0; s < nodes.length - 1; s++) {
    const [ax, ay] = nodes[s], [bx, by] = nodes[s + 1];
    const cnt = Math.max(10, Math.round(n * Math.hypot(bx - ax, by - ay) / (3 * PI)));
    for (let i = 0; i < cnt; i++) {
      const f = i / cnt;
      qs.push([ax + (bx - ax) * f, ay + (by - ay) * f]);
    }
  }
  qs.push(nodes[nodes.length - 1]);
  const gap = 2 * D * S;
  const w = qs.map(([qx, qy]) => {
    const gam = (cos(qx) + cos(qy)) / 2;
    const base = kind === "AF"
      ? 2 * J * S * z * sqrt(Math.max(0, 1 - gam * gam))
      : 2 * J * S * z * (1 - gam);
    return Math.max(0, base + gap);
  });
  return { q: qs, w, gap };
}

/* ── 宏观磁化 M(T) ──────────────────────────────────────── */
export function brillouin(T, { S = 1, Tc = 300, g = 2 }) {
  if (T <= 0) return 1;
  // 自洽求解 m = B_S(3 Tc m / (T (S+1)))
  const BS = (x) => {
    const s = S;
    return ((2 * s + 1) / (2 * s)) / tanh(((2 * s + 1) / (2 * s)) * x) - (1 / (2 * s)) / tanh(x / (2 * s));
  };
  let m = 0.9;
  for (let it = 0; it < 200; it++) {
    const x = (3 * Tc * m) / (T * (S + 1));
    const mn = Math.min(1, BS(x));
    m = 0.7 * m + 0.3 * mn;
    if (Math.abs(m - mn) < 1e-6) break;
  }
  return T > Tc * 1.02 ? 0 : Math.max(0, Math.min(1, m));
}

/** 磁滞回线：Jiles–Atherton 的简化形式 */
export function hysteresis(H, { Hc = 80, Mr = 0.55, Ms = 1.0, a = 40 } = {}) {
  const M = [];
  let m = -1;
  for (const h of H) {
    const target = tanh((h + Hc * (m > 0 ? 1 : -1)) / a);
    m = 0.25 * m + 0.75 * target;
    M.push(Ms * Math.max(-1, Math.min(1, m)));
  }
  return M;
}

/** 自旋波速（AF，线性区）与磁振子热激发 */
export function spinWaveVelocity({ J = 1, S = 1, a = 1 }) {
  const c = 2 * J * S * a * Math.sqrt(2); // 单位：meV·a/ħ 的示意
  return c;
}
