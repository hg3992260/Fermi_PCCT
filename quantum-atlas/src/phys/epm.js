/* ─────────────────────────────────────────────────────────────
   epm.js — 经验赝势（EPM）能带引擎 + 布里渊区几何
   H_{GG'}(k) = (ħ²/2m)|k+G|² δ_GG' + V(|ΔG|²)·Φ(ΔG)
   · V 用 Ry 表（按 |G|²/(2π/a)² 索引），Φ 是实数结构因子
   · 反演中心规范 ⇒ H 实对称 ⇒ 用 Jacobi 求本征值 + 本征矢
   · 所有结果按材料缓存（一次对角化，多处复用）
   ───────────────────────────────────────────────────────────── */

import { HBAR2_2M, RY, phaseOf } from "./materials.js";

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/* ── 倒格矢 ─────────────────────────────────────────────── */
export function recipVectors(A) {
  const [a1, a2, a3] = A;
  const vol = dot(a1, cross(a2, a3));
  const k = (2 * Math.PI) / vol;
  return [
    cross(a2, a3).map((x) => x * k),
    cross(a3, a1).map((x) => x * k),
    cross(a1, a2).map((x) => x * k),
  ];
}

/* ── Jacobi 本征分解（实对称） ───────────────────────────── */
export function eigSym(Ain, n, sweeps = 8) {
  const A = Float64Array.from(Ain);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    if (off < 1e-18) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-13) continue;
        const app = A[p * n + p], aqq = A[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
          A[p * n + k] = A[k * n + p];
          A[q * n + k] = A[k * n + q];
        }
        A[p * n + p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        A[q * n + q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        A[p * n + q] = 0; A[q * n + p] = 0;
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const ev = new Float64Array(n);
  for (let i = 0; i < n; i++) ev[i] = A[i * n + i];
  // 排序
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => ev[i] - ev[j]);
  const values = order.map((i) => ev[i]);
  const vectors = order.map((i) => {
    const col = new Float64Array(n);
    for (let r = 0; r < n; r++) col[r] = V[r * n + i];
    return col;
  });
  return { values, vectors };
}

/* ── 平面波基 ───────────────────────────────────────────── */
function genG(mat, nTarget) {
  const B = recipVectors(mat.A);
  const scale = mat.a / (2 * Math.PI);
  const all = [];
  let nmax = 6;
  for (let h = -nmax; h <= nmax; h++)
    for (let k = -nmax; k <= nmax; k++)
      for (let l = -nmax; l <= nmax; l++) {
        const G = [0, 1, 2].map((i) => h * B[0][i] + k * B[1][i] + l * B[2][i]);
        const g2 = dot(G, G);
        all.push({ G, g2, miller: [h, k, l] });
      }
  all.sort((a, b) => a.g2 - b.g2);
  const picked = all.slice(0, nTarget);
  const gmax = Math.sqrt(picked[picked.length - 1].g2) * 1.0001;
  return { G: picked.map((p) => p.G), g2: picked.map((p) => p.g2 * scale * scale), gmax, B, scale };
}

/** 形状因子：按 |ΔG|²/(2π/a)² 查表（最近整数壳层） */
function formFactor(mat, dG, scale) {
  const g2 = dot(dG, dG) * scale * scale;
  const n2 = Math.round(g2);
  if (Math.abs(g2 - n2) > 0.08) return 0;
  const v = mat.vtab[n2];
  return v === undefined ? 0 : v * RY;
}

/* ── 单点求解 ───────────────────────────────────────────── */
export function solveAt(mat, basis, k) {
  const n = basis.G.length;
  const H = new Float64Array(n * n);
  const phase = phaseOf(mat);
  const scale = basis.scale;
  for (let i = 0; i < n; i++) {
    const ki = [k[0] + basis.G[i][0], k[1] + basis.G[i][1], k[2] + basis.G[i][2]];
    for (let j = i; j < n; j++) {
      let v = 0;
      if (i === j) {
        v = HBAR2_2M * dot(ki, ki);
      } else {
        const dG = sub(basis.G[i], basis.G[j]);
        v = formFactor(mat, dG, scale) * phase(dG);
      }
      H[i * n + j] = v;
      H[j * n + i] = v;
    }
  }
  return eigSym(H, n, 8);
}

/* ── 高对称路径 ─────────────────────────────────────────── */
const PATHS = {
  fcc: {
    nodes: [
      ["Γ", [0, 0, 0]], ["X", [0, 1, 0]], ["W", [0.5, 1, 0]], ["K", [0.75, 0.75, 0]], ["Γ", [0, 0, 0]],
      ["L", [0.5, 0.5, 0.5]], ["U", [0.25, 1, 0.25]], ["W", [0.5, 1, 0]], ["L", [0.5, 0.5, 0.5]],
      ["K", [0.75, 0.75, 0]], ["U", [0.25, 1, 0.25]], ["X", [0, 1, 0]],
    ],
    step: 0.06,
  },
  bcc: {
    nodes: [["Γ", [0, 0, 0]], ["H", [1, 0, 0]], ["N", [0.5, 0.5, 0]], ["Γ", [0, 0, 0]], ["P", [0.5, 0.5, 0.5]], ["H", [1, 0, 0]]],
    step: 0.07,
  },
  hex: {
    nodes: [["Γ", [0, 0, 0]], ["M", [0.5, 0, 0]], ["K", [2 / 3, 0, 0]], ["Γ", [0, 0, 0]]],
    step: 0.05,
  },
};

export function pathFor(mat) {
  if (mat.proj === "2D" || mat.kind === "hex") {
    const B = recipVectors(mat.A);
    const nodes = [["Γ", [0, 0, 0]], ["M", [0.5, 0.5, 0]], ["K", [2 / 3, 1 / 3, 0]], ["Γ", [0, 0, 0]]];
    const pts = [];
    for (let i = 0; i < nodes.length - 1; i++) {
      const a = nodes[i][1], b = nodes[i + 1][1];
      const seg = Math.max(8, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) * 26));
      for (let s = 0; s < seg; s++) {
        const t = s / seg;
        const f = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0];
        pts.push({ f, label: s === 0 ? nodes[i][0] : null, node: nodes[i][0], kAbs: null, B });
      }
    }
    pts.push({ f: nodes[nodes.length - 1][1], label: nodes[nodes.length - 1][0], node: null, kAbs: null, B });
    return pts;
  }
  const kind = mat.kind === "bcc" ? "bcc" : "fcc";
  const { nodes, step } = PATHS[kind];
  const B = recipVectors(mat.A);
  const scale = (2 * Math.PI) / mat.a;
  const pts = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = nodes[i][1].map((x) => x * scale), b = nodes[i + 1][1].map((x) => x * scale);
    const seg = Math.max(6, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / (step * scale)));
    for (let s = 0; s < seg; s++) {
      const t = s / seg;
      pts.push({
        k: [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
        label: s === 0 ? nodes[i][0] : null,
      });
    }
  }
  const last = nodes[nodes.length - 1];
  pts.push({ k: last[1].map((x) => x * scale), label: last[0] });
  return pts;
}

/* ── 石墨烯 TB ──────────────────────────────────────────── */
function grapheneBands(k) {
  const a = 2.46;
  const d = a / Math.sqrt(3);
  const delta = [0, 1, 2].map((j) => {
    const th = (j * 2 * Math.PI) / 3;
    return [d * Math.cos(th), d * Math.sin(th)];
  });
  let re = 0, im = 0;
  for (const dd of delta) {
    const ph = k[0] * dd[0] + k[1] * dd[1];
    re += Math.cos(ph); im += Math.sin(ph);
  }
  const t = 2.7;
  const m = Math.hypot(re, im);
  return [-t * m, t * m];
}

/* ── 结构计算（缓存） ───────────────────────────────────── */
const CACHE = new Map();

export function structure(mat, { nG = 46, mesh = 8 } = {}) {
  const key = `${mat.id}:${nG}:${mesh}`;
  if (CACHE.has(key)) return CACHE.get(key);

  const t0 = performance.now();
  const result = { mat, nG, mesh };

  if (mat.tb) {
    // 二维 TB
    const pts = pathFor(mat);
    const kdist = [0];
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1].f, b = pts[i].f;
      kdist.push(kdist[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]) * ((2 * Math.PI) / mat.a));
    }
    const E = pts.map((p) => grapheneBands([p.f[0] * ((2 * Math.PI) / mat.a), p.f[1] * ((2 * Math.PI) / mat.a), 0]));
    result.E = E;
    result.kdist = kdist;
    result.pts = pts;
    result.nBands = 2;
    result.nOcc = 1;
    result.isMetal = false;
    result.EF = 0;
    result.zero = 0;
    result.shift = 0;
    result.vbm = 0;
    result.cbm = 0;
    result.gap = 0;
    result.directGap = 0;
    result.gapKind = "狄拉克点";
    result.elapsed = performance.now() - t0;
    CACHE.set(key, result);
    return result;
  }

  const basis = genG(mat, nG);
  result.basis = basis;
  const pts = pathFor(mat);
  const kdist = [0];
  const E = [];
  for (let i = 0; i < pts.length; i++) {
    const k = pts[i].k;
    if (i > 0) {
      const a = pts[i - 1].k;
      kdist.push(kdist[i - 1] + Math.hypot(k[0] - a[0], k[1] - a[1], k[2] - a[2]));
    }
    const { values } = solveAt(mat, basis, k);
    E.push(values);
  }
  result.E = E;
  result.kdist = kdist;
  result.pts = pts;
  result.nBands = basis.G.length;

  // 填充与金属判据：nElec/2 非整数 ⇒ 半满带 ⇒ 金属
  const nHalf = mat.nElec / 2;
  result.isMetal = Math.abs(nHalf - Math.round(nHalf)) > 1e-9;
  result.nOcc = result.isMetal ? Math.ceil(nHalf) : Math.round(nHalf);
  const nB = result.nOcc;

  // 能带极值（沿路径）
  let vbm = -Infinity, cbm = Infinity, vbmAt = 0, cbmAt = 0, directGap = Infinity, directAt = 0;
  for (let i = 0; i < E.length; i++) {
    const v = E[i][nB - 1];
    const c = E[i][nB];
    if (v > vbm) { vbm = v; vbmAt = i; }
    if (c < cbm) { cbm = c; cbmAt = i; }
    if (c - v < directGap) { directGap = c - v; directAt = i; }
  }
  result.vbm = vbm; result.cbm = cbm;
  result.vbmAt = vbmAt; result.cbmAt = cbmAt; result.directAt = directAt;
  result.directGap = Math.max(0, directGap);

  // 费米能（k 网格分位数）
  const grid = kMesh(mat, mesh);
  const ev = [];
  const nUp = Math.min(basis.G.length, nB + 2);
  for (const k of grid) {
    const { values } = solveAt(mat, basis, k);
    for (let n = 0; n < nUp; n++) ev.push(values[n]);
  }
  ev.sort((a, b) => a - b);
  const target = Math.max(0, Math.min(ev.length - 1, Math.round(nHalf * grid.length) - 1));
  result.EF = ev[target];
  result.grid = grid;

  if (result.isMetal) {
    result.gap = 0;
    result.gapKind = "金属";
    result.zero = result.EF;
  } else {
    result.gap = Math.max(0, cbm - vbm);
    result.gapKind = Math.abs(cbmAt - vbmAt) <= 1 ? "直接" : "间接";
    result.zero = vbm;
  }
  result.shift = result.zero;
  result.elapsed = performance.now() - t0;
  CACHE.set(key, result);
  return result;
}

/** 以费米能级（金属）或价带顶（半导体）为零点后的能量 */
export function bandsShifted(res) {
  if (!res._shifted) res._shifted = res.E.map((row) => row.map((e) => e - res.zero));
  return res._shifted;
}

/* ── k 网格与费米能 ────────────────────────────────────── */
export function kMesh(mat, n = 8) {
  const B = recipVectors(mat.A);
  const pts = [];
  // 在倒原胞内均匀取样，再折叠到第一布里渊区（近似：取 |k| 最小的等价点）
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let l = 0; l < n; l++) {
        const f = [i / n, j / n, l / n];
        let k = [0, 1, 2].map((d) => f[0] * B[0][d] + f[1] * B[1][d] + f[2] * B[2][d]);
        // 折叠
        let best = k, bestLen = Infinity;
        for (let m = -1; m <= 1; m++)
          for (let p = -1; p <= 1; p++)
            for (let q = -1; q <= 1; q++) {
              const kk = [k[0] + m * B[0][0] + p * B[1][0] + q * B[2][0],
                          k[1] + m * B[0][1] + p * B[1][1] + q * B[2][1],
                          k[2] + m * B[0][2] + p * B[1][2] + q * B[2][2]];
              const len = dot(kk, kk);
              if (len < bestLen) { bestLen = len; best = kk; }
            }
        pts.push(best);
      }
  return pts;
}

function fermiEnergy(mat, basis, grid, nOcc) {
  const ev = [];
  for (const k of grid) {
    const { values } = solveAt(mat, basis, k);
    ev.push(values[nOcc - 1], values[nOcc]);
  }
  ev.sort((a, b) => a - b);
  return ev[Math.floor(ev.length / 2)];
}

/* ── 布里渊区（倒格子的 Wigner–Seitz 胞） ──────────────── */
export function brillouinZone(mat) {
  if (mat._bz) return mat._bz;
  const B = recipVectors(mat.A);
  const scale = mat.a / (2 * Math.PI);

  // 候选平面：|G| 最小的若干倒格矢的中垂面
  const planes = [];
  let nmax = 3;
  const cand = [];
  for (let h = -nmax; h <= nmax; h++)
    for (let k = -nmax; k <= nmax; k++)
      for (let l = -nmax; l <= nmax; l++) {
        if (!h && !k && !l) continue;
        const G = [0, 1, 2].map((i) => h * B[0][i] + k * B[1][i] + l * B[2][i]);
        cand.push({ n: G.map((x) => x / Math.hypot(...G)), d: Math.hypot(...G) / 2, g: Math.hypot(...G) });
      }
  cand.sort((a, b) => a.g - b.g);
  const keep = cand.slice(0, 14);
  for (const c of keep) planes.push(c);

  // 顶点 = 三个平面交点，且满足所有半空间约束
  const verts = [];
  const P = planes.length;
  for (let i = 0; i < P; i++)
    for (let j = i + 1; j < P; j++)
      for (let k = j + 1; k < P; k++) {
        const n1 = planes[i].n, n2 = planes[j].n, n3 = planes[k].n;
        const det = dot(n1, cross(n2, n3));
        if (Math.abs(det) < 1e-8) continue;
        const p = [0, 1, 2].map((d) => (planes[i].d * cross(n2, n3)[d] + planes[j].d * cross(n3, n1)[d] + planes[k].d * cross(n1, n2)[d]) / det);
        let inside = true;
        for (const pl of planes) if (dot(pl.n, p) > pl.d + 1e-6) { inside = false; break; }
        if (!inside) continue;
        if (!verts.some((v) => Math.hypot(v[0] - p[0], v[1] - p[1], v[2] - p[2]) < 1e-5)) verts.push(p);
      }

  // 每个平面上的顶点 → 多边形（按绕法向的角度排序）
  const faces = [];
  for (const pl of planes) {
    const on = verts.filter((v) => Math.abs(dot(pl.n, v) - pl.d) < 1e-5);
    if (on.length < 3) continue;
    const c = on.reduce((acc, v) => [acc[0] + v[0] / on.length, acc[1] + v[1] / on.length, acc[2] + v[2] / on.length], [0, 0, 0]);
    const e1 = on[0] === c ? [1, 0, 0] : [0, 1, 0];
    let u = cross(pl.n, e1);
    if (Math.hypot(...u) < 1e-6) u = cross(pl.n, [0, 0, 1]);
    u = u.map((x) => x / Math.hypot(...u));
    const w = cross(pl.n, u);
    on.sort((a, b) => {
      const va = sub(a, c), vb = sub(b, c);
      return Math.atan2(dot(va, w), dot(va, u)) - Math.atan2(dot(vb, w), dot(vb, u));
    });
    faces.push({ n: pl.n, d: pl.d, verts: on });
  }

  mat._bz = { faces, verts, scale };
  return mat._bz;
}
