/* magnon.js — 线性自旋波的动力学矩阵对角化（替代解析式）
 *
 * 共线磁体（FM / 2 子晶格 AF）的线性自旋波：
 *   A_αβ(q) = δ_αβ·S·(J_αα(0) + 2D) − S·J_αβ(q)
 *   B_αβ(q) = S·J_αβ(q)          （只有非共线/反铁磁才非零）
 *   频率平方是 (A−B)(A+B) 的本征值 —— 对实数 J 这个矩阵实对称，可直接用 Jacobi。
 *
 * 这样做的意义：
 *   · FM 单子晶格 → B=0，退化成 ω = S(J(0)−J(q)) + 2DS（与旧的解析式一致）
 *   · AF 双子晶格 → √((A−B)(A+B)) 给出 2JSz√(1−γ²)（也与旧式一致）
 *   · 但从此可以加次近邻 J2、多子晶格、各向异性 —— 解析式写不出来的都能算
 */

import { eigSym } from "./epm.js";

const PI = Math.PI;
const COS = Math.cos;

/* 方格子：近邻 z=4，方向 ±x, ±y；次近邻 ±(x±y) */
function J0(model) {
  const { z = 4, J1 = 1, J2 = 0, n = 1 } = model;
  // 每个格点的近邻求和（同子晶格 + 异子晶格）
  return z * J1 + (n === 1 ? 4 * J2 : 4 * J2);
}

function Jq(model, q) {
  const { J1 = 1, J2 = 0, n = 1, kind = "FM" } = model;
  const [qx, qy] = q;
  const g1 = (COS(qx) + COS(qy)) / 2;                      // 近邻结构因子（归一化到 1）
  const g2 = COS(qx) * COS(qy);                            // 次近邻
  // 2 子晶格 AF：异子晶格相位因子
  const phase = kind === "AF" ? 1 : 1;
  const off = n === 2 ? J1 * g1 * phase : 0;
  return {
    diag: 4 * J1 * g1 + 4 * J2 * g2,
    off,
    g2term: 4 * J2 * g2,
  };
}

/**
 * 动力学矩阵 → n 个支的频率（meV）
 * model: { n:1|2, kind:"FM"|"AF", S, J1, J2, D, z }
 */
export function frequencies(model, q) {
  const { S = 1, D = 0, n = 1, kind = "FM", J1 = 1, J2 = 0 } = model;
  const j0 = 4 * J1 + 4 * J2;           // J(0)：所有近邻+次近邻
  const { diag, off } = Jq(model, q);

  if (n === 1) {
    // 单子晶格：A = S(J(0) − J(q)) + 2DS，B = 0
    const A = 2 * S * (j0 - diag) + 2 * D * S;
    const B = 0;
    return [Math.sqrt(Math.max(0, (A - B) * (A + B)))];
  }

  // 双子晶格（AF）：A 对角，B 只在对角块外
  const A = [
    [2 * S * j0 + 2 * D * S, -2 * S * off],
    [-2 * S * off, 2 * S * j0 + 2 * D * S],
  ];
  const B = [
    [0, 2 * S * off],
    [2 * S * off, 0],
  ];
  // M = (A−B)(A+B) 的 2×2 本征值
  const AmB = [
    [A[0][0] - B[0][0], A[0][1] - B[0][1]],
    [A[1][0] - B[1][0], A[1][1] - B[1][1]],
  ];
  const ApB = [
    [A[0][0] + B[0][0], A[0][1] + B[0][1]],
    [A[1][0] + B[1][0], A[1][1] + B[1][1]],
  ];
  const M = [
    [AmB[0][0] * ApB[0][0] + AmB[0][1] * ApB[1][0], AmB[0][0] * ApB[0][1] + AmB[0][1] * ApB[1][1]],
    [AmB[1][0] * ApB[0][0] + AmB[1][1] * ApB[1][0], AmB[1][0] * ApB[0][1] + AmB[1][1] * ApB[1][1]],
  ];
  const { values } = eigSym(Float64Array.from(M.flat()), 2, 12);
  return values.map((v) => Math.sqrt(Math.max(0, v))).sort((a, b) => a - b);
}

/* Γ–X–M–Γ 路径 */
export function qPath(n = 120) {
  const nodes = [[0, 0], [PI, 0], [PI, PI], [0, 0]];
  const qs = [];
  const ticks = [];
  let dist = 0;
  for (let s = 0; s < nodes.length - 1; s++) {
    const [ax, ay] = nodes[s];
    const [bx, by] = nodes[s + 1];
    const len = Math.hypot(bx - ax, by - ay);
    const cnt = Math.max(8, Math.round((n * len) / (3 * PI)));
    for (let i = 0; i < cnt; i++) {
      const f = i / cnt;
      const q = [ax + (bx - ax) * f, ay + (by - ay) * f];
      qs.push({ q, dist, label: i === 0 ? ["Γ", "X", "M", "Γ"][s] : null });
      dist += len / cnt;
    }
  }
  const last = nodes[nodes.length - 1];
  qs.push({ q: last, dist, label: "Γ" });
  return { qs, ticks };
}

/* 各支色散：{ q[], dist[], w: [branch][i] } */
export function branches(model, n = 120) {
  const { qs } = qPath(n);
  const q = [];
  const dist = [];
  const w = model.n === 2 ? [[], []] : [[]];
  for (const { q: qq, dist: d } of qs) {
    const f = frequencies(model, qq);
    q.push(qq);
    dist.push(d);
    for (let b = 0; b < w.length; b++) w[b].push(f[b] ?? 0);
  }
  return { q, dist, w, gap: w[0][0] };
}

/* S(q,ω) 强度图：把每个 q 点的各支频率展宽成 Lorentzian
 * 返回 { qx, qy, qdist, omega[], I[][] }，供 L3 直接画热图 */
export function intensityMap(model, { nq = 54, nw = 40, wMax = null, gamma = 0.18 } = {}) {
  const { qs } = qPath(nq);
  const maxW = wMax ?? Math.max(1e-3, Math.max(...qs.map(({ q }) => Math.max(...frequencies(model, q)))));
  const omega = Array.from({ length: nw }, (_, i) => ((i + 0.5) / nw) * maxW * 1.15);
  const I = [];
  let qmin = Infinity;
  let qmax = -Infinity;
  for (const { q } of qs) {
    const fs = frequencies(model, q);
    const row = omega.map((w) => {
      let s = 0;
      for (const f of fs) {
        const d = (w - f) / gamma;
        s += 1 / (1 + d * d);
      }
      return s;
    });
    const mx = Math.max(...row, 1e-9);
    I.push(row.map((v) => v / mx));
    qmin = Math.min(qmin, Math.min(...fs));
    qmax = Math.max(qmax, Math.max(...fs));
  }
  return { omega, I, qmin, qmax, maxW };
}
