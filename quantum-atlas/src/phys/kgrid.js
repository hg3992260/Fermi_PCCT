/* kgrid.js — 不可约 k 网格（立方对称 + BZ 归属）
 *
 * 旧实现是在 [0, π/a)³ 里均匀撒点（一个盒子，不是 BZ），没有权重也没有对称约化。
 * 这里做两件事：
 *   1. 用 Wigner–Seitz 判定把点限制在**第一布里渊区**内（|k·G| ≤ |G|²/2 对所有 G）
 *   2. 按立方群 48 个操作把点折叠到**不可约楔**（|kx| ≥ |ky| ≥ |kz| ≥ 0 之一支），并给出权重
 *
 * 权重 = 该点在 48 个操作下落在 BZ 内的像的个数 / 网格总点数归一化。
 * 这样同样的采样密度只需 ~1/48 的点，省下的预算可以拿去加密网格。
 */

import { recipVectors } from "./epm.js";

const PI = Math.PI;

/* 48 个立方群操作：坐标的符号翻转 × 6 个轴置换 */
const SIGNS = [];
for (let sx = -1; sx <= 1; sx += 2)
  for (let sy = -1; sy <= 1; sy += 2)
    for (let sz = -1; sz <= 1; sz += 2) SIGNS.push([sx, sy, sz]);
const PERMS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
const OPS = [];
for (const p of PERMS) for (const s of SIGNS) OPS.push({ p, s });

/* 第一 BZ 的归属判定：对最近邻壳层的所有 G，要求 |k·G| ≤ |G|²/2 */
function makeBZTest(mat) {
  const B = recipVectors(mat.A);
  const shells = [];
  const nmax = 2;
  for (let h = -nmax; h <= nmax; h++)
    for (let k = -nmax; k <= nmax; k++)
      for (let l = -nmax; l <= nmax; l++) {
        if (!h && !k && !l) continue;
        const G = [0, 1, 2].map((i) => h * B[0][i] + k * B[1][i] + l * B[2][i]);
        shells.push(G);
      }
  // 按 |G| 取前两层，足够界定 WS 胞
  shells.sort((a, b) => (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) - (b[0] ** 2 + b[1] ** 2 + b[2] ** 2));
  const keep = shells.slice(0, 14);
  return (k) => {
    for (const G of keep) {
      const kd = k[0] * G[0] + k[1] * G[1] + k[2] * G[2];
      const g2 = (G[0] ** 2 + G[1] ** 2 + G[2] ** 2) / 2;
      if (Math.abs(kd) > g2 + 1e-9) return false;
    }
    return true;
  };
}

/* 折叠到不可约楔：返回规范化的 |k| 三元组（降序） */
function toWedge(k) {
  const a = k.map((x) => Math.abs(x)).sort((x, y) => y - x);
  return a;
}

/**
 * 不可约网格
 * @param mat 材料（需要 A）
 * @param n   线性网格数（每轴 n 份，覆盖 [-π/a, π/a)³ 的立方胞）
 * @returns { points: [{k:[x,y,z], w}], total, reduced }
 */
export function irreducibleMesh(mat, n) {
  const s = (2 * PI) / mat.a;
  const inBZ = makeBZTest(mat);
  const seen = new Map();
  let total = 0;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let l = 0; l < n; l++) {
        const k = [((i + 0.5) / n - 0.5) * s, ((j + 0.5) / n - 0.5) * s, ((l + 0.5) / n - 0.5) * s];
        total++;
        if (!inBZ(k)) continue;
        const w = toWedge(k);
        const key = w.map((x) => x.toFixed(6)).join(",");
        const hit = seen.get(key);
        if (hit) hit.mult++;
        else seen.set(key, { k: [w[0], w[1], w[2]], mult: 1 });
      }
    }
  }

  const pts = [...seen.values()];
  const sumMult = pts.reduce((a, p) => a + p.mult, 0) || 1;
  return {
    points: pts.map((p) => ({ k: p.k, w: p.mult / sumMult })),
    total,
    inBZ: sumMult,
    reduced: pts.length,
  };
}

/* 稀疏网格（不做对称约化，只做 BZ 归属）——用于对比 */
export function plainMesh(mat, n) {
  const s = (2 * PI) / mat.a;
  const inBZ = makeBZTest(mat);
  const pts = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let l = 0; l < n; l++) {
        const k = [((i + 0.5) / n - 0.5) * s, ((j + 0.5) / n - 0.5) * s, ((l + 0.5) / n - 0.5) * s];
        if (inBZ(k)) pts.push({ k, w: 1 / Math.max(1, n * n * n) });
      }
  return pts;
}
