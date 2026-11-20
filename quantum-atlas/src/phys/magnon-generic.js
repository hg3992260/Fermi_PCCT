/* 通用晶格扩展：把「写死方格子」换成「任意键表 + 任意 q 路径」。
 *
 * 只覆盖共线铁磁（单子晶格）——这是"任意晶格"最干净、物理最直接的子集；
 * AF 双子晶格仍走上面的方格子专路（需要二分划分，泛化代价高，留待下一步）。
 *
 * 约定：
 *   bonds: [{ r: [x,y], J }]   —— 键矢量以最近邻距离为单位（q·r 无量纲，q 单位 π/a）
 *   path:  [{ label, q: [qx,qy] }]  —— 高对称路径节点
 *   J(q) = Σ_b J_b·cos(q·r_b)；J(0) = Σ_b J_b
 *   ω(q) = 2S·(J(0) − J(q)) + 2DS
 */

const PI = Math.PI;

export function jqOf(bonds, q) {
  let s = 0;
  for (const { r, J } of bonds) s += J * Math.cos(q[0] * r[0] + q[1] * r[1]);
  return s;
}

export function j0Of(bonds) {
  let s = 0;
  for (const { r, J } of bonds) s += J;
  return s;
}

export function frequenciesGeneric({ S = 1, D = 0, bonds = [] }, q) {
  const j0 = j0Of(bonds);
  const jq = jqOf(bonds, q);
  const A = 2 * S * (j0 - jq) + 2 * D * S;
  return [Math.sqrt(Math.max(0, A * A))]; // FM 单支，B=0
}

export function qPathGeneric(path, n = 120) {
  const qs = [];
  let dist = 0;
  for (let s = 0; s < path.length - 1; s++) {
    const a = path[s].q;
    const b = path[s + 1].q;
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const cnt = Math.max(8, Math.round((n * len) / (3 * PI)));
    for (let i = 0; i < cnt; i++) {
      const f = i / cnt;
      const q = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
      qs.push({ q, dist, label: i === 0 ? path[s].label : null });
      dist += len / cnt;
    }
  }
  qs.push({ q: path[path.length - 1].q, dist, label: path[path.length - 1].label });
  return { qs };
}

/* 任意晶格的色散，返回与 branches() 同构的 { q, dist, w:[[...]], gap } */
export function branchesGeneric(model, n = 120) {
  const { qs } = qPathGeneric(model.path, n);
  const q = [];
  const dist = [];
  const w = [[]];
  for (const { q: qq, dist: d } of qs) {
    const f = frequenciesGeneric(model, qq);
    q.push(qq);
    dist.push(d);
    w[0].push(f[0]);
  }
  return { q, dist, w, gap: w[0][0] };
}

/* ── 多子晶格（共线 FM）──────────────────────────────────
 * 输入：sublattices=[{S,D},…]（m 个，自旋/各向异性），bonds=[{a,b,r,J}]
 * 线性自旋波矩阵（FM，B=0）：
 *   A_αβ(q) = 2·δ_αβ[Σ_γ S_γ J_αγ(0) + S_α D_α] − 2√(S_α S_β)·J_αβ(q)
 * 频率 = √eig(A)。m=1 退化为 frequenciesGeneric；2 子晶格 FM 退化为 2S[zJ+D±J(q)]。
 */

import { eigSym } from "./epm.js";

function multiMatrix(sublattices, bonds, q) {
  const m = sublattices.length;
  const J0 = Array.from({ length: m }, () => new Float64Array(m));
  const Jq = Array.from({ length: m }, () => new Float64Array(m));
  for (const bd of bonds) {
    const ph = Math.cos(q[0] * bd.r[0] + q[1] * bd.r[1]);
    J0[bd.a][bd.b] += bd.J;
    Jq[bd.a][bd.b] += bd.J * ph;
    if (bd.a !== bd.b) {
      J0[bd.b][bd.a] += bd.J;
      Jq[bd.b][bd.a] += bd.J * ph;
    }
  }
  const A = Array.from({ length: m }, () => new Float64Array(m));
  for (let a = 0; a < m; a++) {
    for (let b = 0; b < m; b++) {
      let val = -2 * Math.sqrt(sublattices[a].S * sublattices[b].S) * Jq[a][b];
      if (a === b) {
        let self = 0;
        for (let g = 0; g < m; g++) self += sublattices[g].S * J0[a][g];
        val += 2 * (self + sublattices[a].S * (sublattices[a].D ?? 0));
      }
      A[a][b] = val;
    }
  }
  return A;
}

export function frequenciesMulti(sublattices, bonds, q) {
  const A = multiMatrix(sublattices, bonds, q);
  const m = sublattices.length;
  // 注意：A.flat() 不会展平 TypedArray（Array.prototype.flat 只对真数组生效），必须手动铺平
  const flat = new Float64Array(m * m);
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) flat[i * m + j] = A[i][j];
  const { values } = eigSym(flat, m, 30);
  // A 已经是"频率矩阵"（不是 ω²），所以频率 = 本征值本身，不再开方
  return values.slice().sort((a, b) => a - b);
}

export function branchesMulti(sublattices, bonds, path, n = 120) {
  const { qs } = qPathGeneric(path, n);
  const m = sublattices.length;
  const q = [];
  const dist = [];
  const w = Array.from({ length: m }, () => []);
  for (const { q: qq, dist: d } of qs) {
    const f = frequenciesMulti(sublattices, bonds, qq);
    q.push(qq);
    dist.push(d);
    for (let b = 0; b < m; b++) w[b].push(f[b] ?? 0);
  }
  return { q, dist, w, gap: w[0][0] };
}
