/* quantum-noise.js — 给态矢仿真加退相干（振幅阻尼 + 相位阻尼）
 *
 * 密度矩阵：n ≤ 4 时 16×16，代价可接受。
 * 每个门之后对**每个比特**施加信道：
 *   振幅阻尼 γ = 1 − e^{−t/T₁}
 *   相位阻尼 λ = 1 − e^{−t/T_φ}，1/T_φ = 1/T₂ − 1/(2T₁)
 *
 * 实现要点：单比特算符作用在 ρ 上，等价于对"其余比特固定"的每个 2×2 子块做
 *   B' = Σ_a K_a B K_a†
 * 这样不用手写 16×16 的索引，也不容易错。
 */

import { zeroState } from "./quantum.js";

/* 复数用小工具 */
const C = (re, im = 0) => [re, im];
const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const conj = (a) => [a[0], -a[1]];

/* 2×2 复数矩阵（行优先） */
const mat = (a, b, c, d) => [[a, b], [c, d]];
const m2mul = (A, B) => A.map((row, i) =>
  B[0].map((_, j) => add(mul(row[0], B[0][j]), mul(row[1], B[1][j]))));
const m2dag = (A) => [[conj(A[0][0]), conj(A[1][0])], [conj(A[0][1]), conj(A[1][1])]];

export function gateMatrix(g) {
  const s = 1 / Math.SQRT2;
  switch (g) {
    case "H": return mat(C(s), C(s), C(s), C(-s));
    case "X": return mat(C(0), C(1), C(1), C(0));
    case "Y": return mat(C(0), C(0, -1), C(0, 1), C(0));
    case "Z": return mat(C(1), C(0), C(0), C(-1));
    case "S": return mat(C(1), C(0), C(0), C(0, 1));
    case "T": return mat(C(1), C(0), C(0), C(s, s));
    default: return mat(C(1), C(0), C(0), C(1));
  }
}

const densityFromState = (st) => {
  const d = st.length;
  const rho = Array.from({ length: d }, () => Array.from({ length: d }, () => C(0)));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      rho[i][j] = [st[i][0] * st[j][0] + st[i][1] * st[j][1], st[i][1] * st[j][0] - st[i][0] * st[j][1]];
    }
  }
  return rho;
};

/* 在比特 q 上施加一组 Kraus 算符（K[0]=K0 …），ρ' = Σ K ρ K† */
function applyKraus(rho, n, q, Ks) {
  const dim = 1 << n;
  const mask = 1 << (n - 1 - q);
  const out = Array.from({ length: dim }, () => Array.from({ length: dim }, () => C(0)));
  const others = [];
  for (let i = 0; i < dim; i++) if (!(i & mask)) others.push(i);

  for (const i0 of others) {
    const i1 = i0 | mask;
    for (const j0 of others) {
      const j1 = j0 | mask;
      // 2×2 子块 B = [[ρ(i0,j0), ρ(i0,j1)], [ρ(i1,j0), ρ(i1,j1)]]
      const B = [
        [rho[i0][j0], rho[i0][j1]],
        [rho[i1][j0], rho[i1][j1]],
      ];
      let acc = [[C(0), C(0)], [C(0), C(0)]];
      for (const K of Ks) {
        const t = m2mul(m2mul(K, B), m2dag(K));
        acc = acc.map((row, r) => row.map((v, c) => add(v, t[r][c])));
      }
      out[i0][j0] = acc[0][0];
      out[i0][j1] = acc[0][1];
      out[i1][j0] = acc[1][0];
      out[i1][j1] = acc[1][1];
    }
  }
  return out;
}

const amplitudeDamping = (gamma) => {
  const s = Math.sqrt(Math.max(0, 1 - gamma));
  const g = Math.sqrt(Math.max(0, gamma));
  return [
    mat(C(1), C(0), C(0), C(s)),
    mat(C(0), C(g), C(0), C(0)),
  ];
};

const phaseDamping = (lambda) => {
  const s = Math.sqrt(Math.max(0, 1 - lambda));
  const l = Math.sqrt(Math.max(0, lambda));
  return [
    mat(C(1), C(0), C(0), C(s)),
    mat(C(0), C(0), C(0), C(l)),
  ];
};

/* CNOT：在计算基上交换控制位为 1 的 |x0⟩ 与 |x1⟩ */
function applyCNOT(rho, n, control, target) {
  const dim = 1 << n;
  const cm = 1 << (n - 1 - control);
  const tm = 1 << (n - 1 - target);
  const map = Array.from({ length: dim }, (_, i) => (i & cm ? i ^ tm : i));
  const out = Array.from({ length: dim }, () => Array.from({ length: dim }, () => C(0)));
  for (let r = 0; r < dim; r++) {
    for (let c = 0; c < dim; c++) out[map[r]][map[c]] = rho[r][c];
  }
  return out;
}

/* 态矢版：用于理想参考 */
function apply1Vec(st, n, gate, q) {
  const m = gateMatrix(gate);
  const dim = 1 << n;
  const mask = 1 << (n - 1 - q);
  const out = Array.from({ length: dim }, () => C(0));
  for (let i = 0; i < dim; i++) {
    if (i & mask) continue;
    const j = i | mask;
    out[i] = add(mul(m[0][0], st[i]), mul(m[0][1], st[j]));
    out[j] = add(mul(m[1][0], st[i]), mul(m[1][1], st[j]));
  }
  return out;
}

function cnotVec(st, n, control, target) {
  const dim = 1 << n;
  const cm = 1 << (n - 1 - control);
  const tm = 1 << (n - 1 - target);
  const out = st.slice();
  for (let i = 0; i < dim; i++) {
    if (i & cm) {
      const j = i ^ tm;
      out[i] = st[j];
      out[j] = st[i];
    }
  }
  return out;
}

export function idealState(n, ops) {
  let st = zeroState(n);
  for (const o of ops) {
    if (o.gate === "M") continue;
    if (o.gate === "CNOT") st = cnotVec(st, n, o.control ?? 0, o.qubit);
    else st = apply1Vec(st, n, o.gate, o.qubit ?? 0);
  }
  return st;
}

export function simulateNoisy(n, ops, steps, { T1 = 6, T2 = 2.2, gateTime = 0.02 } = {}) {
  const dim = 1 << n;
  let rho = densityFromState(zeroState(n));
  let ideal = zeroState(n);
  const frames = [{ step: 0, F: 1, purity: 1, concurrence: n === 2 ? 1 : null }];

  for (let s = 0; s < steps; s++) {
    const here = ops.filter((o) => o.step === s);
    // 理想参考
    for (const o of here) {
      if (o.gate === "M") continue;
      if (o.gate === "CNOT") ideal = cnotVec(ideal, n, o.control ?? 0, o.qubit);
      else ideal = apply1Vec(ideal, n, o.gate, o.qubit ?? 0);
    }
    // 含噪：先按门演化，再在**每个比特**上加信道
    for (const o of here) {
      if (o.gate === "M") continue;
      if (o.gate === "CNOT") rho = applyCNOT(rho, n, o.control ?? 0, o.qubit);
      else rho = applyKraus(rho, n, o.qubit ?? 0, [gateMatrix(o.gate)]);
    }
    const gamma = 1 - Math.exp(-gateTime / Math.max(1e-12, T1));
    const tphi = 1 / Math.max(1e-12, 1 / Math.max(1e-12, T2) - 1 / (2 * Math.max(1e-12, T1)));
    const lambda = Math.min(0.999, 1 - Math.exp(-gateTime / Math.max(1e-12, tphi)));
    for (let q = 0; q < n; q++) {
      if (gamma > 1e-9) rho = applyKraus(rho, n, q, amplitudeDamping(gamma));
      if (lambda > 1e-9) rho = applyKraus(rho, n, q, phaseDamping(lambda));
    }
    frames.push({ step: s + 1, F: fidelity(rho, ideal), purity: purity(rho), concurrence: n === 2 ? conc2(rho) : null });
  }
  return frames;
}

/* 保真度 F = ⟨ψ|ρ|ψ⟩ */
export function fidelity(rho, ideal) {
  let re = 0;
  for (let i = 0; i < ideal.length; i++) {
    for (let j = 0; j < ideal.length; j++) {
      const t = mul(conj(ideal[i]), rho[i][j]);
      re += t[0] * ideal[j][0] - t[1] * ideal[j][1];
    }
  }
  return Math.max(0, Math.min(1, re));
}

/* 纯度 Tr(ρ²) */
export function purity(rho) {
  let tr = 0;
  for (let i = 0; i < rho.length; i++) {
    for (let j = 0; j < rho.length; j++) {
      const p = mul(rho[i][j], rho[j][i]);
      tr += p[0];
    }
  }
  return Math.max(0, Math.min(1, tr));
}

/* 2 比特 concurrence（Wootters 的常用近似式） */
export function conc2(rho) {
  const m = (i, j) => Math.hypot(rho[i][j][0], rho[i][j][1]);
  // X 型密度矩阵：C = 2·max(0, |ρ03| − √(ρ11ρ22), |ρ12| − √(ρ00ρ33))
  const c1 = m(0, 3) - Math.sqrt(m(1, 1) * m(2, 2));
  const c2 = m(1, 2) - Math.sqrt(m(0, 0) * m(3, 3));
  return Math.max(0, Math.min(1, 2 * Math.max(c1, c2)));
}
