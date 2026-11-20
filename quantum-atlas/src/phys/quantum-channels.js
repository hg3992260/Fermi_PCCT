/* quantum-channels.js — 任意 Kraus 噪声通道 + 任意门 × 噪声的通用密度矩阵演化
 *
 * 把 quantum-noise.js 的"振幅+相位阻尼"泛化成：
 *   · 命名通道：depolarizing / bitflip / phaseflip / ampdamp / phasedamp
 *   · 自定义通道：任意 2×2 Kraus 算符列表
 *   · 门可以是任意 2×2 复矩阵（与 circuit 求解器的无噪声路径对齐）
 *
 * 返回帧序列 {step, F, purity, concurrence}，与 simulateNoisy 同构。
 */

import { zeroState } from "./quantum.js";

const C = (re, im = 0) => [re, im];
const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const conj = (a) => [a[0], -a[1]];
const mat = (a, b, c, d) => [[a, b], [c, d]];
const m2mul = (A, B) => A.map((row, i) => B[0].map((_, j) => add(mul(row[0], B[0][j]), mul(row[1], B[1][j]))));
const m2dag = (A) => [[conj(A[0][0]), conj(A[1][0])], [conj(A[0][1]), conj(A[1][1])]];

const densityFromState = (st) => {
  const d = st.length;
  const rho = Array.from({ length: d }, () => Array.from({ length: d }, () => C(0)));
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
    rho[i][j] = [st[i][0] * st[j][0] + st[i][1] * st[j][1], st[i][1] * st[j][0] - st[i][0] * st[j][1]];
  }
  return rho;
};

/* 在比特 q 上施加一组 Kraus 算符 ρ' = Σ K ρ K† */
export function applyChannel(rho, n, q, Ks) {
  const dim = 1 << n;
  const mask = 1 << (n - 1 - q);
  const out = Array.from({ length: dim }, () => Array.from({ length: dim }, () => C(0)));
  const others = [];
  for (let i = 0; i < dim; i++) if (!(i & mask)) others.push(i);
  for (const i0 of others) {
    const i1 = i0 | mask;
    for (const j0 of others) {
      const j1 = j0 | mask;
      const B = [[rho[i0][j0], rho[i0][j1]], [rho[i1][j0], rho[i1][j1]]];
      let acc = [[C(0), C(0)], [C(0), C(0)]];
      for (const K of Ks) {
        const t = m2mul(m2mul(K, B), m2dag(K));
        acc = acc.map((row, r) => row.map((v, c) => add(v, t[r][c])));
      }
      out[i0][j0] = acc[0][0]; out[i0][j1] = acc[0][1];
      out[i1][j0] = acc[1][0]; out[i1][j1] = acc[1][1];
    }
  }
  return out;
}

/* ── 命名通道（返回 Kraus 算符数组） ── */
export function depolarizing(p) {
  const r = Math.sqrt(Math.max(0, 1 - 3 * p / 4));
  const e = Math.sqrt(Math.max(0, p / 4));
  return [
    mat(C(r), C(0), C(0), C(r)),
    mat(C(0), C(e), C(e), C(0)),        // X
    mat(C(0), C(0, -e), C(0, e), C(0)), // Y
    mat(C(e), C(0), C(0), C(-e)),       // Z
  ];
}
export function bitflip(p) {
  const r = Math.sqrt(Math.max(0, 1 - p));
  const e = Math.sqrt(Math.max(0, p));
  return [mat(C(r), C(0), C(0), C(r)), mat(C(0), C(e), C(e), C(0))];
}
export function phaseflip(p) {
  const r = Math.sqrt(Math.max(0, 1 - p));
  const e = Math.sqrt(Math.max(0, p));
  return [mat(C(r), C(0), C(0), C(r)), mat(C(e), C(0), C(0), C(-e))];
}
export function ampdamp(gamma) {
  const s = Math.sqrt(Math.max(0, 1 - gamma));
  const g = Math.sqrt(Math.max(0, gamma));
  return [mat(C(1), C(0), C(0), C(s)), mat(C(0), C(g), C(0), C(0))];
}
export function phasedamp(lambda) {
  const s = Math.sqrt(Math.max(0, 1 - lambda));
  const l = Math.sqrt(Math.max(0, lambda));
  return [mat(C(1), C(0), C(0), C(s)), mat(C(0), C(0), C(0), C(l))];
}

/* 任意 2×2 复矩阵作用在态矢上（理想参考用） */
function applyU1Vec(st, n, U, q) {
  const dim = 1 << n;
  const mask = 1 << (n - 1 - q);
  const out = Array.from({ length: dim }, () => C(0));
  for (let i = 0; i < dim; i++) {
    if (i & mask) continue;
    const j = i | mask;
    out[i] = add(mul(U[0][0], st[i]), mul(U[0][1], st[j]));
    out[j] = add(mul(U[1][0], st[i]), mul(U[1][1], st[j]));
  }
  return out;
}

function cnotVec(st, n, control, target) {
  const dim = 1 << n;
  const cm = 1 << (n - 1 - control);
  const tm = 1 << (n - 1 - target);
  const out = st.slice();
  for (let i = 0; i < dim; i++) {
    if (i & cm) { const j = i ^ tm; out[i] = st[j]; out[j] = st[i]; }
  }
  return out;
}

function applyCNOT(rho, n, control, target) {
  const dim = 1 << n;
  const cm = 1 << (n - 1 - control);
  const tm = 1 << (n - 1 - target);
  const map = Array.from({ length: dim }, (_, i) => (i & cm ? i ^ tm : i));
  const out = Array.from({ length: dim }, () => Array.from({ length: dim }, () => C(0)));
  for (let r = 0; r < dim; r++) for (let c = 0; c < dim; c++) out[map[r]][map[c]] = rho[r][c];
  return out;
}

function fidelity(rho, ideal) {
  let re = 0;
  for (let i = 0; i < ideal.length; i++) for (let j = 0; j < ideal.length; j++) {
    const t = mul(conj(ideal[i]), rho[i][j]);
    re += t[0] * ideal[j][0] - t[1] * ideal[j][1];
  }
  return Math.max(0, Math.min(1, re));
}
function purity(rho) {
  let tr = 0;
  for (let i = 0; i < rho.length; i++) for (let j = 0; j < rho.length; j++) tr += mul(rho[i][j], rho[j][i])[0];
  return Math.max(0, Math.min(1, tr));
}
function conc2(rho) {
  const m = (i, j) => Math.hypot(rho[i][j][0], rho[i][j][1]);
  const c1 = m(0, 3) - Math.sqrt(m(1, 1) * m(2, 2));
  const c2 = m(1, 2) - Math.sqrt(m(0, 0) * m(3, 3));
  return Math.max(0, Math.min(1, 2 * Math.max(c1, c2)));
}

/**
 * 通用密度矩阵演化：任意门（2×2 矩阵或 CNOT）+ 任意信道。
 * @param n          比特数
 * @param ops        [{gate:<2×2 矩阵|"CNOT">, qubit, control?, step}]
 * @param steps      步数
 * @param channels   [{q, kraus:[K0,K1,…]}] 每步之后施加
 */
export function simulateChannels(n, ops, steps, { channels = [] } = {}) {
  const dim = 1 << n;
  let rho = densityFromState(zeroState(n));
  let ideal = zeroState(n);
  const frames = [{ step: 0, F: 1, purity: 1, concurrence: n === 2 ? 1 : null }];

  for (let s = 0; s < steps; s++) {
    const here = ops.filter((o) => o.step === s);
    for (const o of here) {
      if (o.gate === "M") continue;
      if (o.gate === "CNOT") { ideal = cnotVec(ideal, n, o.control ?? 0, o.qubit); rho = applyCNOT(rho, n, o.control ?? 0, o.qubit); }
      else { ideal = applyU1Vec(ideal, n, o.gate, o.qubit ?? 0); rho = applyChannel(rho, n, o.qubit ?? 0, [o.gate]); }
    }
    for (const ch of channels) rho = applyChannel(rho, n, ch.q, ch.kraus);
    frames.push({ step: s + 1, F: fidelity(rho, ideal), purity: purity(rho), concurrence: n === 2 ? conc2(rho) : null });
  }
  return frames;
}
