/* ─────────────────────────────────────────────────────────────
   quantum.js — n-qubit 态、门、Q-sphere、纠缠
   态矢量用 [re, im] 对偶数组表示；门是 2×2 / 4×4 复矩阵
   ───────────────────────────────────────────────────────────── */

const { sqrt, cos, sin, atan2, PI, hypot, abs } = Math;

/* ── 复数小工具 ─────────────────────────────────────────── */
export const cx = (re, im = 0) => [re, im];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const scale = (a, s) => [a[0] * s, a[1] * s];
export const abs2 = (a) => a[0] * a[0] + a[1] * a[1];
export const phase = (a) => atan2(a[1], a[0]);
const I2 = [[cx(1), cx(0)], [cx(0), cx(1)]];

/* ── 单比特门 ───────────────────────────────────────────── */
export const GATES = {
  H: [[cx(1 / sqrt(2)), cx(1 / sqrt(2))], [cx(1 / sqrt(2)), cx(-1 / sqrt(2))]],
  X: [[cx(0), cx(1)], [cx(1), cx(0)]],
  Y: [[cx(0), cx(0, -1)], [cx(0, 1), cx(0)]],
  Z: [[cx(1), cx(0)], [cx(0), cx(-1)]],
  S: [[cx(1), cx(0)], [cx(0), cx(0, 1)]],
  T: [[cx(1), cx(0)], [cx(0), cx(1 / sqrt(2), 1 / sqrt(2))]],
  "Rx(π/2)": [[cx(cos(PI / 4)), cx(0, -sin(PI / 4))], [cx(0, -sin(PI / 4)), cx(cos(PI / 4))]],
  "Ry(π/2)": [[cx(cos(PI / 4)), cx(-sin(PI / 4))], [cx(sin(PI / 4)), cx(cos(PI / 4))]],
  "Rz(π/2)": [[cx(cos(PI / 4), -sin(PI / 4)), cx(0)], [cx(0), cx(cos(PI / 4), sin(PI / 4))]],
};

export const GATE_LIST = ["H", "X", "Y", "Z", "S", "T", "Rx(π/2)", "Ry(π/2)", "Rz(π/2)", "CNOT", "CZ", "SWAP", "M"];

/* ── 态 ─────────────────────────────────────────────────── */
export function zeroState(n) {
  const dim = 1 << n;
  const v = Array.from({ length: dim }, () => [0, 0]);
  v[0] = [1, 0];
  return v;
}

export function toString(i, n) {
  return i.toString(2).padStart(n, "0");
}

/** 作用单比特门 */
export function apply1(state, n, gate, target) {
  const m = GATES[gate];
  if (!m) return state;
  const out = state.map((a) => [...a]);
  const bit = 1 << (n - 1 - target);
  for (let i = 0; i < state.length; i++) {
    if (i & bit) continue;
    const j = i | bit;
    const a = state[i], b = state[j];
    out[i] = add(mul(m[0][0], a), mul(m[0][1], b));
    out[j] = add(mul(m[1][0], a), mul(m[1][1], b));
  }
  return out;
}

/** 作用两比特门 */
export function apply2(state, n, gate, ctrl, target) {
  const out = state.map((a) => [...a]);
  const bc = 1 << (n - 1 - ctrl);
  const bt = 1 << (n - 1 - target);
  for (let i = 0; i < state.length; i++) {
    if (!(i & bc)) continue;
    if (gate === "CNOT") {
      const j = i ^ bt;
      if (j > i) { out[i] = [...state[j]]; out[j] = [...state[i]]; }
    } else if (gate === "CZ") {
      if (i & bt) out[i] = scale(state[i], -1);
    } else if (gate === "SWAP") {
      const j = i ^ bt;
      if (j > i) { out[i] = [...state[j]]; out[j] = [...state[i]]; }
    }
  }
  return out;
}

/* ── 电路模拟 ───────────────────────────────────────────── */
/**
 * ops: [{gate, qubit, step}, {gate:"CNOT", control, qubit, step}]
 * 返回每步之后的态
 */
export function simulate(n, ops, steps = 8) {
  const timeline = [];
  let st = zeroState(n);
  timeline.push(st);
  for (let s = 0; s < steps; s++) {
    const here = ops.filter((o) => o.step === s);
    for (const o of here) {
      if (o.gate === "M") continue;
      if (["CNOT", "CZ", "SWAP"].includes(o.gate)) st = apply2(st, n, o.gate, o.control ?? 0, o.qubit);
      else st = apply1(st, n, o.gate, o.qubit);
    }
    timeline.push(st);
  }
  return timeline;
}

/* ── Q-sphere ───────────────────────────────────────────── */
export function qsphere(state, n, R = 1) {
  const dim = state.length;
  const amp = state.map(abs2);
  const mx = Math.max(...amp, 1e-9);
  const nodes = [];
  for (let i = 0; i < dim; i++) {
    const p = amp[i];
    const h = countBits(i);
    const theta = (PI * h) / Math.max(1, n);
    const ph = phase(state[i]);
    const rho = 0.32 + 0.68 * Math.sqrt(p / mx);
    const r = R * rho;
    nodes.push({
      i,
      bits: toString(i, n),
      p,
      amp: Math.sqrt(p),
      phase: ph,
      theta,
      pos: [r * sin(theta) * cos(ph), -r * cos(theta), r * sin(theta) * sin(ph)],
      size: 4 + 16 * Math.sqrt(p / mx),
    });
  }
  return nodes;
}

export function countBits(i) {
  let c = 0;
  while (i) { c += i & 1; i >>= 1; }
  return c;
}

/* ── 纠缠与纯度 ─────────────────────────────────────────── */
export function concurrence2(state) {
  // 两比特纯态的 concurrence：C = |<ψ*|σy⊗σy|ψ>|
  const [a, b, c, d] = state;
  // |ψ> = a|00> + b|01> + c|10> + d|11>
  const amp = (z) => Math.sqrt(abs2(z));
  const ph = (z) => phase(z);
  // 振幅形式：C = 2|a d − b c|（含相位）
  const reAD = a[0] * d[0] - a[1] * d[1];
  const imAD = a[0] * d[1] + a[1] * d[0];
  const reBC = b[0] * c[0] - b[1] * c[1];
  const imBC = b[0] * c[1] + b[1] * c[0];
  return 2 * hypot(reAD - reBC, imAD - imBC);
}

export function purity(state) {
  // 纯态 → 1；这里对任意（未归一化）矢量返回归一化后的 Tr ρ²
  const norm = state.reduce((s, a) => s + abs2(a), 0) || 1;
  return 1 / norm;
}

/** 约化密度矩阵（对 n-qubit 态，保留 subset 个比特） */
export function reducedRho(state, n, keep) {
  const dim = 1 << keep;
  const rho = Array.from({ length: dim }, () => Array.from({ length: dim }, () => [0, 0]));
  for (let i = 0; i < state.length; i++) {
    for (let j = 0; j < state.length; j++) {
      const hi = i & ((1 << keep) - 1);
      const hj = j & ((1 << keep) - 1);
      if ((i >> keep) !== (j >> keep)) continue;
      const v = mul(state[i], [state[j][0], -state[j][1]]);
      rho[hi][hj] = add(rho[hi][hj], v);
    }
  }
  return rho;
}

/* ── 预设态 ─────────────────────────────────────────────── */
export const PRESETS = {
  "|00…0⟩": (n) => zeroState(n),
  "Bell (Φ⁺)": (n) => {
    const s = zeroState(n);
    const h = 1 / sqrt(2);
    s[0] = [h, 0];
    s[(1 << n) - 1] = [h, 0];
    return s;
  },
  "GHZ": (n) => {
    const s = zeroState(n);
    const h = 1 / sqrt(2);
    s[0] = [h, 0];
    s[(1 << n) - 1] = [h, 0];
    return s;
  },
  "W": (n) => {
    const s = zeroState(n);
    const a = 1 / sqrt(n);
    for (let q = 0; q < n; q++) {
      const i = 1 << (n - 1 - q);
      s[i] = [a, 0];
    }
    return s;
  },
  "均匀叠加": (n) => {
    const dim = 1 << n;
    const a = 1 / sqrt(dim);
    return Array.from({ length: dim }, () => [a, 0]);
  },
  "随机": (n) => {
    const dim = 1 << n;
    const raw = Array.from({ length: dim }, () => [Math.random() - 0.5, Math.random() - 0.5]);
    const norm = sqrt(raw.reduce((s, a) => s + abs2(a), 0)) || 1;
    return raw.map((a) => [a[0] / norm, a[1] / norm]);
  },
};
