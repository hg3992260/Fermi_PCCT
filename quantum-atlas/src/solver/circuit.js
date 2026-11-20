/* circuit.js — 「量子电路」域求解内核：门序列演化（可含任意噪声）
 *
 * spec: { n, ops, steps = 6, noise = null }
 *   ops 里每个门可以是：
 *     · 命名门        { gate: "H"|"X"|"Y"|"Z"|"S"|"T", qubit, step }
 *     · 参数化旋转    { gate: "Rx"|"Ry"|"Rz", theta, qubit, step }
 *     · 任意 2×2 复矩阵 { gate: [[[re,im],[re,im]],[[re,im],[re,im]]], qubit, step }
 *     · 双比特门      { gate: "CNOT", control, qubit, step }
 *   noise:
 *     · { T1, T2, gateTime }  —— 振幅+相位阻尼（旧接口）
 *     · { channels: [{q, type:"depolarizing"|"bitflip"|"phaseflip"|"ampdamp"|"phasedamp", p} |
 *                      {q, kraus:[[2×2]…]} ] }  —— 任意 Kraus 通道
 */

import { GATES, apply2, simulate } from "../phys/quantum.js";
import { simulateNoisy } from "../phys/quantum-noise.js";
import { simulateChannels, depolarizing, bitflip, phaseflip, ampdamp, phasedamp } from "../phys/quantum-channels.js";
import { makeSolution } from "./envelope.js";

const { cos, sin } = Math;
const C = (re, im = 0) => [re, im];
const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];

export function rotationMatrix(axis, theta) {
  const c = cos(theta / 2), s = sin(theta / 2);
  if (axis === "Rx") return [[C(c), C(0, -s)], [C(0, -s), C(c)]];
  if (axis === "Ry") return [[C(c), C(-s)], [C(s), C(c)]];
  if (axis === "Rz") return [[C(c, -s), C(0)], [C(0), C(c, s)]];
  throw new Error(`rotationMatrix 只支持 Rx/Ry/Rz（收到 "${axis}"）`);
}

function normalizeU(u) {
  if (!Array.isArray(u) || u.length !== 2) throw new Error("2×2 矩阵应是 [[…],[…]]");
  return u.map((row) => {
    if (!Array.isArray(row) || row.length !== 2) throw new Error("2×2 矩阵每行两个元素");
    return row.map((el) => (Array.isArray(el) ? el : [Number(el), 0]));
  });
}

/* 把一个 ops 条目解析成"可作用的门"（2×2 矩阵，或双比特门名） */
function gateOf(op) {
  if (Array.isArray(op.gate)) return normalizeU(op.gate);
  if (op.gate === "Rx" || op.gate === "Ry" || op.gate === "Rz") return rotationMatrix(op.gate, op.theta ?? 0);
  const named = GATES[op.gate];
  if (named) return named;
  return null;
}

function applyU1(state, n, U, target) {
  const out = state.map((a) => [...a]);
  const bit = 1 << (n - 1 - target);
  for (let i = 0; i < state.length; i++) {
    if (i & bit) continue;
    const j = i | bit;
    out[i] = add(mul(U[0][0], state[i]), mul(U[0][1], state[j]));
    out[j] = add(mul(U[1][0], state[i]), mul(U[1][1], state[j]));
  }
  return out;
}

function runTimeline(n, ops, steps) {
  let st = Array.from({ length: 1 << n }, (_, i) => (i === 0 ? [1, 0] : [0, 0]));
  const timeline = [st];
  for (let s = 0; s < steps; s++) {
    for (const o of ops.filter((x) => x.step === s)) {
      if (o.gate === "M") continue;
      if (["CNOT", "CZ", "SWAP"].includes(o.gate)) { st = apply2(st, n, o.gate, o.control ?? 0, o.qubit); continue; }
      const U = gateOf(o);
      if (!U) throw new Error(`未知门："${typeof o.gate === "string" ? o.gate : "(矩阵)"}"`);
      st = applyU1(st, n, U, o.qubit ?? 0);
    }
    timeline.push(st.map((a) => [...a]));
  }
  return timeline;
}

const probsOf = (st) => st.map((a) => a[0] * a[0] + a[1] * a[1]);

/* 解析噪声 → 通用信道列表 [{q, kraus}]，以及理想参考用的 resolvedOps */
function resolveNoise(n, ops, noise) {
  const channels = [];
  if (!noise) return { channels, resolvedOps: ops };

  // 旧接口：T1/T2/gateTime → 每个比特的振幅+相位阻尼
  if (noise.T1 !== undefined || noise.T2 !== undefined) {
    const { T1 = 6, T2 = 2.2, gateTime = 0.02 } = noise;
    const gamma = 1 - Math.exp(-gateTime / Math.max(1e-12, T1));
    const tphi = 1 / Math.max(1e-12, 1 / Math.max(1e-12, T2) - 1 / (2 * Math.max(1e-12, T1)));
    const lambda = Math.min(0.999, 1 - Math.exp(-gateTime / Math.max(1e-12, tphi)));
    for (let q = 0; q < n; q++) {
      if (gamma > 1e-9) channels.push({ q, kraus: ampdamp(gamma) });
      if (lambda > 1e-9) channels.push({ q, kraus: phasedamp(lambda) });
    }
    return { channels, resolvedOps: ops };
  }

  // 通用信道
  if (Array.isArray(noise.channels)) {
    const named = { depolarizing, bitflip, phaseflip, ampdamp, phasedamp };
    for (const ch of noise.channels) {
      if (ch.kraus) {
        const Ks = ch.kraus.map((k) => normalizeU(k));
        channels.push({ q: ch.q ?? 0, kraus: Ks });
      } else if (ch.type && named[ch.type]) {
        channels.push({ q: ch.q ?? 0, kraus: named[ch.type](ch.p ?? 0) });
      } else {
        throw new Error(`未知噪声通道："${ch.type || ch.kraus}"`);
      }
    }
  }
  return { channels, resolvedOps: ops };
}

/* 把 ops 里的门解析成 2×2 矩阵（CNOT 保留），供密度矩阵演化用 */
function resolveOps(n, ops) {
  return ops.map((o) => {
    if (o.gate === "M" || o.gate === "CNOT") return o;
    if (["CZ", "SWAP"].includes(o.gate)) throw new Error(`噪声路径暂只支持 CNOT + 单比特门（收到 ${o.gate}）`);
    const U = gateOf(o);
    if (!U) throw new Error(`未知门："${typeof o.gate === "string" ? o.gate : "(矩阵)"}"`);
    return { ...o, gate: U };
  });
}

export const circuitSolver = {
  id: "circuit.statevec",
  version: "0.9.2",

  solve(problem) {
    try {
      const { n, ops = [], steps = 6, noise = null } = problem.spec;
      if (!Number.isInteger(n) || n < 1 || n > 14) throw new Error(`n 需为 1..14（收到 ${n}）`);
      if (!Array.isArray(ops)) throw new Error("ops 必须是门数组");

      if (noise) {
        const resolved = resolveOps(n, ops);
        const { channels } = resolveNoise(n, resolved, noise);
        const frames = simulateChannels(n, resolved, steps, { channels });
        return makeSolution(problem, {
          solver: this.id, version: this.version,
          data: { n, steps, noise, frames: frames.map((f) => ({ step: f.step, F: f.F, purity: f.purity, concurrence: f.concurrence })) },
          meta: { dim: 1 << n, gates: ops.length, noisy: true, channels: channels.length },
          provenance: { calc: ["F", "purity", "concurrence"], input: ["ops", "noise"] },
          convergence: { note: `密度矩阵 ${1 << n}×${1 << n}；读出误差未建模` },
        });
      }

      const timeline = runTimeline(n, ops, steps);
      const probs = timeline.map(probsOf);
      return makeSolution(problem, {
        solver: this.id, version: this.version,
        data: { n, steps, timeline: probs, final: probs[probs.length - 1], raw: { timeline } },
        meta: { dim: 1 << n, gates: ops.length, noisy: false },
        provenance: { calc: ["timeline"], input: ["ops"] },
        convergence: { note: "理想态矢，无噪声" },
      });
    } catch (e) {
      return makeSolution(problem, { solver: this.id, version: this.version, data: null, error: e.message });
    }
  },
};
