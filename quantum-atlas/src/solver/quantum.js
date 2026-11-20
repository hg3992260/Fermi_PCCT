/* quantum.js — 「量子态」域求解内核：精确态矢
 * spec: { n, state }  —— n 比特数；state 为 2^n 个 [re, im] 复数振幅
 * data: { probabilities, qsphere, concurrence (n=2 时) }
 */

import { qsphere, concurrence2 } from "../phys/quantum.js";
import { makeSolution } from "./envelope.js";

function parseState(spec) {
  const { n, state } = spec;
  if (!Number.isInteger(n) || n < 1 || n > 14) throw new Error(`n 需为 1..14 的整数（收到 ${n}）`);
  if (!Array.isArray(state) || state.length !== (1 << n)) {
    throw new Error(`state 长度应为 2^n = ${1 << n}（收到 ${state?.length}）`);
  }
  // 归一化
  let norm = 0;
  for (const a of state) norm += a[0] * a[0] + a[1] * a[1];
  if (!(norm > 0)) throw new Error("state 全零，无法归一化");
  const s = Math.sqrt(norm);
  return { n, state: state.map((a) => [a[0] / s, a[1] / s]) };
}

export const quantumSolver = {
  id: "quantum.statevec",
  version: "0.9.0",
  solve(problem) {
    try {
      const { n, state } = parseState(problem.spec);
      const R = problem.params.R ?? 1;
      const probs = state.map((a) => a[0] * a[0] + a[1] * a[1]);
      const nodes = qsphere(state, n, R);

      return makeSolution(problem, {
        solver: this.id, version: this.version,
        data: {
          n,
          state,
          probabilities: probs,
          qsphere: nodes.map((nd) => ({ i: nd.i, bits: nd.bits, p: nd.p, phase: nd.phase })),
          concurrence: n === 2 ? concurrence2(state) : null,
          raw: { state, nodes },   // 完整态矢 + 完整 Q-sphere 节点（含 theta/pos/size），供 3D 场景直接渲染
        },
        meta: { dim: 1 << n },
        provenance: { calc: ["probabilities", "concurrence", "qsphere"], input: ["n", "state"] },
        convergence: { note: "精确态矢；n>14 内存爆（指数墙）" },
      });
    } catch (e) {
      return makeSolution(problem, { solver: this.id, version: this.version, data: null, error: e.message });
    }
  },
};
