/* index.js — 求解器库的对外出口（Phase 3 第一步）
 *
 * 外部脚本 / MCP / 未来的批处理统一从这里 import：
 *   import { solve, makeProblem, sweep, defineMaterial, defineDetector } from "../src/solver/index.js";
 */

export * from "./solve.js";
export { defineMaterial, resolveMaterial, KINDS as MATERIAL_KINDS } from "./material.js";
export { defineDetector, resolveDetector } from "./detector.js";
export { sweep, sweepAsync } from "./sweep.js";
export { fingerprint, dataFingerprint, replay } from "./repro.js";
export { toQASM, u3Matrix, decomposeU } from "./qasm.js";
export { rotationMatrix } from "./circuit.js";
