/* solve.js — 统一的求解入口：atlas.solve(problem) → Solution
 *
 * 把八个域的内核注册进 registry，导出 solve()。
 * 这是「编程 API」的第一步：外部脚本 / MCP / 未来的批量扫描都走这里。
 */

import { register, get } from "./registry.js";
import { makeProblem } from "./envelope.js";
import { crystalSolver } from "./crystal.js";
import { bandsSolver } from "./bands.js";
import { spectraSolver } from "./spectra.js";
import { spinSolver } from "./spin.js";
import { liquidSolver } from "./liquid.js";
import { quantumSolver } from "./quantum.js";
import { circuitSolver } from "./circuit.js";
import { deviceSolver } from "./device.js";

for (const [kind, solver] of [
  ["crystal", crystalSolver],
  ["bands", bandsSolver],
  ["spectra", spectraSolver],
  ["spin", spinSolver],
  ["liquid", liquidSolver],
  ["quantum", quantumSolver],
  ["circuit", circuitSolver],
  ["device", deviceSolver],
]) register(kind, solver);

export function solve(problem) {
  const p = problem.id ? problem : makeProblem(problem);
  const solver = get(p.kind);
  if (!solver) {
    const err = new Error(`没有注册 "${p.kind}" 的求解器（已注册：${kinds().join(", ")}）`);
    err.kind = "unregistered";
    throw err;
  }
  return solver.solve(p);
}

export {
  crystalSolver, bandsSolver, spectraSolver, spinSolver,
  liquidSolver, quantumSolver, circuitSolver, deviceSolver,
};
export { register, get, kinds, list } from "./registry.js";
export { makeProblem, makeSolution, validateProblem, KINDS } from "./envelope.js";
