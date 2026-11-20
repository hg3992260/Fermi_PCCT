/* repro.js — 可复现（Phase 5 第一步）
 *
 * 每个 Solution 都带着完整 problem 快照（kind/spec/params/observables）+ solver 版本，
 * 所以可以：
 *   · fingerprint(solution) —— 一个"同样输入"的指纹（FNV-1a，8 位十六进制）
 *   · replay(solution)       —— 从快照重新 solve，并比对数据指纹是否一致
 *
 * 这回答了"换个机器/换个时间，能不能跑出一模一样的结果"。
 */

import { solve } from "./solve.js";

function fnv(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = (h * 0x01000193) >>> 0;
  }
  return ("0000000" + h.toString(16)).slice(-8);
}

/* 输入指纹：同样输入 → 同样指纹 */
export function fingerprint(solution) {
  const core = JSON.stringify({
    problem: solution.problem,
    solver: solution.solver,
    version: solution.version,
  });
  return fnv(core);
}

/* 数据指纹：用来比对两次求解的结果是否逐位一致 */
export function dataFingerprint(solution) {
  return fnv(JSON.stringify(solution.data ?? null));
}

/* 回放：从快照重算，返回 { ok, solution, same } */
export function replay(solution) {
  if (!solution.problem || !solution.problem.kind) {
    return { ok: false, error: "solution 缺少 problem 快照，无法回放" };
  }
  const fresh = solve(solution.problem);
  if (fresh.error !== null) {
    return { ok: false, solution: fresh, same: false, error: fresh.error };
  }
  return { ok: true, solution: fresh, same: dataFingerprint(fresh) === dataFingerprint(solution) };
}
