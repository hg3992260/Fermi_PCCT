/* sweep.js — 参数网格批量（Phase 3 第二步 + 第三步：非阻塞）
 *
 * sweep(problem, { path, values, extract })
 *   —— 同步，把每个值算一遍，返回 { kind, path, values, solutions, table }。
 *
 * sweepAsync(problem, { path, values, extract, onProgress })
 *   —— 异步，每算完一个点就让出事件循环（yield 一次），onProgress(i, sol) 逐个回报。
 *   这样整条批量不会把 GUI 主线程冻结成一整块；每个单点仍是同步算（μs~ms 级），
 *   真·多核/WASM 卸载是下一步（见路线图 §5.10）。
 */

import { solve } from "./solve.js";
import { makeProblem } from "./envelope.js";

const DEFAULT_EXTRACT = {
  bands: (s) => s.data.gap,
  spectra: (s) => (s.data.Eg != null ? s.data.Eg : s.data.gap),
  spin: (s) => Math.max(...s.data.w.flat()),
  crystal: (s) => s.data.bzFaces,
  liquid: (s) => Math.max(...s.data.g),
  quantum: (s) => (s.data.concurrence != null ? s.data.concurrence : Math.max(...s.data.probabilities)),
  circuit: (s) => Math.max(...s.data.final),
  device: (s) => s.data.scalars.cceAvg,
};

function setPath(obj, path, value) {
  const parts = path.split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (!cur[parts[i]] || typeof cur[parts[i]] !== "object") cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
  return obj;
}

function prepare(problem, { path, values, extract }) {
  if (!path || !Array.isArray(values)) throw new Error("sweep 需要 path 与 values 数组");
  const base = problem.id ? problem : makeProblem(problem);
  const fn = extract ?? DEFAULT_EXTRACT[base.kind];
  const key = path.split(".").pop();
  return { base, fn, key, path, values };
}

function runPoint(base, path, fn, key, v) {
  const p = structuredClone(base);
  setPath(p, path, v);
  const sol = solve(p);
  const y = sol.error === null && fn ? fn(sol) : null;
  return { sol, row: { [key]: v, value: y, error: sol.error } };
}

const yieldTurn = () => new Promise((r) => setTimeout(r, 0));

export function sweep(problem, opts = {}) {
  const { base, fn, key, path, values } = prepare(problem, opts);
  const solutions = [];
  const table = [];
  for (const v of values) {
    const { sol, row } = runPoint(base, path, fn, key, v);
    solutions.push(sol);
    table.push(row);
  }
  return { kind: base.kind, path, values, solutions, table };
}

export async function sweepAsync(problem, opts = {}) {
  const { base, fn, key, path, values } = prepare(problem, opts);
  const onProgress = opts.onProgress;
  const solutions = [];
  const table = [];
  for (let i = 0; i < values.length; i++) {
    const { sol, row } = runPoint(base, path, fn, key, values[i]);
    solutions.push(sol);
    table.push(row);
    if (onProgress) onProgress(i + 1, sol, row);
    await yieldTurn();           // 让出事件循环：GUI 不冻成一整块
  }
  return { kind: base.kind, path, values, solutions, table };
}
