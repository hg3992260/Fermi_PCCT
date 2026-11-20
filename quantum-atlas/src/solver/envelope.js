/* envelope.js — 统一的问题→求解信封
 *
 * 这是「通用求解工具」的地基：不管哪个域，问题都用同一个 Problem 描述、
 * 结果都用同一个 Solution 返回。数值内核各域不同，但壳一样——
 * 所以求解器可以注册、可以组合、可以批量、可以被脚本调用。
 *
 *   Problem  = { kind, spec, params, observables, version, id }
 *   Solution = { problemId, kind, solver, version, data, meta, provenance, convergence, error }
 *
 * spec   —— 域相关的模型定义（晶格/基元、自旋模型、门列表…），是"输入"
 * params —— 数值参数（截断、网格、收敛判据…），是"怎么算"
 * observables —— 声明要观测什么（可选，留作批量/选择用）
 */

export const KINDS = ["crystal", "bands", "spectra", "spin", "liquid", "quantum", "circuit", "device"];

const isPlain = (x) => x === null || (typeof x === "object" && !Array.isArray(x));

export function validateProblem(p) {
  const errors = [];
  if (!isPlain(p)) return { ok: false, errors: ["problem 必须是对象"] };
  if (!KINDS.includes(p.kind)) errors.push(`未知 kind："${p.kind}"（应为 ${KINDS.join("/")}）`);
  if (!isPlain(p.spec)) errors.push("spec 必须是对象");
  if (p.params !== undefined && !isPlain(p.params)) errors.push("params 必须是对象");
  if (p.observables !== undefined && !Array.isArray(p.observables)) errors.push("observables 必须是数组");
  return { ok: errors.length === 0, errors };
}

let seq = 0;

export function makeProblem({ kind, spec = {}, params = {}, observables = [], id = null } = {}) {
  const problem = {
    kind,
    spec,
    params,
    observables,
    version: 1,
    id: id ?? `p${Date.now().toString(36)}-${++seq}`,
  };
  const check = validateProblem(problem);
  if (!check.ok) throw new Error("Problem 校验失败：" + check.errors.join("；"));
  return problem;
}

export const APP_VERSION = "0.9.0";

/* 深快照：spec/params 都是 JSON 可序列化的，用 JSON 往返即可（去掉函数/引用） */
export function snapshotProblem(problem) {
  return JSON.parse(JSON.stringify({
    kind: problem.kind,
    spec: problem.spec ?? {},
    params: problem.params ?? {},
    observables: problem.observables ?? [],
  }));
}

export function makeSolution(problem, { data, meta = {}, provenance = {}, convergence = null, error = null, solver, version }) {
  return {
    problemId: problem.id,
    kind: problem.kind,
    solver: solver ?? null,
    version: version ?? null,
    appVersion: APP_VERSION,
    problem: snapshotProblem(problem),   // 回放所需的完整快照
    data,
    meta,
    provenance,        // {calc:[], calib:[], input:[]} 之类的来源标注，或自由对象
    convergence,       // null | {cutoff, mesh, note} 等域相关
    error,             // null 或错误信息（求解器内部捕获后在此返回，不抛）
    solvedAt: new Date().toISOString(),
  };
}

export function describe(problem) {
  return `Problem(${problem.kind}) id=${problem.id} params=${JSON.stringify(problem.params)}`;
}
