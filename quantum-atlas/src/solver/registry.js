/* registry.js — 求解器注册表
 *
 * 每个域注册一个内核，统一通过 solve(problem) 调用。
 * 内核约定：{ id, version, solve(problem) -> Solution }
 */

const registry = new Map();

export function register(kind, solver) {
  if (typeof kind !== "string" || !solver || typeof solver.solve !== "function") {
    throw new Error(`register(${kind}, …) 需要 { id, version, solve(problem) }`);
  }
  if (registry.has(kind)) throw new Error(`kind "${kind}" 已注册`);
  registry.set(kind, solver);
  return solver;
}

export function get(kind) {
  return registry.get(kind) ?? null;
}

export function has(kind) {
  return registry.has(kind);
}

export function kinds() {
  return [...registry.keys()];
}

export function list() {
  return [...registry.entries()].map(([kind, s]) => ({ kind, id: s.id, version: s.version }));
}
