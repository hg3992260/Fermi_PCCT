/* spin.js — 「自旋」域的求解内核：线性自旋波（方格子专路 + 任意晶格通路）
 *
 * 两种 spec 形态：
 *   1) 方格子（兼容旧）：{ kind:"FM"|"AF", S=1, J1=1, J2=0, D=0, n=90 }
 *   2) 任意晶格（共线铁磁）：{ S=1, D=0, bonds:[{r:[x,y],J}], path:[{label,q}], n=90 }
 *
 * data 返回 { q, dist, w:[分支], gap }，两种形态同构。
 */

import { branches } from "../phys/magnon.js";
import { branchesGeneric, branchesMulti } from "../phys/magnon-generic.js";
import { occupations, magnetizationT } from "../phys/magnon-thermal.js";
import { makeSolution } from "./envelope.js";

export const spinSolver = {
  id: "spin.magnon",
  version: "0.9.3",

  solve(problem) {
    try {
      const s = problem.spec;
      const T = s.T ?? 0;
      const Tc = s.Tc ?? 300;

      // 多子晶格（共线 FM）：sublattices=[{S,D},…] + bonds=[{a,b,r,J}] + path
      if (Array.isArray(s.sublattices)) {
        if (!Array.isArray(s.path) || s.path.length < 2) throw new Error("多子晶格需要 path");
        if (!Array.isArray(s.bonds) || s.bonds.length === 0) throw new Error("多子晶格需要 bonds");
        const b = branchesMulti(s.sublattices, s.bonds, s.path, s.n ?? 90);
        return makeSolution(problem, {
          solver: this.id, version: this.version,
          data: { q: b.q, dist: b.dist, w: b.w, gap: b.gap, ...finiteT(b, T, s.sublattices[0].S, Tc) },
          meta: { nBranches: b.w.length, nQ: b.q.length, lattice: "multi", sublattices: s.sublattices.length, T },
          provenance: { calc: ["w", "gap", "occupation", "magnetization"], input: ["sublattices", "bonds", "path", "T", "Tc"] },
          convergence: { note: "多子晶格共线 FM 线性自旋波（B=0）；ω² = eig(A)" },
        });
      }

      // 任意晶格通路（单子晶格共线 FM）
      if (Array.isArray(s.bonds)) {
        if (!Array.isArray(s.path) || s.path.length < 2) throw new Error("任意晶格需要 path（至少两个高对称节点）");
        const b = branchesGeneric({ S: s.S ?? 1, D: s.D ?? 0, bonds: s.bonds, path: s.path }, s.n ?? 90);
        return makeSolution(problem, {
          solver: this.id, version: this.version,
          data: { q: b.q, dist: b.dist, w: b.w, gap: b.gap, ...finiteT(b, T, s.S ?? 1, Tc) },
          meta: { nBranches: b.w.length, nQ: b.q.length, lattice: "custom", bonds: s.bonds.length, T },
          provenance: { calc: ["w", "gap", "occupation", "magnetization"], input: ["S", "D", "bonds", "path", "T", "Tc"] },
          convergence: { note: "共线铁磁线性自旋波（零温色散）+ Bose 占据 + 平均场磁化；非自洽有限温" },
        });
      }

      // 方格子专路
      const { kind = "FM", S = 1, J1 = 1, J2 = 0, D = 0, n = 90 } = s;
      if (kind !== "FM" && kind !== "AF") throw new Error(`spin kind 只支持 FM/AF，收到 "${kind}"`);
      const model = { n: kind === "AF" ? 2 : 1, kind, S, J1, J2, D };
      const b = branches(model, n);

      return makeSolution(problem, {
        solver: this.id, version: this.version,
        data: { q: b.q, dist: b.dist, w: b.w, gap: b.gap, ...finiteT(b, T, S, Tc) },
        meta: { nBranches: b.w.length, nQ: b.q.length, lattice: "square", T },
        provenance: { calc: ["w", "gap", "occupation", "magnetization"], input: ["kind", "S", "J1", "J2", "D", "T", "Tc"] },
        convergence: { note: "方格子共线线性自旋波；与解析式逐点一致（见 test-physics）；有限温为非自洽 Bose 占据" },
      });
    } catch (e) {
      return makeSolution(problem, { solver: this.id, version: this.version, data: null, error: e.message });
    }
  },
};

/* 有限温度附加量：Bose 占据（沿最高支）+ 平均场磁化 */
function finiteT(b, T, S, Tc) {
  if (!(T > 0)) return { occupation: null, magnetization: 1 };
  const top = b.w[b.w.length - 1];
  const occ = occupations(top, T);
  return {
    occupation: { avg: occ.avg, n: occ.n },
    magnetization: magnetizationT(T, { S, Tc }),
  };
}
