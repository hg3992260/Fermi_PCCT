/* crystal.js — 「晶体」域求解内核：晶格几何与布里渊区
 * spec.material: "Si" | 材料对象 | 自定义材料 spec（经 defineMaterial 规范化）
 */

import { MATERIALS } from "../phys/materials.js";
import { brillouinZone } from "../phys/epm.js";
import { makeSolution } from "./envelope.js";
import { resolveMaterial } from "./material.js";

export const crystalSolver = {
  id: "crystal.geometry",
  version: "0.9.0",
  solve(problem) {
    try {
      const mat = resolveMaterial(problem.spec.material, MATERIALS);
      const bz = brillouinZone(mat);
      const A = mat.A || [[mat.a, 0, 0], [0, mat.a, 0], [0, 0, mat.a]];
      return makeSolution(problem, {
        solver: this.id, version: this.version,
        data: {
          a: mat.a,
          lattice: A,
          basis: mat.basis || [],
          nElec: mat.nElec,
          bzFaces: bz.faces.length,
          bzVertices: (bz.verts || []).length,
        },
        meta: { materialId: mat.id, kind: mat.kind },
        provenance: { input: ["a", "basis", "material"], calc: ["bzFaces", "bzVertices"] },
        convergence: null,
      });
    } catch (e) {
      return makeSolution(problem, { solver: this.id, version: this.version, data: null, error: e.message });
    }
  },
};
