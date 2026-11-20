/* spectra.js — 「光谱」域求解内核：介电函数
 * spec: { material: "Si" | 材料对象 }
 * params: { nG=26, mesh=8, gamma=0.10, wMax=8, dW=0.02, T=300, sym=true }
 */

import { MATERIALS } from "../phys/materials.js";
import { dielectric, drude } from "../phys/spectra.js";
import { makeSolution } from "./envelope.js";
import { resolveMaterial } from "./material.js";

export const spectraSolver = {
  id: "spectra.jdos",
  version: "0.9.1",
  solve(problem) {
    try {
      const mat = resolveMaterial(problem.spec.material, MATERIALS);
      const { nG = 26, mesh = 8, gamma = 0.1, wMax = 8, dW = 0.02, T = 300, sym = true } = problem.params;
      const d = mat.id === "Na" ? drude(mat, { wp: 8.2, gam: gamma }) : dielectric(mat, { nG, mesh, gamma, wMax, dW, T, sym });
      if (mat.id === "Na") d.Eg = 0;

      return makeSolution(problem, {
        solver: this.id, version: this.version,
        data: {
          w: Array.from(d.w),
          eps1: Array.from(d.eps1),
          eps2: Array.from(d.eps2),
          alpha: d.alpha ? Array.from(d.alpha) : null,
          Eg: d.Eg,
          epsInf: d.epsInf,
          raw: d,   // 完整内核结果（含 elapsed/mesh 等），供 3D 场景直接渲染
        },
        meta: { materialId: mat.id, nG, mesh, gamma, T, sym, model: mat.id === "Na" ? "drude" : "jdos" },
        provenance: {
          calc: ["eps2", "eps1", "alpha"],
          calib: ["epsInf"],
          input: ["material.formFactors", "nG", "mesh", "gamma"],
        },
        convergence: { mesh, note: mat.id === "Na" ? "Drude 自由电子模型（金属）" : "ε₂ 峰值对网格敏感，收敛带见 L4 读数 / test-convergence" },
      });
    } catch (e) {
      return makeSolution(problem, { solver: this.id, version: this.version, data: null, error: e.message });
    }
  },
};
