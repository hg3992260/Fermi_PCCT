/* bands.js — 「能带」域的求解内核（把 EPM 包进统一信封）
 *
 * spec.material:
 *   "Si" | "Ge" | "C" | "Na" | "graphene" | 完整材料对象 | 自定义材料 spec（经 defineMaterial 规范化）
 * params:
 *   nG = 46、mesh = 6、nBands = 12
 */

import { MATERIALS } from "../phys/materials.js";
import { structure, bandsShifted } from "../phys/epm.js";
import { makeSolution } from "./envelope.js";
import { resolveMaterial } from "./material.js";

export const bandsSolver = {
  id: "bands.epm",
  version: "0.9.0",

  solve(problem) {
    try {
      const mat = resolveMaterial(problem.spec.material, MATERIALS);
      const { nG = 46, mesh = 6, nBands = 12 } = problem.params;
      const res = structure(mat, { nG, mesh });
      const E = bandsShifted(res) || [];
      const nKeep = Math.min(nBands, res.nBands ?? E[0]?.length ?? 0);

      const data = {
        E: E.map((row) => row.slice(0, nKeep)),
        kdist: res.kdist,
        pts: (res.pts || []).map((p) => ({ f: p.f, label: p.label ?? null })),
        gap: res.gap,
        gapKind: res.gapKind,
        nBands: res.nBands,
        nOcc: res.nOcc,
        isMetal: !!res.isMetal,
        raw: res,   // 完整内核结果（含 mat/E/zero/basis/…），供 3D 场景直接渲染，避免二次 structure()
      };

      return makeSolution(problem, {
        solver: this.id,
        version: this.version,
        data,
        meta: { materialId: mat.id, nPW: res.basis?.G?.length ?? null, nK: E.length },
        provenance: {
          calc: ["E", "gap", "gapKind", "nOcc"],
          calib: ["gap", "gapKind"],
          input: ["material.formFactors", "nG", "mesh"],
        },
        convergence: { cutoff: nG, mesh, note: "带隙 46→89 漂移见 test-convergence；局部 EPM 无 SOC" },
      });
    } catch (e) {
      return makeSolution(problem, { solver: this.id, version: this.version, data: null, error: e.message });
    }
  },
};
