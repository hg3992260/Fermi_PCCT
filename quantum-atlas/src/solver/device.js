/* device.js — 「器件」域求解内核：探测器模型 + 温度/通量扫描
 * spec: { material: "CZT" | 探测器对象, d=1(mm), V=300(V), E=60(keV), T=300(K) }
 */

import { deviceModel, tempScan, fluxScan, DETECTORS } from "../phys/device.js";
import { makeSolution } from "./envelope.js";
import { resolveDetector } from "./detector.js";

export const deviceSolver = {
  id: "device.hecht",
  version: "0.9.2",

  solve(problem) {
    try {
      const { material, d = 1, V = 300, E = 60, T = 300 } = problem.spec;
      const mat = resolveDetector(material, DETECTORS);

      // deviceModel 读 opts.thickness/voltage/energy/temp（不是 d/V/E/T）
      const model = deviceModel(mat, { thickness: d, voltage: V, energy: E, temp: T });
      const scalars = {};
      for (const [k, v] of Object.entries(model)) if (typeof v === "number") scalars[k] = v;

      const ts = tempScan(mat, { thickness: d, voltage: V }).map((r) => ({ T: r.T, fwhm: r.fwhm, dark: r.dark }));
      const fs = fluxScan(mat, { thickness: d, voltage: V }).map((r) => ({ ...r }));

      return makeSolution(problem, {
        solver: this.id, version: this.version,
        data: { scalars, tempScan: ts, fluxScan: fs, raw: { model, tempScan: ts, fluxScan: fs } },
        meta: { material: mat.id, d, V, E, T },
        provenance: {
          calc: ["cceAvg", "fwhmTotal", "ni", "rho", "Ef", "current"],
          calib: [],
          input: ["material", "d", "V", "E", "muTau", "attenuation"],
        },
        convergence: { note: "1D Hecht + 泊松 + MC；无像素间电荷共享/读出电子学/权重势" },
      });
    } catch (e) {
      return makeSolution(problem, { solver: this.id, version: this.version, data: null, error: e.message });
    }
  },
};
