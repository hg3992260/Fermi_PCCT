/* liquid.js — 「液态」域求解内核：构型 → g(r) → S(k)
 * spec:
 *   方法一（RSA 硬球，旧）：{ N=150, seed=7, sigma=3.3, rho=0.0164 }
 *   方法二（LJ + Monte Carlo）：{ N=150, seed=7, sigma=1, rho=0.6, method:"mc", epsilon=1, T=1.2, steps }
 */

import { liquidConfig, pairCorrelation, structureFactor } from "../phys/liquid.js";
import { metropolisConfig } from "../phys/liquid-mc.js";
import { makeSolution } from "./envelope.js";

export const liquidSolver = {
  id: "liquid.rsa",
  version: "0.9.2",

  solve(problem) {
    try {
      const s = problem.spec;
      const method = s.method ?? "rsa";

      let cfg;
      if (method === "mc") {
        cfg = metropolisConfig({
          N: s.N ?? 150, rho: s.rho ?? 0.6, sigma: s.sigma ?? 1, epsilon: s.epsilon ?? 1,
          T: s.T ?? 1.2, seed: s.seed ?? 7, steps: s.steps ?? 20000, delta: s.delta ?? 0.1,
        });
      } else if (method === "rsa") {
        cfg = liquidConfig({ N: s.N ?? 150, seed: s.seed ?? 7, sigma: s.sigma ?? 3.3, rho: s.rho ?? 0.0164 });
      } else {
        throw new Error(`liquid method 只支持 rsa/mc（收到 "${method}"）`);
      }

      const { bins = 90, rMax = 6, qMax = 6, nQ = 140 } = problem.params;
      const pc = pairCorrelation(cfg, { bins, rMax });
      const sf = structureFactor(pc, { qMax, n: nQ });

      return makeSolution(problem, {
        solver: this.id, version: this.version,
        data: {
          r: Array.from(pc.r), g: Array.from(pc.g),
          q: Array.from(sf.q), S: Array.from(sf.S),
          ...(cfg.E !== undefined ? { energy: cfg.E, accRate: cfg.accRate } : {}),
          raw: { cfg, pc, sf },   // 完整构型 + g(r) + S(k)，供 3D 场景直接渲染
        },
        meta: { N: cfg.N, seed: s.seed ?? 7, sigma: s.sigma, rho: cfg.rho, method },
        provenance: { calc: ["g", "S", "energy"], input: ["N", "seed", "sigma", "rho", "method", "T", "epsilon"] },
        convergence: { note: method === "mc" ? `LJ 势 + Metropolis（${s.steps ?? 20000} 步，接受率 ${(cfg.accRate * 100).toFixed(0)}%）` : "RSA 硬球静态构型" },
      });
    } catch (e) {
      return makeSolution(problem, { solver: this.id, version: this.version, data: null, error: e.message });
    }
  },
};
