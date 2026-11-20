/* L5 · 液态：无序体系中的原子自旋 + g(r) + S(q,ω) */
import { C, V3, hex } from "../engine.js";
import { spinArrow, curve3, plane } from "./util.js";
import { dynamicFactor } from "../phys/liquid.js";
import { solve } from "../solver/solve.js";

const CACHE = new Map();

function liquid(S) {
  const key = `liq:${S.liqSeed ?? 7}:${S.liqN ?? 150}`;
  if (CACHE.has(key)) return CACHE.get(key);
  // 构型 + g(r) + S(q) —— 从通用求解器拿，单一数据源
  const sol = solve({ kind: "liquid", spec: { N: S.liqN ?? 150, seed: S.liqSeed ?? 7 }, params: { bins: 90, rMax: 9, qMax: 4.2, nQ: 110 } });
  const { cfg, pc, sf } = sol.data.raw;
  const out = { cfg, pc, sf };
  CACHE.set(key, out);
  return out;
}

export default {
  id: "liquid",
  name: "L5 · 液态",
  space: "实空间 + 关联空间",
  spaceNote: "无序构型 / g(r) / S(q,ω)",
  note:
    "液体没有晶格，<b>k 不再是好量子数</b>；但自旋波函数依然存在——每个原子带一个旋量，只是相位在空间上只保持有限相干长度（这里 ~2 个原子间距）。宏观上，这些旋量被热涨落搅乱，只剩<b>关联函数</b>：g(r) 描述位置，S(q,ω) 描述自旋的扩散。³He 是费米液体：低能激发是准粒子，自旋扩散系数 D 由 Landau 参数决定。",
  time: {
    label: "扩散时间 t",
    min: 0, max: 1, step: 0.002,
    get: (S) => S.liqT ?? 0,
    set: (S, val) => { S.liqT = val; },
    fmt: (S) => (S.liqT ?? 0).toFixed(2),
  },
  params: [
    { key: "liqN", label: "原子数", type: "slider", min: 60, max: 220, step: 10 },
    { key: "liqSeed", label: "构型种子", type: "slider", min: 1, max: 40, step: 1, hint: "改变无序实现" },
    { key: "Ds", label: "自旋扩散 D", type: "slider", min: 0.2, max: 4, step: 0.1, unit: "a²/τ" },
    { key: "showSqw", label: "S(q,ω) 曲面", type: "switch" },
    { key: "showGr", label: "g(r)", type: "switch" },
  ],

  build(v, S) {
    const { cfg, pc, sf } = liquid(S);
    const t = (S.liqT ?? 0) * 3;

    // 把盒子映射到世界；液体盒子居中
    const L = cfg.L;
    const sc = 6.4 / L;
    const P = (p) => [(p[0] - L / 2) * sc, (p[1] - L / 2) * sc, (p[2] - L / 2) * sc];

    // 原子与自旋旋量
    cfg.pos.forEach((p, i) => {
      const sp = cfg.spins[i];
      const w = P(p);
      const c = Math.cos(sp.phase + t * 1.6);
      const col = V3.lerp(C.pink, C.ice, (c + 1) / 2);
      v.point(w, col, 3.4 + 3.4 * sp.amp, 0.7);
      const th = sp.theta;
      const ph = sp.phase + t * 1.6;
      const m = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
      spinArrow(v, w, m, 0.26 * sp.amp, col, 0.5);
    });
    v.label([0, 3.55, 0], `液态 ³He · ${cfg.N} 原子 · ρ = ${cfg.rho.toFixed(4)} Å⁻³`, { cls: "tag" });

    // g(r)：右侧竖直平面
    if (S.showGr !== false) {
      const gx = 3.6, gy = -2.4, wSpan = 2.1, hSpan = 1.9;
      const rMax = pc.r[pc.r.length - 1];
      v.line([gx, gy, 0], [gx + wSpan, gy, 0], C.slate, 0.3);
      v.line([gx, gy, 0], [gx, gy + hSpan, 0], C.slate, 0.3);
      const gpts = pc.r.map((r, i) => [gx + (r / rMax) * wSpan, gy + Math.min(hSpan, pc.g[i] * 0.55), 0]);
      curve3(v, gpts, C.cyan, C.violet, 0.95);
      v.line([gx, gy + 0.55, 0], [gx + wSpan, gy + 0.55, 0], C.slate, 0.12);
      v.label([gx + wSpan * 0.5, gy + hSpan + 0.25, 0], "g(r) 径向分布", { cls: "tag" });
      v.label([gx + wSpan + 0.08, gy - 0.04, 0], "r (Å)", { cls: "dim", offset: [16, 0] });
    }

    // S(q,ω)：后方悬浮曲面
    if (S.showSqw) {
      const dyn = dynamicFactor(sf, { D: S.Ds ?? 1.6, wMax: 2.2, nw: 34 });
      const x0 = -2.9, z0 = -4.4, base = 2.05;
      const nq = Math.min(46, sf.q.length);
      const amax = Math.max(...dyn.grid.flat());
      const hOf = (val) => Math.pow(Math.max(0, val) / amax, 0.55) * 1.5;
      for (let iq = 0; iq < nq - 1; iq++) {
        for (let iw = 0; iw < dyn.ws.length - 1; iw++) {
          const x_a = x0 + (iq / nq) * 5.8, x_b = x0 + ((iq + 1) / nq) * 5.8;
          const z_a = z0 + (dyn.ws[iw] / 2.2) * 1.5, z_b = z0 + (dyn.ws[iw + 1] / 2.2) * 1.5;
          const h00 = hOf(dyn.grid[iq][iw]), h01 = hOf(dyn.grid[iq][iw + 1]);
          const h10 = hOf(dyn.grid[iq + 1][iw]), h11 = hOf(dyn.grid[iq + 1][iw + 1]);
          const c = V3.lerp(C.ice, C.pink, Math.min(1, h00 / 1.4));
          v.quad([x_a, base + h00, z_a], [x_b, base + h10, z_a], [x_b, base + h11, z_b], [x_a, base + h01, z_b], c, 0.26);
        }
      }
      v.label([0, base + 1.75, z0 + 0.7], "S(q,ω) · 自旋扩散", { cls: "tag" });
      v.label([x0 - 0.15, base - 0.05, z0 - 0.15], "q", { cls: "dim" });
      v.label([x0 + 5.95, base - 0.05, z0 + 1.6], "ω", { cls: "dim" });
    }
  },

  readout(S) {
    const { cfg, pc, sf } = liquid(S);
    const gmax = Math.max(...pc.g);
    const iPeak = pc.g.indexOf(gmax);
    return [
      { k: "原子数", v: `${cfg.N}` },
      { k: "数密度 ρ", v: `${pc.rho.toFixed(3)} Å⁻³` },
      { k: "第一配位壳", v: `${pc.r[iPeak].toFixed(2)} Å`, cls: "good" },
      { k: "g(r) 峰值", v: gmax.toFixed(2) },
      { k: "S(q) 主峰 q", v: `${sf.q[sf.S.indexOf(Math.max(...sf.S))].toFixed(2)} Å⁻¹` },
      { k: "自旋扩散 D", v: `${(S.Ds ?? 1.6).toFixed(2)}` },
    ];
  },
};
