/* L6 · Bloch：定理 / 球 / 方程 三重身份 */
import { C, V3, hex } from "../engine.js";
import { curve3, plane, dashLine } from "./util.js";
import { blochWave, integrateBloch, resonance } from "../phys/bloch.js";

const MODES = ["Bloch 球", "Bloch 定理", "弛豫曲线"];

export default {
  id: "bloch",
  name: "L6 · Bloch",
  space: "状态空间 CP¹ + 时间",
  spaceNote: "自旋 1/2 / 二能级",
  note:
    "Bloch 这个名字在这里指三件不同的事，但它们共享同一个数学内核：<b>周期性</b>与<b>二能级</b>都是把连续问题约化到有限维的方式。<b>定理</b>把连续空间折成能带指标 k；<b>球</b>把二维复矢量折成 CP¹ 上的一个点；<b>方程</b>把这个点的时间演化写成一阶微分方程——三者都是「约化 + 流」。",
  time: {
    label: "时间 t",
    min: 0, max: 1, step: 0.002,
    get: (S) => S.bt ?? 0,
    set: (S, val) => { S.bt = val; },
    fmt: (S) => `${((S.bt ?? 0) * 12).toFixed(2)} ns`,
  },
  params: [
    { key: "blochMode", label: "视角", type: "seg", options: MODES },
    { key: "B0", label: "静场 B₀", type: "slider", min: 0.05, max: 1.2, step: 0.01, unit: "T" },
    { key: "B1", label: "驱动 B₁", type: "slider", min: 0, max: 0.6, step: 0.01, unit: "T" },
    { key: "omega", label: "驱动频率 ω", type: "slider", min: 0.2, max: 4, step: 0.02, unit: "γB₀" },
    { key: "T1", label: "纵向弛豫 T₁", type: "slider", min: 0.5, max: 20, step: 0.1 },
    { key: "T2", label: "横向弛豫 T₂", type: "slider", min: 0.2, max: 12, step: 0.1 },
  ],

  build(v, S) {
    const mode = S.blochMode || "Bloch 球";
    if (mode === "Bloch 定理") return this.buildWave(v, S);
    if (mode === "弛豫曲线") return this.buildRelax(v, S);
    return this.buildSphere(v, S);
  },

  buildSphere(v, S) {
    const R = 1.9;
    v.sphereWire([0, 0, 0], R, C.slate, { a: 0.22, seg: 14 });
    // 三个主轴
    v.line([-R, 0, 0], [R, 0, 0], C.slate, 0.25);
    v.line([0, -R, 0], [0, R, 0], C.slate, 0.25);
    v.line([0, 0, -R], [0, 0, R], C.slate, 0.25);
    v.label([0, R + 0.2, 0], "|0⟩", { cls: "tag" });
    v.label([0, -R - 0.2, 0], "|1⟩", { cls: "tag" });
    v.label([R + 0.2, 0, 0], "x", { cls: "dim" });
    v.label([0, 0, R + 0.2], "y", { cls: "dim" });

    const path = integrateBloch({
      B0: S.B0 ?? 0.4, B1: S.B1 ?? 0.25, omega: (S.omega ?? 1.2), gamma: 2 * Math.PI,
      T1: S.T1 ?? 6, T2: S.T2 ?? 2.2, tMax: 12, dt: 0.02,
    });
    const cur = Math.max(0, Math.min(path.length - 1, Math.round((S.bt ?? 0) * (path.length - 1))));
    const pts = path.map((p) => [p.M[0] * R, p.M[2] * R, p.M[1] * R]);
    for (let i = 0; i < cur; i++) {
      const f = i / Math.max(1, cur);
      v.line(pts[i], pts[i + 1], V3.lerp(C.violet, C.pink, f), 0.5 + 0.45 * f);
    }
    // 态矢量
    const M = path[cur].M;
    const tip = [M[0] * R, M[2] * R, M[1] * R];
    v.arrow([0, 0, 0], tip, C.pink, { head: 0.22 });
    v.point(tip, C.white, 8, 1);
    v.pick(tip, "state", 0.3, { M });
    v.label(tip, `M = (${M[0].toFixed(2)}, ${M[1].toFixed(2)}, ${M[2].toFixed(2)})`, { cls: "tag", offset: [0, -16] });
    plane(v, -R - 0.5, 2.6, C.slate, 0.02, 0);
  },

  buildWave(v, S) {
    const w = blochWave({ N: 8, k: 0.35 + (S.omega ?? 1.2) * 0.25, n: 240 });
    const sx = 5.2 / w.N, sy = 1.15;
    const x0 = -2.6, y0 = -0.4;
    // 周期势（离子实）
    for (let i = 0; i <= w.N; i++) {
      v.point([x0 + i * sx, y0 - 0.55, 0], C.amber, 7, 0.9);
      v.line([x0 + i * sx, y0 - 0.55, 0], [x0 + i * sx, y0 - 0.2, 0], C.amber, 0.3);
    }
    // 波函数
    const pts = w.xs.map((x, i) => [x0 + x * sx, y0 + w.re[i] * sy, 0]);
    curve3(v, pts, C.ice, C.violet, 0.95, 0.02);
    // 包络 |u|²
    const env = w.xs.map((x, i) => [x0 + x * sx, y0 + w.mod[i] * sy * 1.1, 0.35]);
    curve3(v, env, C.pink, C.pink, 0.45);
    v.line([x0, y0, 0], [x0 + w.N * sx, y0, 0], C.slate, 0.3);
    v.label([x0 + w.N * sx + 0.1, y0, 0], "x", { cls: "dim" });
    v.label([x0 + 1.2, y0 + 1.35, 0], "Re ψ(x)", { cls: "tag" });
    v.label([x0 + 1.2, y0 + 0.95, 0.35], "|u(x)|² 周期包络", { cls: "dim" });
    v.label([x0, y0 - 1.0, 0], `ψ_nk = e^{ikx} u_nk ,  k = ${w.k.toFixed(2)} (2π/a)`, { cls: "dim" });

    // 能带折叠示意
    const ex = 2.6;
    for (let b = 0; b < 3; b++) {
      const bp = [];
      for (let i = 0; i <= 40; i++) {
        const kk = (i / 40) * 2 - 1;
        bp.push([ex + kk * 0.9, -1.2 + b * 0.55 + 0.28 * Math.cos(Math.PI * kk), 0]);
      }
      curve3(v, bp, b === 0 ? C.amber : C.slate, b === 0 ? C.amber : C.slate, b === 0 ? 0.9 : 0.35);
    }
    v.label([ex, 0.9, 0], "E(k)", { cls: "tag" });
    dashedFrame(v, ex);
  },

  buildRelax(v, S) {
    const path = integrateBloch({
      B0: S.B0 ?? 0.4, B1: S.B1 ?? 0.25, omega: S.omega ?? 1.2, gamma: 2 * Math.PI,
      T1: S.T1 ?? 6, T2: S.T2 ?? 2.2, tMax: 12, dt: 0.03,
    });
    const n = path.length;
    const tMax = path[n - 1].t;
    const x0 = -3.2, sx = 6.2 / tMax, sy = 1.35;

    v.line([x0, 0, 0], [x0 + 6.2, 0, 0], C.slate, 0.3);
    v.line([x0, -0.2, 0], [x0, 1.9, 0], C.slate, 0.22);
    const z0 = path.map((p) => [x0 + p.t * sx, p.M[2] * sy, 0]);
    const rxy = path.map((p) => [x0 + p.t * sx, Math.hypot(p.M[0], p.M[1]) * sy, 0.6]);
    curve3(v, z0, C.cyan, C.cyan, 0.9);
    curve3(v, rxy, C.pink, C.pink, 0.85);
    v.label([x0 + 3, 1.7, 0], "M_z(t)  纵向 T₁", { cls: "tag" });
    v.label([x0 + 3, 1.3, 0.6], "√(M_x²+M_y²)  横向 T₂", { cls: "tag" });

    const cur = Math.max(0, Math.min(n - 1, Math.round((S.bt ?? 0) * (n - 1))));
    const xc = x0 + path[cur].t * sx;
    v.line([xc, -0.2, 0], [xc, 1.9, 0], C.white, 0.16);
    v.point([xc, path[cur].M[2] * sy, 0], C.cyan, 7, 1);
    v.point([xc, Math.hypot(path[cur].M[0], path[cur].M[1]) * sy, 0.6], C.pink, 7, 1);
    v.label([xc, 2.0, 0], `t = ${path[cur].t.toFixed(2)}`, { cls: "tag" });
    plane(v, -0.3, 3.4, C.slate, 0.02, 0);
  },

  readout(S) {
    const mode = S.blochMode || "Bloch 球";
    const path = integrateBloch({
      B0: S.B0 ?? 0.4, B1: S.B1 ?? 0.25, omega: S.omega ?? 1.2, gamma: 2 * Math.PI,
      T1: S.T1 ?? 6, T2: S.T2 ?? 2.2, tMax: 12, dt: 0.03,
    });
    const cur = Math.max(0, Math.min(path.length - 1, Math.round((S.bt ?? 0) * (path.length - 1))));
    const M = path[cur].M;
    const r = resonance({ B0: S.B0 ?? 0.4, gamma: 2 * Math.PI, omega: S.omega ?? 1.2 });
    const rows = [
      { k: "|M|", v: Math.hypot(...M).toFixed(3), cls: Math.hypot(...M) > 1.02 ? "bad" : "" },
      { k: "M_z", v: M[2].toFixed(3) },
      { k: "纯度 √(Mx²+My²)", v: Math.hypot(M[0], M[1]).toFixed(3) },
      { k: "Larmor ω₀", v: `${(r.w0 / (2 * Math.PI)).toFixed(2)} γB₀` },
      { k: "失谐 Δ", v: `${(r.detuning / (2 * Math.PI)).toFixed(2)} γB₀` },
    ];
    if (mode === "Bloch 定理") rows.push({ k: "k", v: `${(0.35 + (S.omega ?? 1.2) * 0.25).toFixed(2)} (2π/a)` });
    return rows;
  },
};

function dashedFrame(v, x) {
  dashLine(v, [x, -1.3, 0], [x, 0.6, 0], C.slate, 0.08, 0.3);
}
