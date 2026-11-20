/* L7 · 量子态：Q-sphere / 密度矩阵 / 纠缠 */
import { C, V3, hex } from "../engine.js";
import { curve3, plane, bar } from "./util.js";
import { PRESETS, countBits } from "../phys/quantum.js";
import { solve } from "../solver/solve.js";

export default {
  id: "qubit",
  name: "L7 · 量子态",
  space: "状态空间 C^{2ⁿ}",
  spaceNote: "Q-sphere / ρ / 纠缠",
  note:
    "n 个量子比特的纯态是 C^{2ⁿ} 里的一个单位矢量。Q-sphere 是它的<b>半径–相位嵌入</b>：纬度 = 汉明重量（有几个 1），经度 = 复相位，半径 = 振幅模。纠缠无法在这个图上直接看出——它藏在<b>约化密度矩阵</b>里：整体是纯态，局部却是混态。",
  time: null,
  params: [
    { key: "nq", label: "比特数 n", type: "slider", min: 1, max: 4, step: 1 },
    { key: "statePreset", label: "预设态", type: "select", options: Object.keys(PRESETS), labels: Object.keys(PRESETS) },
    { key: "qview", label: "表示法", type: "seg", options: ["Q-sphere", "半球扇区"], hint: "半球扇区：张角 ∝ 概率、针向 ∝ 相位（arXiv:2303.08366 的惯例）" },
    { key: "showRho", label: "显示 ρ 柱状", type: "switch" },
  ],

  stateOf(S) {
    const n = Math.max(1, Math.round(S.nq ?? 2));
    const f = PRESETS[S.statePreset] || PRESETS["Bell (Φ⁺)"];
    return f(n);
  },

  build(v, S) {
    const n = Math.max(1, Math.round(S.nq ?? 2));
    const st = this.stateOf(S);
    const sol = solve({ kind: "quantum", spec: { n, state: st }, params: { R: 1.9 } });
    const nodes = sol.data.raw.nodes;

    const wedge = S.qview === "半球扇区";
    if (!wedge) {
    // 球面骨架与主轴（Q-sphere 的方向参照）
    v.sphereWire([0, 0, 0], 1.9, C.slate, { a: 0.12, seg: 10 });
    v.line([0, -2.0, 0], [0, 2.0, 0], C.slate, 0.22);

    // 纬度环（每个汉明重量一层）
    const levels = new Map();
    for (const nd of nodes) {
      const h = countBits(nd.i);
      if (!levels.has(h)) levels.set(h, nd.theta);
    }
    for (const [h, th] of levels) {
      const pts = [];
      const r = 1.9 * 0.98;
      for (let i = 0; i <= 60; i++) {
        const ph = (i / 60) * Math.PI * 2;
        pts.push([r * Math.sin(th) * Math.cos(ph), -r * Math.cos(th), r * Math.sin(th) * Math.sin(ph)]);
      }
      v.poly(pts, C.slate, { close: true, a: 0.10 });
    }

    // 节点
    for (const nd of nodes) {
      const col = V3.lerp(C.ice, C.amber, (1 - Math.cos(nd.phase)) / 2);
      const active = nd.p > 1e-6;
      v.line([0, 0, 0], nd.pos, col, active ? 0.28 : 0.06);
      v.point(nd.pos, active ? col : C.slate, nd.size, active ? 1 : 0.25);
      if (active && nd.p > 0.02) {
        v.label(nd.pos, `|${nd.bits}⟩`, { cls: "dim", offset: [0, -14] });
        v.pick(nd.pos, `q:${nd.i}`, 0.25, { bits: nd.bits, p: nd.p, phase: nd.phase });
      }
    }
    // 赤道
    const eq = [];
    for (let i = 0; i <= 80; i++) {
      const ph = (i / 80) * Math.PI * 2;
      eq.push([1.9 * Math.cos(ph), 0, 1.9 * Math.sin(ph)]);
    }
    v.poly(eq, C.ice, { a: 0.18 });

    v.label([0, 2.25, 0], `n = ${n} · dim = ${st.length}`, { cls: "tag" });
    } else {
      /* 半球扇区：另一种惯例（arXiv:2303.08366 fig1）——
         扇区张角 ∝ 该基态概率；扇区内的"相位针"角度 ∝ 复相位 */
      this.drawWedges(v, S, st, n);
    }

    // 密度矩阵柱状（右侧）
    if (S.showRho) {
      const dim = st.length;
      const x0 = 2.7, z0 = -1.6, sc = Math.min(0.34, 1.6 / dim);
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          const amp = [st[i][0] * st[j][0] + st[i][1] * st[j][1], st[i][0] * st[j][1] - st[i][1] * st[j][0]];
          const mag = Math.hypot(amp[0], amp[1]);
          if (mag < 1e-4) continue;
          const col = amp[0] >= 0 ? C.ice : C.pink;
          bar(v, [x0 + i * sc, 0, z0 + j * sc], sc * 0.32, mag * 1.5, col, 0.6);
        }
      }
      v.label([x0 + 0.4, 1.7, z0], "密度矩阵 |ρ_ij|", { cls: "tag" });
      v.label([x0 + 0.4, -0.18, z0 + dim * sc], "实部 = 冰蓝 · 虚部符号 = 品红", { cls: "dim" });
    }
    plane(v, 0, 2.6, C.slate, 0.02, 0);
  },

  /** 半球扇区：面积 = 概率、扇区颜色 = 相位、相位针 = arg(amp) */
  drawWedges(v, S, st, n) {
    const R = 1.9;
    const eye = v.eye();
    const zA = V3.norm(V3.sub(eye, [0, 0, 0]));
    const right = V3.norm(V3.cross([0, 1, 0], zA));
    const up = V3.cross(zA, right);
    const at = (r, a) => [
      right[0] * r * Math.cos(a) + up[0] * r * Math.sin(a),
      right[1] * r * Math.cos(a) + up[1] * r * Math.sin(a),
      right[2] * r * Math.cos(a) + up[2] * r * Math.sin(a),
    ];

    const probs = st.map((a) => a[0] * a[0] + a[1] * a[1]);
    let ang = Math.PI / 2;
    const steps = 14;
    const dim = st.length;
    for (let i = 0; i < dim; i++) {
      const p = probs[i];
      const span = Math.PI * 2 * p;
      if (span < 1e-4) continue;
      const amp = st[i];
      const phase = Math.atan2(amp[1], amp[0]);
      const col = V3.lerp(C.ice, C.amber, (1 - Math.cos(phase)) / 2);
      const a0 = ang, a1 = ang + span;
      // 扇区面：从圆心到外弧做"相位色 + 明度"的径向渐变
      // （圆心近白 → 外弧是饱和的相位色，这样至少外圈能读出相位）
      const NR = 9;
      for (let ri = 0; ri < NR; ri++) {
        const r0 = (ri / NR) * R;
        const r1 = ((ri + 1) / NR) * R;
        const tt = (ri + 0.5) / NR;
        const tint = V3.lerp([1, 1, 1], col, 0.12 + 0.80 * Math.pow(tt, 0.85));
        for (let k = 0; k < steps; k++) {
          const u0 = a0 + ((a1 - a0) * k) / steps;
          const u1 = a0 + ((a1 - a0) * (k + 1)) / steps;
          v.quad(at(r0, u0), at(r1, u0), at(r1, u1), at(r0, u1), tint, 0.16 + 0.34 * tt * (0.5 + 0.5 * p));
        }
      }
      // 相位针：在本扇区内按 arg(amp) 偏转
      const mid = (a0 + a1) / 2 + (phase / (Math.PI * 2)) * Math.min(span, Math.PI * 0.5);
      v.thick([at(R * 0.30, mid), at(R * 0.95, mid)], col, 0.016, 0.95);
      v.point(at(R * 0.95, mid), col, 7, 1);
      // 扇区分界：从圆心到圆周的实线，保证等相位时也能看出分区
      v.thick([[0, 0, 0], at(R, a0)], C.slate, 0.012, 0.85);
      // 扇区外弧
      const arc = [];
      for (let k = 0; k <= 10; k++) arc.push(at(R, a0 + ((a1 - a0) * k) / 10));
      v.thick(arc, col, 0.018, 0.95);
      // 标签
      const bits = i.toString(2).padStart(n, "0");
      v.label(at(R * 0.62, (a0 + a1) / 2), `|${bits}⟩`, { cls: "dim", offset: [0, -10] });
      if (p > 0.02) v.label(at(R * 1.08, (a0 + a1) / 2), `${(p * 100).toFixed(0)}%`, { cls: "tag", offset: [0, 0] });
      ang = a1;
    }
    // 外圈与分隔线
    const ring = [];
    for (let k = 0; k <= 96; k++) ring.push(at(R, (k / 96) * Math.PI * 2));
    v.poly(ring, C.slate, { close: true, a: 0.45 });
    v.label([0, R + 0.42, 0], "半球扇区：张角 ∝ 概率 · 针向 ∝ 相位", { cls: "tag" });
  },

  readout(S) {
    const n = Math.max(1, Math.round(S.nq ?? 2));
    const st = this.stateOf(S);
    const sol = solve({ kind: "quantum", spec: { n, state: st } });
    const probs = sol.data.probabilities;
    const H = -probs.reduce((s, p) => (p > 1e-12 ? s + p * Math.log2(p) : s), 0);
    const rows = [
      { k: "希尔伯特维数", v: `${st.length}` },
      { k: "参与基态数", v: `${probs.filter((p) => p > 1e-6).length}` },
      { k: "香农熵 H", v: `${H.toFixed(3)} bit` },
      { k: "归一化", v: probs.reduce((a, b) => a + b, 0).toFixed(6), cls: "good" },
    ];
    if (n === 2) rows.push({ k: "concurrence C", v: sol.data.concurrence.toFixed(3), cls: sol.data.concurrence > 0.9 ? "good" : "" });
    return rows;
  },
};
