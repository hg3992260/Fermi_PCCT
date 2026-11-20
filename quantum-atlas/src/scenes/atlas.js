/* L0 · 全景：尺度塔 —— 八个站点由"重正化流"串成螺旋阶梯 */
import { C, V3, hex } from "../engine.js";
import { PAL } from "./util.js";

const LAYERS = [
  { id: "crystal", name: "L1 晶格", sub: "实空间 · 10⁻¹⁰ m", color: C.amber, glyph: "cube" },
  { id: "bands", name: "L2 能带", sub: "倒空间 · eV", color: C.ice, glyph: "bands" },
  { id: "spin", name: "L3 自旋", sub: "局域矩 · meV", color: C.pink, glyph: "spin" },
  { id: "spectra", name: "L4 响应", sub: "光子 · 10⁻⁷ m", color: C.amber, glyph: "spectrum" },
  { id: "liquid", name: "L5 液态", sub: "无序 · 10⁻¹⁰ m", color: C.cyan, glyph: "liquid" },
  { id: "bloch", name: "L6 Bloch", sub: "二能级 · µeV", color: C.violet, glyph: "sphere" },
  { id: "qubit", name: "L7 量子态", sub: "n-qubit · GHz", color: C.violet, glyph: "qsphere" },
  { id: "circuit", name: "L8 电路", sub: "门序列 · 10⁻³ m", color: C.lime, glyph: "circuit" },
];

function plate(v, center, radius, color, alpha = 0.10) {
  const pts = [];
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    pts.push([center[0] + radius * Math.cos(a), center[1], center[2] + radius * Math.sin(a)]);
  }
  v.poly(pts, color, { close: true, a: 0.55 });
  // 半透明面片
  for (let i = 1; i < pts.length - 1; i++) v.tri(pts[0], pts[i], pts[i + 1], color, alpha);
}

function glyph(v, kind, c, color, t) {
  const s = 0.42;
  if (kind === "cube") {
    const p = (x, y, z) => [c[0] + x * s, c[1] + y * s, c[2] + z * s];
    const V = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]];
    const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    for (const [i, j] of E) v.line(p(...V[i]), p(...V[j]), color, 0.85);
  } else if (kind === "bands") {
    for (let b = 0; b < 3; b++) {
      const pts = [];
      for (let i = 0; i <= 10; i++) {
        const x = -s + (2 * s * i) / 10;
        const y = s * 0.6 * Math.sin((x * 2.2 + b * 1.1)) * (0.5 + 0.5 * b);
        pts.push([c[0] + x, c[1] + y + b * 0.12, c[2]]);
      }
      v.poly(pts, color, { a: 0.9 });
    }
  } else if (kind === "spin") {
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * Math.PI * 2 + 0.4;
      const r = 0.34;
      const p0 = [c[0] + r * Math.cos(a), c[1], c[2] + r * Math.sin(a)];
      const p1 = [p0[0], c[1] + 0.28 + 0.12 * Math.sin(a + t), p0[2]];
      v.arrow(p0, p1, color, { head: 0.09, a: 0.9 });
    }
  } else if (kind === "spectrum") {
    const pts = [];
    for (let i = 0; i <= 14; i++) {
      const x = -s + (2 * s * i) / 14;
      const y = 0.34 * Math.exp(-((x + 0.18) ** 2) / 0.05) + 0.26 * Math.exp(-((x - 0.34) ** 2) / 0.02);
      pts.push([c[0] + x, c[1] + y, c[2]]);
    }
    v.poly(pts, color, { a: 0.95 });
    v.line([c[0] - s, c[1], c[2]], [c[0] + s, c[1], c[2]], color, 0.25);
  } else if (kind === "liquid") {
    const rnd = (i) => (Math.sin(i * 12.9898 + 78.233) * 43758.5453) % 1;
    for (let i = 0; i < 9; i++) {
      const a = rnd(i) * Math.PI * 2, r = 0.18 + 0.26 * Math.abs(rnd(i + 3));
      v.point([c[0] + r * Math.cos(a), c[1] + 0.08 * Math.abs(rnd(i + 7)), c[2] + r * Math.sin(a)], color, 5, 0.9);
    }
  } else if (kind === "sphere") {
    v.sphereWire(c, s * 0.85, color, { a: 0.4, seg: 7 });
    v.arrow(c, [c[0], c[1] + s * 1.1, c[2]], C.pink, { head: 0.1 });
  } else if (kind === "qsphere") {
    v.sphereWire(c, s * 0.85, color, { a: 0.25, seg: 6 });
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2;
      v.point([c[0] + 0.34 * Math.cos(a), c[1] + (i % 2 ? 0.16 : -0.16), c[2] + 0.34 * Math.sin(a)], color, 7, 1);
    }
  } else if (kind === "circuit") {
    for (let q = 0; q < 3; q++) v.line([c[0] - s, c[1] + 0.16 * q - 0.16, c[2]], [c[0] + s, c[1] + 0.16 * q - 0.16, c[2]], color, 0.5);
    for (let g = 0; g < 2; g++) {
      const x = c[0] - 0.2 + g * 0.4;
      v.line([x, c[1] - 0.2, c[2]], [x, c[1] + 0.2, c[2]], color, 0.9);
      v.point([x, c[1] - 0.16, c[2]], color, 4.5);
      v.point([x, c[1] + 0.16, c[2]], color, 4.5);
    }
  }
}

export default {
  id: "atlas",
  name: "L0 · 全景",
  space: "复合空间",
  spaceNote: "能量 × 长度 × 状态",
  note:
    "八层不是八个软件功能，而是<b>同一条重正化流的八个站点</b>：每上升一层，就积掉一部分自由度，留下一个有效生成元。螺旋阶梯上的箭头方向就是「积掉」的方向；每一步都伴随一次<b>对称性降级</b>（从空间群到点群到自旋旋转群，最后到门集合生成的有限群）。",
  time: null,
  params: [],

  build(v, S) {
    const n = LAYERS.length;
    for (let k = 0; k < n; k++) {
      const L = LAYERS[k];
      const a = k * 0.62;
      const r = 2.9;
      const y = k * 0.78 - (n - 1) * 0.39;
      const c = [r * Math.cos(a), y, r * Math.sin(a)];
      plate(v, c, 0.98, L.color, 0.16);
      glyph(v, L.glyph, [c[0], c[1] + 0.34, c[2]], L.color, 0.4);

      v.label([c[0], c[1] + 1.04, c[2]], L.name, { cls: "tag", offset: [0, -4] });
      v.label([c[0], c[1] + 0.84, c[2]], L.sub, { cls: "dim", offset: [0, 8] });
      v.pick(c, L.id, 0.9, L.id);

      if (k < n - 1) {
        const a2 = (k + 1) * 0.62;
        const c2 = [r * Math.cos(a2), y + 0.78, r * Math.sin(a2)];
        const mid = V3.lerp(c, c2, 0.5);
        v.arrow([mid[0], c[1] + 0.12, mid[2]], [mid[0], c[1] + 0.52, mid[2]], C.slate, { head: 0.1, a: 0.35 });
      }
    }
    // 中心轴
    v.line([0, -3.6, 0], [0, 3.6, 0], C.slate, 0.18);
    v.label([0, 3.9, 0], "重正化流 ↑ 能量尺度", { cls: "dim" });
  },

  readout(S) {
    return [
      { k: "站点数", v: "8" },
      { k: "尺度跨度", v: "10⁻¹⁰ → 10⁻³ m" },
      { k: "能标跨度", v: "neV → 10 eV" },
      { k: "贯穿不变式", v: "谱 / Kubo / 粗粒化" },
    ];
  },
};
