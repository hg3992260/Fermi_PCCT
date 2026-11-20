/* L1 · 凝聚态：晶格（实空间）→ 布里渊区（倒空间） */
import { C, V3, hex } from "../engine.js";
import { drawBZ, PAL, frame } from "./util.js";
import { materialOf } from "../phys/materials.js";
import { recipVectors } from "../phys/epm.js";

// 原子统一用品牌主色（Petrol），离子/元素差异留给标签

function atomsOf(mat, rep) {
  const A = mat.A;
  const out = [];
  for (let i = 0; i < rep; i++)
    for (let j = 0; j < rep; j++)
      for (let k = 0; k < rep; k++) {
        mat.basis.forEach((b, bi) => {
          const p = [0, 1, 2].map(
            (d) => (b[0] + i) * A[0][d] + (b[1] + j) * A[1][d] + (b[2] + k) * A[2][d]
          );
          out.push({ p, species: mat.species[bi] || mat.species[0], bi });
        });
      }
  return out;
}

export default {
  id: "crystal",
  name: "L1 · 晶格",
  space: "实空间 R³",
  spaceNote: "晶胞 / 原子 / 键",
  note:
    "<b>对称性 → 几何</b>：实空间的平移群决定倒空间的<b>布里渊区</b>——它是倒格子的 Wigner–Seitz 胞。右侧多面体不是装饰，而是「哪些 k 点是可约的」这一问题的答案：点群越小，BZ 越不对称，能带的简并越少。",
  time: null,
  params: [
    { key: "rep", label: "超胞", type: "seg", options: [1, 2, 3], hint: "n×n×n 原胞；观察平移对称性" },
    { key: "showBZ", label: "显示布里渊区", type: "switch" },
    { key: "showBonds", label: "显示化学键", type: "switch" },
    { key: "showCell", label: "显示晶胞棱", type: "switch" },
    { key: "showPrim", label: "显示原胞", type: "switch", hint: "红 = 原胞，黑 = 惯用胞（论文 fig1a 的双胞套嵌）" },
  ],

  build(v, S) {
    const mat = materialOf(S.material);
    const rep = S.rep || 1;
    const A = mat.A;
    const atoms = atomsOf(mat, rep);
    const base = V3.add(V3.add(A[0], A[1]), A[2]);
    const center = V3.mul(base, (rep - 1) / 2);
    // 归一化：整块晶体缩放到约 3.2 个世界单位
    const span = Math.max(
      V3.len(A[0]) * rep, V3.len(A[1]) * rep, V3.len(A[2]) * rep,
      V3.len(base) * rep * 0.55
    );
    const sc = 3.2 / span;
    const O = [0, 1.15, 0];
    const P = (p) => V3.add(V3.mul(V3.sub(p, center), sc), O);

    // 晶胞棱
    if (S.showCell !== false) {
      const o = [0, 0, 0];
      const cs = [
        o, A[0], V3.add(A[0], A[1]), A[1],
        A[2], V3.add(A[2], A[0]), V3.add(V3.add(A[2], A[0]), A[1]), V3.add(A[2], A[1]),
      ].map(P);
      // 六个面：极淡填充，让"盒子"有体积感
      const quads = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
      for (const q of quads) v.quad(cs[q[0]], cs[q[1]], cs[q[2]], cs[q[3]], C.ice, 0.035);
      for (let i = 0; i < 4; i++) {
        v.thick([cs[i], cs[(i + 1) % 4]], C.slate, 0.012, 0.9);
        v.thick([cs[i + 4], cs[4 + ((i + 1) % 4)]], C.slate, 0.012, 0.9);
        v.thick([cs[i], cs[i + 4]], C.slate, 0.010, 0.8);
      }
      v.arrow(P([0, 0, 0]), P(A[0]), C.amber, { head: 0.24, a: 1 });
      v.arrow(P([0, 0, 0]), P(A[1]), C.lime, { head: 0.24, a: 1 });
      v.arrow(P([0, 0, 0]), P(A[2]), C.ice, { head: 0.24, a: 0.95 });
      // 惯用胞的轴不再单独标注（论文惯例：a₁a₂a₃ 标在原胞上）
    }

    /* 原胞：与惯用胞共角点套嵌（arXiv:1803.09279 fig1a 的画法：
       黑线 = Bravais 惯用胞，红线 = 原胞） */
    if (S.showPrim !== false) {
      const rel = mat.id === "Na"
        ? [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]     // bcc
        : mat.id === "graphene" ? null
        : [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]];             // fcc / 金刚石
      if (rel) {
        const PV = rel.map((v) => [0, 1, 2].map((d) => v[0] * A[0][d] + v[1] * A[1][d] + v[2] * A[2][d]));
        const o = [0, 0, 0];
        const cs = [
          o, PV[0], V3.add(PV[0], PV[1]), PV[1],
          PV[2], V3.add(PV[2], PV[0]), V3.add(V3.add(PV[2], PV[0]), PV[1]), V3.add(PV[2], PV[1]),
        ].map(P);
        for (let i = 0; i < 4; i++) {
          v.thick([cs[i], cs[(i + 1) % 4]], C.pink, 0.011, 0.95);
          v.thick([cs[i + 4], cs[4 + ((i + 1) % 4)]], C.pink, 0.011, 0.95);
          v.thick([cs[i], cs[i + 4]], C.pink, 0.010, 0.85);
        }
        const names = ["a₁", "a₂", "a₃"];
        PV.forEach((pv, i) => {
          const q = P(pv);
          v.label(V3.lerp(P([0, 0, 0]), q, 1.06), names[i], { cls: "dim", offset: [8, 6] });
        });

      }
    }

    // 最近邻距离
    let dmin = Infinity;
    for (let i = 0; i < atoms.length; i++)
      for (let j = i + 1; j < atoms.length; j++) {
        const d = V3.len(V3.sub(atoms[i].p, atoms[j].p));
        if (d > 1e-6 && d < dmin) dmin = d;
      }

    // 键
    if (S.showBonds !== false && dmin < Infinity) {
      const cut = dmin * 1.12;
      for (let i = 0; i < atoms.length; i++)
        for (let j = i + 1; j < atoms.length; j++) {
          const d = V3.len(V3.sub(atoms[i].p, atoms[j].p));
          if (d <= cut) v.thick([P(atoms[i].p), P(atoms[j].p)], C.slate, 0.010, 0.6);
        }
    }

    // 原子
    atoms.forEach((a, i) => {
      const col = C.ice;
      const hi = v.highlight === `atom:${i}`;
      v.point(P(a.p), col, hi ? 16 : 13, 0.96);
      if (a.bi === 0) v.pick(P(a.p), `atom:${i}`, 0.28 * sc * 20, { species: a.species, pos: a.p });
    });
    v.label([O[0], O[1] + 1.9, O[2]], `${mat.name} · a = ${mat.a} Å`, { cls: "tag" });

    // 倒空间：布里渊区（下方）
    if (S.showBZ !== false) {
      const by = -3.35;
      const r = 1.45;
      drawBZ(v, mat, { offset: [0, by, 0], r, color: C.ice, alpha: 0.62, labels: true, width: 0.012 });
      v.label([0, by - r - 0.45, 0], "第一布里渊区（倒格子的 Wigner–Seitz 胞）", { cls: "tag" });
      v.arrow([1.25, -0.6, 0], [1.25, by + r + 0.5, 0], C.slate, { head: 0.16, a: 0.45 });
      v.label([1.42, -1.3, 0], "ℱ  傅里叶变换", { cls: "dim", offset: [26, 0] });
    }

    frame(v, { size: 4.4, grid: 1, axes: false });
  },

  readout(S) {
    const mat = materialOf(S.material);
    const A = mat.A;
    const vol = Math.abs(V3.dot(A[0], V3.cross(A[1], A[2])));
    const B = recipVectors(A);
    return [
      { k: "晶格常数 a", v: `${mat.a.toFixed(3)} Å` },
      { k: "原胞体积", v: `${vol.toFixed(2)} Å³` },
      { k: "原子/原胞", v: `${mat.basis.length}` },
      { k: "倒格矢 |b₁|", v: `${Math.hypot(...B[0]).toFixed(3)} Å⁻¹` },
      { k: "结构", v: mat.cn },
    ];
  },
};
