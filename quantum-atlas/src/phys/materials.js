/* ─────────────────────────────────────────────────────────────
   materials.js — 晶体结构 + 经验赝势参数
   约定：长度 Å，能量 eV（赝势表用 Ry，乘 RY 转换）
   赝势采用"反演中心为原点"规范，使结构因子为实数 → H 实对称
   （diamond/rocksalt 均有反演中心；这是 EPM 的标准技巧）
   ───────────────────────────────────────────────────────────── */

export const RY = 13.605693; // eV
export const HBAR2_2M = 3.809982; // eV·Å²

const A_FCC = (a) => [
  [0, a / 2, a / 2],
  [a / 2, 0, a / 2],
  [a / 2, a / 2, 0],
];
const A_BCC = (a) => [
  [-a / 2, a / 2, a / 2],
  [a / 2, -a / 2, a / 2],
  [a / 2, a / 2, -a / 2],
];
const A_HEX = (a, c) => [
  [a, 0, 0],
  [a / 2, (a * Math.sqrt(3)) / 2, 0],
  [0, 0, c],
];

/** 结构因子（实数）：Φ(dG) = (1/N) Σ_j cos(dG·τ_j)，反演中心规范 */
function phase(kind, a) {
  if (kind === "diamond") {
    const t = [a / 8, a / 8, a / 8];
    return (g) => Math.cos(g[0] * t[0] + g[1] * t[1] + g[2] * t[2]);
  }
  if (kind === "rocksalt") {
    const t = [a / 2, a / 2, a / 2];
    return (g) => 0.5 * (1 + Math.cos(g[0] * t[0] + g[1] * t[1] + g[2] * t[2]));
  }
  return () => 1; // bcc / 单原子
}

/**
 * 材料定义
 * basis: 分数坐标（相对原胞基矢）
 * vtab:  EPM 形状因子（Ry），键为 |G|²/(2π/a)² 的整数值
 * proj:  "3D" | "2D"
 */
export const MATERIALS = {
  Si: {
    id: "Si", name: "硅 Si", cn: "金刚石结构 · 间接带隙半导体",
    kind: "diamond", proj: "3D",
    a: 5.431,
    A: A_FCC(5.431),
    basis: [[0, 0, 0], [0.25, 0.25, 0.25]],
    species: ["Si", "Si"],
    vtab: { 3: -0.22, 8: 0.04, 11: 0.08 },
    nElec: 8,
    gapExp: 1.12,
    color: "#5aa9ff",
    note: "8 个价电子填满 4 条价带；价带顶在 Γ，导带底在 Γ–X 的 0.85 处 → 间接带隙。",
  },
  Ge: {
    id: "Ge", name: "锗 Ge", cn: "金刚石结构 · 窄带隙半导体",
    kind: "diamond", proj: "3D",
    a: 5.658,
    A: A_FCC(5.658),
    basis: [[0, 0, 0], [0.25, 0.25, 0.25]],
    species: ["Ge", "Ge"],
    vtab: { 3: -0.20, 8: 0.04, 11: 0.06 },
    nElec: 8,
    gapExp: 0.66,
    color: "#7c9cff",
    note: "与 Si 同构，但 Γ 点导带低得多：间接与直接带隙只差 0.14 eV。",
  },
  Na: {
    id: "Na", name: "钠 Na", cn: "体心立方 · 近自由电子金属",
    kind: "bcc", proj: "3D",
    a: 4.225,
    A: A_BCC(4.225),
    basis: [[0, 0, 0]],
    species: ["Na"],
    vtab: { 2: -0.024, 4: 0.008, 6: -0.004 },
    nElec: 1,
    gapExp: 0,
    color: "#ffb84d",
    note: "1 个价电子 → 半满的 s 带，费米面接近球面；赝势很弱，能带几乎就是空晶格折叠。",
  },
  C: {
    id: "C", name: "金刚石 C", cn: "金刚石结构 · 宽禁带半导体",
    kind: "diamond", proj: "3D",
    a: 3.567,
    A: A_FCC(3.567),
    basis: [[0, 0, 0], [0.25, 0.25, 0.25]],
    species: ["C", "C"],
    vtab: { 3: -0.85, 8: 0.6, 11: 0.05 },
    nElec: 8,
    gapExp: 5.47,
    color: "#e9e4ff",
    note: "与 Si 同构，但 2s/2p 更紧致 → 赝势更强、带隙 5.5 eV；同时介电常数小、激子束缚能大。",
  },
  graphene: {
    id: "graphene", name: "石墨烯", cn: "二维蜂窝 · 狄拉克半金属",
    kind: "hex", proj: "2D",
    a: 2.46, c: 20,
    A: A_HEX(2.46, 20),
    basis: [[0, 0, 0], [1 / 3, 1 / 3, 0]],
    species: ["C", "C"],
    tb: { t: 2.7 },
    nElec: 2,
    gapExp: 0,
    color: "#37d6c0",
    note: "最近邻紧束缚（t = 2.7 eV）：K/K′ 点线性色散，费米面退化为两个点。",
  },
  He3: {
    id: "He3", name: "液态 ³He", cn: "量子液体 · 费米液体（S=1/2）",
    kind: "liquid", proj: "3D",
    a: 3.6,
    A: A_BCC(3.6),
    basis: [[0, 0, 0]],
    species: ["³He"],
    nElec: 2,
    gapExp: 0,
    color: "#8fa3bf",
    note: "无晶格。3.6 Å 是平均原子间距；自旋 1/2 的费米液体，T_F ≈ 1.8 K。",
  },
};

export function materialOf(id) {
  return MATERIALS[id] || MATERIALS.Si;
}

/** 结构因子函数（缓存） */
export function phaseOf(mat) {
  if (!mat._phase) mat._phase = phase(mat.kind, mat.a);
  return mat._phase;
}
