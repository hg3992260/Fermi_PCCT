/* L4 · 光谱：ε(ω) 的三维曲线 + 两面"投影墙" + 独立的 α 面板 */
import { C, V3, hex } from "../engine.js";
import { curve3, plane, dashLine } from "./util.js";
import { materialOf } from "../phys/materials.js";
import { wannier } from "../phys/exciton.js";
import { solve } from "../solver/solve.js";

const CACHE = new Map();

/** k 网格档位：9³ 快速（~0.4 s） / 15³ 精确（~2 s，ε₁ 的 KK 更平滑） */
function meshOf(S) {
  return S.quality === "精确 15³" ? 15 : 9;
}

function cout(a, b) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.abs(a[i] - (b[i] ?? a[i]));
  return out;
}

function spectraOf(mat, S) {
  const mesh = meshOf(S);
  const key = `${mat.id}:${mesh}:${S.Temp ?? 300}:${(S.gamma ?? 0.1).toFixed(2)}`;
  if (CACHE.has(key)) return CACHE.get(key);
  // 主量：介电函数（Na 走 Drude）——从通用求解器拿，单一数据源
  const sol = solve({ kind: "spectra", spec: { material: mat.id }, params: { nG: 26, mesh, gamma: S.gamma ?? 0.1, wMax: 8, dW: 0.02, T: S.Temp ?? 300 } });
  const out = sol.data.raw;
  out.mesh = mesh;
  // 收敛带：主量在 mesh 与 mesh/2 上的差（误差估计，属"衍生量"）
  if (mat.id !== "Na" && mesh >= 6) {
    const coarse = solve({ kind: "spectra", spec: { material: mat.id }, params: { nG: 26, mesh: Math.max(4, Math.round(mesh / 2)), gamma: S.gamma ?? 0.1, wMax: 8, dW: 0.02, T: S.Temp ?? 300 } }).data;
    out.band = Array.from(cout(out.eps2, coarse.eps2));
    out.bandMesh = [Math.round(mesh / 2), mesh];
  }
  CACHE.set(key, out);
  return out;
}

export default {
  id: "spectra",
  name: "L4 · 光谱",
  space: "参数空间 (ω, ε₁, ε₂)",
  spaceNote: "介电响应 · 带间跃迁",
  note:
    "光谱不是「另一种测量」，而是同一个哈密顿量的<b>另一个投影</b>：把占据带与空带之间的所有垂直跃迁按能量装箱（JDOS），乘上动量矩阵元，就得到 ε₂(ω)；再用 Kramers–Kronig 换出 ε₁(ω)。画面里那条悬空的曲线是三维的 ε(ω)，地面上是 ε₂ 的影子，后墙上是 ε₁ 的影子——同一个复函数的两面。",
  time: {
    label: "光子能量 ħω",
    min: 0, max: 8, step: 0.01,
    get: (S) => (S.omega ?? 2) / 8,
    set: (S, val) => { S.omega = val * 8; },
    fmt: (S) => `${(S.omega ?? 2).toFixed(2)} eV`,
  },
  params: [
    { key: "Temp", label: "温度", type: "slider", min: 4, max: 600, step: 4, unit: "K", hint: "Varshni 移动带边" },
    { key: "gamma", label: "展宽 Γ", type: "slider", min: 0.02, max: 0.4, step: 0.01, unit: "eV" },
    {
      key: "quality", label: "谱精度", type: "seg", options: ["快速 9³", "精确 15³"],
      hint: "k 网格越密，JDOS 采样越收敛（9³ 约 0.35 s / 15³ 约 0.6 s，之后走缓存）；ε₁ 在 vHs 处的陡降主要由展宽 Γ 决定",
    },
    { key: "showEps1", label: "显示 ε₁ 投影", type: "switch" },
    { key: "showAlpha", label: "显示吸收 α", type: "switch" },
  ],

  /** 换网格/换材料这类重算要提前跑在 busy 遮罩后面，别让画面卡住 */
  expensive(key, S) {
    if (meshOf(S) < 15) return null;
    if (key === "quality" || key === "gamma" || key === "Temp" || key === "material") {
      return "正在重算介电函数（k 网格 15³）…";
    }
    return null;
  },
  precompute(S) {
    spectraOf(materialOf(S.material), S);
  },

  build(v, S) {
    const mat = materialOf(S.material);
    const d = spectraOf(mat, S);
    const w = d.w;
    const WMAX = 8;

    /* ── 坐标系：地面 = (ω, ε₂)，后墙 = (ω, ε₁)，主曲线悬在两者之间 ── */
    const X = (ww) => -3.15 + (ww / WMAX) * 5.0;
    const Y = (e1) => -1.42 + ((Math.max(-6, Math.min(26, e1)) + 6) / 32) * 2.55;
    const zMax = Math.max(0.4, ...d.eps2);
    const Z = (e2) => -0.12 - (e2 / zMax) * 1.5;
    const y0 = Y(0);
    const zWall = -1.72;
    const x0 = X(0), x1 = X(WMAX);



    v.line([x0, y0, zWall], [x1, y0, zWall], C.slate, 0.5);
    v.line([x0, y0, 0], [x0, y0, zWall], C.slate, 0.5);
    v.line([x0, y0, zWall], [x0, Y(26), zWall], C.slate, 0.6);

    /* ── ω 轴（地面前沿）：每 1 eV 一个刻度，每 2 eV 标注 ── */
    for (let e = 0; e <= WMAX; e += 1) {
      const x = X(e);
      const major = e % 2 === 0;
      v.line([x, y0, 0], [x, y0, major ? -0.12 : -0.07], C.slate, major ? 0.9 : 0.55);
      if (major) v.label([x, y0 - 0.24, 0], `${e}`, { cls: "dim", offset: [0, 8] });
    }
    v.label([x1 + 0.22, y0 - 0.02, 0], "ħω (eV)", { cls: "tag", offset: [26, 6] });

    /* ── ε₁ 轴（后墙左缘） ── */
    for (const e of [-10, 0, 10, 20]) {
      const y = Y(e);
      v.line([x0 - 0.07, y, zWall], [x0 + 0.07, y, zWall], C.slate, 0.8);
      v.label([x0 - 0.12, y, zWall], `${e}`, { cls: "dim", offset: [-14, 0] });
      if (e !== 0) v.line([x0, y, zWall], [x1, y, zWall], C.slate, 0.05);
    }
    v.label([x0 - 0.1, Y(26) + 0.2, zWall], "ε₁", { cls: "tag", offset: [-18, 0] });

    /* ── ε₂ 轴（地面左缘） ── */
    for (const f of [0, 1]) {
      const z = -0.12 - f * 1.5;
      v.line([x0 - 0.07, y0, z], [x0 + 0.07, y0, z], C.slate, 0.8);
      v.label([x0 - 0.14, y0, z], f === 0 ? "0" : (f * zMax).toFixed(1), { cls: "dim", offset: [-16, 0] });
    }
    v.label([x0 - 0.1, y0 + 0.16, zWall * 0.92], "ε₂", { cls: "tag", offset: [-14, 0] });

    /* ── 主曲线：三维 ε(ω) = ε₁ + iε₂ ── */
    const main = w.map((ww, i) => [X(ww), Y(d.eps1[i]), Z(d.eps2[i])]);
    for (let i = 0; i < main.length - 1; i++) {
      const t = i / (main.length - 1);
      v.thick([main[i], main[i + 1]], V3.lerp(C.ice, C.amber, t), 0.022, 0.95);
    }
    v.label([X(2.0), Y(18), 1.35], "ε(ω) = ε₁ + iε₂", { cls: "tag" });

    /* ── 两个投影：地面 ε₂，后墙 ε₁ ── */
    const floor = w.map((ww, i) => [X(ww), y0, Z(d.eps2[i])]);
    curve3(v, floor, C.violet, C.violet, 0.7, 0.014);
    // 收敛带：ε₂ 的上下包络（mesh 与 mesh/2 的差），画成两条浅色细线
    if (d.band) {
      const hi = w.map((ww, i) => [X(ww), Y(d.eps1[i]), Z(d.eps2[i] + d.band[i] * 0.5)]);
      const lo = w.map((ww, i) => [X(ww), Y(d.eps1[i]), Z(Math.max(0, d.eps2[i] - d.band[i] * 0.5))]);
      curve3(v, hi, C.slate, C.slate, 0.35, 0.008);
      curve3(v, lo, C.slate, C.slate, 0.35, 0.008);
    }
    if (S.showEps1) {
      const wall = w.map((ww, i) => [X(ww), Y(d.eps1[i]), zWall]);
      curve3(v, wall, C.amber, C.amber, 0.55, 0.014);
    }

    /* ── 带隙：地面上的竖直标记 ── */
    if (d.Eg > 0.02) {
      const xg = X(Math.min(WMAX, d.Eg));
      v.thick([[xg, y0, 0], [xg, y0, zWall]], C.lime, 0.016, 0.85);
      v.line([xg, y0, zWall], [xg, y0 + 1.1, zWall], C.lime, 0.35);
      v.label([xg, y0 + 1.2, zWall], `E_g = ${d.Eg.toFixed(2)} eV`, { cls: "tag" });
    }

    /* ── 当前 ω：三个读数点 + 一条"阅读竖线" ── */
    const iw = Math.max(0, Math.min(w.length - 1, Math.round((S.omega ?? 2) / (w[1] - w[0]))));
    const xw = X(w[iw]);
    const pMain = [xw, Y(d.eps1[iw]), Z(d.eps2[iw])];
    const pFloor = [xw, y0, Z(d.eps2[iw])];
    const pWall = [xw, Y(d.eps1[iw]), zWall];
    v.line(pFloor, pWall, C.slate, 0.35);
    v.point(pFloor, C.violet, 6, 0.85);
    v.point(pWall, C.amber, 6, 0.85);
    v.point(pMain, C.white, 11, 1);
    v.pick(pMain, "eps", 0.34, { w: w[iw] });
    v.label(pMain, `ε = ${d.eps1[iw].toFixed(1)} + i${d.eps2[iw].toFixed(1)}`, { cls: "tag", offset: [0, -18] });

    /* ── α 面板：单独一块，不跟主曲线抢位置 ── */
    if (S.showAlpha) {
      const ax0 = 2.2, ax1 = 4.3, abase = Y(-4.5);
      const amax = Math.max(...d.alpha, 1e-3);
      v.line([ax0, abase, 0], [ax1, abase, 0], C.slate, 0.7);
      v.line([ax0, abase, 0], [ax0, abase + 1.9, 0], C.slate, 0.6);
      for (let e = 0; e <= WMAX; e += 2) {
        const x = ax0 + (e / WMAX) * (ax1 - ax0);
        v.line([x, abase, 0], [x, abase - 0.08, 0], C.slate, 0.7);
        v.label([x, abase - 0.18, 0], `${e}`, { cls: "dim", offset: [0, 8] });
      }
      const apts = w.map((ww, i) => [ax0 + (ww / WMAX) * (ax1 - ax0), abase + (d.alpha[i] / amax) * 1.8, 0]);
      curve3(v, apts, C.amber, C.red, 0.95, 0.02);
      const peak = d.alpha.indexOf(amax);
      v.point(apts[peak], C.amber, 8, 1);
      v.label([ax0 + 0.1, abase + 2.05, 0], "吸收系数 α(ω)", { cls: "tag" });
      v.label([apts[peak][0], apts[peak][1] + 0.16, 0], `${(amax * 1e4).toFixed(0)} cm⁻¹`, { cls: "dim" });
      v.label([ax1 + 0.1, abase - 0.02, 0], "ħω (eV)", { cls: "dim", offset: [18, 0] });
    }
  },

  readout(S) {
    const mat = materialOf(S.material);
    const d = spectraOf(mat, S);
    const iw = Math.max(0, Math.min(d.w.length - 1, Math.round((S.omega ?? 2) / (d.w[1] - d.w[0]))));
    const e1 = d.eps1[iw], e2 = d.eps2[iw];
    const n = Math.sqrt(Math.max(0, (Math.hypot(e1, e2) + e1) / 2));
    const k = Math.sqrt(Math.max(0, (Math.hypot(e1, e2) - e1) / 2));
    const rows = [
      { k: "ε₁ / ε₂", v: `${e1.toFixed(2)} / ${e2.toFixed(2)}` },
      { k: "折射率 n", v: n.toFixed(3), cls: "good" },
      { k: "消光系数 κ", v: k.toFixed(3) },
      { k: "吸收 α", v: `${(d.alpha[iw] * 1e4).toFixed(0)} cm⁻¹` },
      { k: "E_g(T)", v: d.Eg > 0.02 ? `${d.Eg.toFixed(3)} eV` : "金属 / 无能隙" },
      { k: "ε∞", v: d.epsInf.toFixed(1) },
    ];
    if (d.elapsed) rows.push({ k: "谱计算", v: `${d.elapsed.toFixed(0)} ms · ${d.mesh || 9}³` });
    if (d.band) {
      let iMax = 0;
      for (let i = 1; i < d.eps2.length; i++) if (d.eps2[i] > d.eps2[iMax]) iMax = i;
      const rel = d.band[iMax] / Math.max(1e-9, d.eps2[iMax]);
      rows.push({ k: "收敛带 @峰", v: `±${(rel * 100).toFixed(1)}%（mesh ${d.bandMesh[0]} vs ${d.bandMesh[1]}）` });
    }
    // Wannier–Mott 激子：从 EPM 曲率算有效质量，再解氢原子型方程
    try {
      const w = wannier(mat, { eps: d.epsInf });
      if (w) {
        rows.push({ k: "m*_e / m*_h", v: `${w.me.toFixed(2)} / ${w.mh.toFixed(2)} m₀` });
        rows.push({ k: "约化质量 μ", v: `${w.mu.toFixed(3)} m₀` });
        rows.push({ k: "激子束缚能", v: `${(w.eb * 1000).toFixed(1)} meV`, cls: "good" });
        rows.push({ k: "激子玻尔半径", v: `${w.aB.toFixed(0)} Å` });
        rows.push({ k: "吸收边 Eg−Eb", v: `${(w.gap - w.eb).toFixed(3)} eV` });
      }
    } catch (e) { /* 材料缺曲率就跳过 */ }
    return rows;
  },
};
