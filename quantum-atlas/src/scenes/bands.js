/* L2 · 能带：带状图 / E(kx,ky) 曲面 / 费米面（三种三维视图）+ 布里渊区路径 */
import { C, V3, hex } from "../engine.js";
import { drawBZ, plane, dashLine } from "./util.js";
import { gapRigid, varshniGap } from "../phys/thermal.js";
import { splittings, spinPolarization, SOC_LIT } from "../phys/spinbands.js";
import { socAtGamma } from "../phys/soc-epm.js";
import { solveAt, recipVectors } from "../phys/epm.js";
import { solve } from "../solver/solve.js";

const SURF_CACHE = new Map();
const FS_CACHE = new Map();

function kPtsOf(res) {
  if (res.mat.proj === "2D") {
    const B = recipVectors(res.mat.A);
    return res.pts.map((p) => [0, 1, 2].map((d) => p.f[0] * B[0][d] + p.f[1] * B[1][d]));
  }
  return res.pts.map((p) => p.k);
}

export default {
  id: "bands",
  name: "L2 · 能带",
  space: "倒空间 (R³)*",
  spaceNote: "E(k) · 布洛赫谱",
  note:
    "能带不是「一条曲线」，而是<b>布里渊区上的一个函数</b>。左侧把高对称路径展开成能带图，右侧把它按真实几何放回布里渊区——两幅图是同一条曲线的两种投影。费米面是 E = E_F 的等能面；带隙是它在能量轴上的空隙。",
  time: {
    label: "路径位置",
    min: 0, max: 1, step: 0.002,
    get: (S) => (S.kIndex ?? 45) / 100,
    set: (S, val, ctx) => { S.kIndex = Math.round(val * 100); },
    fmt: (S, ctx) => `${Math.round(((S.kIndex ?? 45) / 100) * ((ctx?.nPts ?? 100) - 1))} / ${(ctx?.nPts ?? 100) - 1}`,
  },
  params: [
    { key: "bandView", label: "视图", type: "seg", options: ["带状", "曲面", "费米面"], hint: "同一份 E(k) 数据的三种三维呈现" },
    { key: "band", label: "能带指标 n", type: "slider", min: 0, max: 10, step: 1 },
    { key: "nBandsShown", label: "显示带数", type: "slider", min: 4, max: 18, step: 1 },
    { key: "showBZ", label: "布里渊区与路径", type: "switch" },
    { key: "showSOC", label: "自旋分裂（SOC/Zeeman）", type: "switch" },
    { key: "spinB", label: "磁场 B", type: "slider", min: 0, max: 10, step: 0.5, unit: "T" },
    { key: "spinDex", label: "交换劈裂 D_ex", type: "slider", min: 0, max: 100, step: 1, unit: "meV" },
    { key: "temp", label: "温度 T", type: "slider", min: 300, max: 800, step: 25, unit: "K", hint: "刚性离子 EPM 的温漂（热膨胀 + 德拜-瓦尔）" },
  ],

  /** 左侧能量轴：eV 刻度 + 轻网格 + 单位标签，让"能带图"真的可以读数 */
  drawEnergyAxis(v, res, X0, XW, EY, y0) {
    const xa = X0 - 0.34;
    const xEnd = X0 + XW;
    const eLo = res.isMetal ? -6 : -6;
    const eHi = 6;
    v.line([xa, EY(eLo), 0], [xa, EY(eHi), 0], C.slate, 0.9);
    for (let e = eLo; e <= eHi; e += 2) {
      const y = EY(e);
      const major = e % 2 === 0;
      v.line([xa - 0.07, y, 0], [xa + 0.07, y, 0], C.slate, major ? 0.95 : 0.6);
      // 横向网格：从轴一直拉到图右端
      if (e !== 0) v.line([xa, y, 0], [xEnd, y, 0], C.slate, e === 0 ? 0.0 : 0.10);
      const txt = e > 0 ? `+${e}` : `${e}`;
      v.label([xa - 0.10, y, 0], txt, { cls: "dim", offset: [-13, 0] });
    }
    // 零点（VBM / E_F）强调
    v.line([xa - 0.09, y0, 0], [xa + 0.09, y0, 0], C.slate, 1);
    v.label([xa - 0.10, y0, 0], "0", { cls: "", offset: [-13, 0] });
    v.label([xa, EY(eHi) + 0.22, 0], res.isMetal ? "E − E_F  (eV)" : "E − E_VBM  (eV)", { cls: "tag", offset: [-6, 0] });
  },

  build(v, S) {
    const sol = solve({ kind: "bands", spec: { material: S.material }, params: { nG: 46, mesh: 6 } });
    const res = sol.data.raw;
    const mat = res.mat;
    const E = res._shifted || res.E.map((row) => row.map((e) => e - res.zero));
    const kp = kPtsOf(res);
    const nPts = E.length;
    const kMax = Math.max(...kp.map((p) => Math.hypot(p[0], p[1], p[2]))) || 1;
    const nB = S.nBandsShown || 10;
    const sel = Math.min(S.band ?? res.nOcc, res.nBands - 1);
    const view = S.bandView || "带状";

    const X0 = -3.4, XW = 5.6;
    const sMax = res.kdist[nPts - 1] || 1;
    const X = (i) => X0 + (res.kdist[i] / sMax) * XW;
    // 能量压缩比 0.30/世界单位；在 ±7.6 eV 处硬截断，保证 ±6 eV 的刻度落在真实位置
    const EY = (e) => Math.max(-2.28, Math.min(2.28, e * 0.30));
    const y0 = 0;

    // 右侧布里渊区
    if (S.showBZ !== false) {
      const bzr = 1.55;
      const bzo = [4.75, 0.55, 0];
      drawBZ(v, mat, { offset: bzo, r: bzr, color: C.ice, alpha: 0.34, labels: false });
      const ks = bzr / kMax;
      // 路径（俯视投影到 BZ 的赤道面）
      const pp = kp.map((p) => [bzo[0] + p[0] * ks, bzo[1], bzo[2] + p[1] * ks]);
      v.thick(pp, C.amber, 0.034, 0.95);
      res.pts.forEach((p, i) => {
        if (!p.label) return;
        const k = kp[i];
        const q = [bzo[0] + k[0] * ks, bzo[1] + 0.02, bzo[2] + k[1] * ks];
        v.point(q, C.white, 4, 0.9);
        v.label(q, p.label, { cls: "dim", offset: [0, -10] });
      });
      v.label([bzo[0], bzo[1] + bzr + 0.45, bzo[2]], "布里渊区 · 高对称路径", { cls: "tag" });
      v.label([0.6, -2.75, 0], "← 展开为能带图", { cls: "dim" });
    }

    if (view === "曲面" && res.basis) {
      this.drawSurface(v, res, E, sel, X0, XW, EY);
    } else if (view === "费米面" && (res.isMetal || mat.id === "graphene")) {
      this.drawFermi(v, res, kp, E, sel, X0, XW, EY);
    } else {
      this.drawBands(v, res, E, X, EY, nB, sel, nPts, y0, S);
    }

    // 能量轴 + 刻度 + 横向网格（科学可读性的核心）
    if (view !== "费米面") {
      this.drawEnergyAxis(v, res, X0, XW, EY, y0);
    }

    // 能量零点
    plane(v, y0, 2.7, C.slate, 0.014, 0);
    dashLine(v, [X0 - 0.1, y0, 0], [X0 + XW + 0.1, y0, 0], C.slate, 0.1, 0.3);

    // 带隙标注：在导带底所在的 k 处画一个双箭头
    if (res.gap > 0.05 && res.gap < 9) {
      const xg = X(res.cbmAt);
      const yv = EY(res.vbm - res.zero);
      const yc = EY(res.cbm - res.zero);
      if (Math.abs(yc - yv) > 0.16) {
        v.thick([[xg, yv, 0], [xg, yc, 0]], C.lime, 0.016, 0.95);
        v.line([xg - 0.10, yv, 0], [xg + 0.10, yv, 0], C.lime, 0.95);
        v.line([xg - 0.10, yc, 0], [xg + 0.10, yc, 0], C.lime, 0.95);
        v.label([xg + 0.14, (yv + yc) / 2, 0], `E_g = ${res.gap.toFixed(3)} eV`, { cls: "tag", offset: [56, 0] });
      }
    }

    // 高对称点
    res.pts.forEach((p, i) => {
      if (!p.label) return;
      const x = X(i);
      v.line([x, y0 - 0.10, 0], [x, y0 + 0.10, 0], C.slate, 0.5);
      v.label([x, y0 - 0.26, 0], p.label, { cls: "tag" });
    });
  },

  drawBands(v, res, E, X, EY, nB, sel, nPts, y0, S) {
    for (let b = 0; b < Math.min(nB, res.nBands); b++) {
      const occ = b < res.nOcc;
      const col = b === sel ? C.amber : occ ? C.ice : C.violet;
      const a = b === sel ? 1 : 0.5;
      const pts = [];
      for (let i = 0; i < nPts; i++) pts.push([X(i), EY(E[i][b]), 0]);
      v.thick(pts, col, b === sel ? 0.046 : 0.028, a);
      if (b === sel) {
        for (let i = 0; i < nPts; i += 5) v.line([X(i), y0, 0], [X(i), EY(E[i][b]), 0], col, 0.10);
      }
    }
    // 当前 k 切片
    const nPtsNow = E.length;
    const ki = Math.max(0, Math.min(nPtsNow - 1, Math.round(((S.kIndex ?? 45) / 100) * (nPtsNow - 1))));
    const xk = X(ki);
    v.line([xk, -2.4, 0], [xk, 2.5, 0], C.white, 0.16);
    for (let b = 0; b < Math.min(nB, res.nBands); b++) {
      v.point([xk, EY(E[ki][b]), 0], b < res.nOcc ? C.amber : C.cyan, b === sel ? 9 : 4, 0.95);
    }
    v.pick([xk, EY(E[ki][sel]), 0], "k", 0.4, { kIdx: ki });
    v.label([xk, 2.62, 0], `k #${ki}`, { cls: "dim" });
  },

  drawSurface(v, res, E, sel, X0, XW, EY) {
    const key = `${res.mat.id}:${sel}`;
    let surf = SURF_CACHE.get(key);
    if (!surf) {
      const n = 26;
      const basis = { G: res.basis.G.slice(0, 20), scale: res.basis.scale };
      const s = (2 * Math.PI) / res.mat.a;
      const kMax = s * 0.58;
      const grid = [];
      for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
          const kx = -kMax + (2 * kMax * i) / (n - 1);
          const ky = -kMax + (2 * kMax * j) / (n - 1);
          const { values } = solveAt(res.mat, basis, [kx, ky, 0]);
          row.push(values[Math.min(sel, values.length - 1)] - res.zero);
        }
        grid.push(row);
      }
      surf = { grid, n, kMax };
      SURF_CACHE.set(key, surf);
    }
    const col = sel < res.nOcc ? C.amber : C.cyan;
    const sc = 5.4 / 2;
    for (let i = 0; i < surf.n - 1; i++) {
      for (let j = 0; j < surf.n - 1; j++) {
        const x0 = -2.7 + (5.4 * i) / (surf.n - 1);
        const x1 = -2.7 + (5.4 * (i + 1)) / (surf.n - 1);
        const z0 = -2.7 + (5.4 * j) / (surf.n - 1);
        const z1 = -2.7 + (5.4 * (j + 1)) / (surf.n - 1);
        const q = (a, b) => EY(surf.grid[a][b]);
        v.quad([x0, q(i, j), z0], [x1, q(i + 1, j), z0], [x1, q(i + 1, j + 1), z1], [x0, q(i, j + 1), z1], col, 0.30);
      }
    }
    v.label([-2.7, 2.65, -2.7], `E_${sel}(k_x, k_y) · k_z = 0`, { cls: "tag" });
    v.label([2.9, 0.05, -2.9], "k_y", { cls: "dim" });
    v.label([2.9, 0.05, 2.9], "k_x", { cls: "dim" });
  },

  drawFermi(v, res, kp, E, sel, X0, XW, EY) {
    const key = `${res.mat.id}`;
    let pts = FS_CACHE.get(key);
    if (!pts) {
      pts = [];
      const n = 15;
      const s = (2 * Math.PI) / res.mat.a;
      const basis = res.basis ? { G: res.basis.G.slice(0, 18), scale: res.basis.scale } : null;
      if (basis) {
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++)
            for (let l = 0; l < n; l++) {
              const k = [((i / n) - 0.5) * s, ((j / n) - 0.5) * s, ((l / n) - 0.5) * s];
              const { values } = solveAt(res.mat, basis, k);
              const dE = values[res.nOcc - 1] - res.zero;
              if (Math.abs(dE) < 0.3) pts.push([k[0], k[1], k[2]]);
            }
      }
      FS_CACHE.set(key, pts);
    }
    const sc = 2.2 / ((2 * Math.PI) / res.mat.a);
    for (const p of pts) v.point([p[0] * sc, p[1] * sc, p[2] * sc], C.cyan, 2.8, 0.5);
    v.sphereWire([0, 0, 0], 0.4, C.slate, { a: 0.2, seg: 8 });
    v.label([0, 1.9, 0], "E = E_F 等能面（k 空间点云）", { cls: "tag" });
    v.label([0, -2.0, 0], `占据带数 n_occ = ${res.nOcc}`, { cls: "dim" });
  },

  readout(S) {
    const sol = solve({ kind: "bands", spec: { material: S.material }, params: { nG: 46, mesh: 6 } });
    const res = sol.data.raw;
    const mat = res.mat;
    const E = res._shifted || [];
    const nPts = E.length;
    if (!nPts) return [{ k: "状态", v: "能带计算中…" }];
    const kIdx = Math.max(0, Math.min(nPts - 1, Math.round(((S.kIndex ?? 45) / 100) * (nPts - 1))));
    const sel = Math.max(0, Math.min(S.band ?? res.nOcc ?? 0, (res.nBands ?? 1) - 1));
    const row = E[kIdx] || [];
    const rows = [];
    if (res.isMetal) {
      rows.push({ k: "金属", v: res.gapKind, cls: "good" });
      rows.push({ k: "费米能", v: "0.000 eV（参考）" });
    } else {
      rows.push({ k: "带隙", v: `${res.gap.toFixed(3)} eV`, cls: "good" });
      rows.push({ k: "类型", v: `${res.gapKind}（模型）` });
      rows.push({ k: "实验带隙", v: `${mat.gapExp} eV` });
      rows.push({ k: "直接带隙", v: `${res.directGap.toFixed(3)} eV` });
    }
    rows.push({ k: `E_${sel}(k)`, v: row[sel] != null ? `${row[sel].toFixed(3)} eV` : "—" });
    rows.push({ k: "平面波数", v: `${res.nBands}` });
    // 温度：算出"刚性离子"那部分（热膨胀 + 德拜-瓦尔），和 Varshni 对照
    try {
      const T = S.temp ?? 300;
      const gT = gapRigid(mat, T);
      const g300 = gapRigid(mat, 300);
      const vT = varshniGap(mat, T);
      const v300 = varshniGap(mat, 300);
      rows.push({ k: "温度", v: `${T} K` });
      rows.push({ k: "带隙(T)", v: `${gT.toFixed(3)} eV`, cls: "good" });
      rows.push({ k: "ΔEg 刚性(300→T)", v: `${(gT - g300 >= 0 ? "+" : "")}${((gT - g300) * 1000).toFixed(0)} meV` });
      if (vT != null) rows.push({ k: "ΔEg Varshni", v: `${((vT - v300) * 1000).toFixed(0)} meV` });
    } catch (e) { /* 材料缺热参数就跳过 */ }
    // 自旋分裂：Zeeman 严格可算，SOC 是给定量级（文献 Δ_SO，标成输入）
    if (S.showSOC) {
      const socLit = SOC_LIT[mat.id] ?? 0;
      const sp = splittings({ B: S.spinB ?? 0, Dex: S.spinDex ?? 0, soc: socLit });
      rows.push({ k: "Zeeman 分裂", v: `${sp.zeeman.toFixed(3)} meV` });
      if (sp.exchange > 0) rows.push({ k: "交换劈裂", v: `${sp.exchange.toFixed(1)} meV` });
      rows.push({ k: "Δ_SO（文献输入）", v: `${socLit.toFixed(0)} meV` });
      // 原型：用局部赝势从 ∇V×p 直接算 SOC 矩阵元（结果证明这条路走不通）
      if (S.socProbe) {
        try {
          const r = socAtGamma(mat, { nG: 46 });
          rows.push({ k: "Δ_SO（局部 EPM 算出）", v: `${(r.dsoc * 1e3).toExponential(1)} µeV`, cls: "warn" });
        } catch (e) { /* 跳过 */ }
      }
      rows.push({ k: "价带顶总分裂", v: `${(sp.soc / 2 + sp.total / 2).toFixed(1)} meV` });
      rows.push({ k: "自旋极化率 P", v: `${sp.spinPolarization ?? spinPolarization({ B: S.spinB ?? 0, Dex: S.spinDex ?? 0 }).toFixed(3)}` });
    }
    rows.push({ k: "计算耗时", v: `${res.elapsed.toFixed(0)} ms` });
    return rows;
  },
};
