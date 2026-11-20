/* export.js — 把每个图层的数值结果导成 CSV / JSON
 *
 * 原则：导出的必须是"算出来的数组"，不是屏幕上的字符串。
 * 每份数据都带元数据头（材料、截断、网格、参数、时间、版本），
 * 让下游能复现，也让人一眼看出"哪些是标定量"。
 */

import { materialOf } from "./phys/materials.js";
import { structure, bandsShifted, brillouinZone, recipVectors } from "./phys/epm.js";
import { dielectric } from "./phys/spectra.js";
import { magnon, atomicLevels } from "./phys/spin.js";
import { liquidConfig, pairCorrelation, structureFactor } from "./phys/liquid.js";
import { integrateBloch } from "./phys/bloch.js";
import { deviceModel, attenuation, tempScan, fluxScan } from "./phys/device.js";

export const VERSION = "0.9.0";

const num = (x, d = 6) => (Number.isFinite(x) ? Number(x.toFixed(d)) : "");

function metaOf(kind, S, extra = {}) {
  return {
    kind,
    material: S.material || null,
    params: Object.fromEntries(Object.entries(S).filter(([, v]) => typeof v === "number" || typeof v === "string" || typeof v === "boolean")),
    generatedAt: new Date().toISOString(),
    app: `Quantum Atlas ${VERSION}`,
    ...extra,
  };
}

function tableOf(name, note, arr) {
  if (!Array.isArray(arr) || !arr.length) return { name, note, columns: ["(empty)"], rows: [] };
  const keys = Object.keys(arr[0]);
  return { name, note, columns: keys, rows: arr.map((o) => keys.map((k) => num(o[k], 5))) };
}

/* ── 各图层的取数器 ─────────────────────────────────────── */
const EXPORTERS = {
  crystal(S) {
    const mat = materialOf(S.material);
    const bz = brillouinZone(mat);
    const A = mat.A || [[mat.a, 0, 0], [0, mat.a, 0], [0, 0, mat.a]];
    return [
      {
        name: "lattice",
        note: "晶格矢量（Å）与基元；BZ 面/顶点数",
        columns: ["a1x", "a1y", "a1z", "a2x", "a2y", "a2z", "a3x", "a3y", "a3z"],
        rows: [[...A.flat().map((x) => num(x, 6))]],
      },
      {
        name: "basis",
        note: "基元（分数坐标）",
        columns: ["index", "fx", "fy", "fz"],
        rows: (mat.basis || []).map((b, i) => [i, num(b[0], 4), num(b[1], 4), num(b[2], 4)]),
      },
      {
        name: "bz",
        note: "布里渊区概要",
        columns: ["metric", "value"],
        rows: [["faces", bz.faces.length], ["vertices", (bz.verts || []).length], ["a(Å)", num(mat.a, 4)]],
      },
    ];
  },

  bands(S) {
    const mat = materialOf(S.material);
    const res = structure(mat, { nG: 46, mesh: 6 });
    const E = bandsShifted(res) || [];
    const nB = Math.min(E[0]?.length || 0, 12);
    const cols = ["k_index", "kdist", ...Array.from({ length: nB }, (_, i) => `E${i + 1}_eV`)];
    const rows = E.map((row, i) => [i, num(res.kdist?.[i] ?? i, 5), ...Array.from({ length: nB }, (_, b) => num(row[b], 4))]);
    return [
      {
        name: "Ek",
        note: `能带 E(k)，VBM 对齐到 0；nG=46、mesh=6；gap=${num(res.gap, 4)} eV（${res.gapKind || "间接"}）`,
        columns: cols,
        rows,
      },
      {
        name: "path",
        note: "k 路径节点（分数坐标与标签）",
        columns: ["index", "f", "label", "kdist"],
        rows: (res.pts || []).map((p, i) => [i, num(p.f, 5), p.label || "", num(res.kdist?.[i] ?? 0, 5)]),
      },
    ];
  },

  spin(S) {
    const m = magnon({
      kind: S.spinKind || "FM",
      J: S.J ?? 1,
      S: S.spinS ?? 1,
      D: S.Danis ?? 0.12,
      n: 120,
    });
    const rows = m.q.map((q, i) => [i, num(q[0], 4), num(q[1], 4), num(Math.hypot(q[0], q[1]), 4), num(m.w[i], 4)]);
    const lv = atomicLevels({ S: S.spinS ?? 1, D: S.Danis ?? 0.12, B: S.Bfield ?? 1 });
    return [
      {
        name: "magnon",
        note: `自旋波沿 Γ–X–M–Γ；J=${S.J}，D=${S.Danis}，能隙 ${num(m.gap, 3)} meV`,
        columns: ["q_index", "qx", "qy", "q_abs", "omega_meV"],
        rows,
      },
      {
        name: "levels",
        note: "单原子 Zeeman/各向异性能级（meV）",
        columns: ["m", "E_meV", "dE_GHz"],
        rows: (Array.isArray(lv) ? lv : lv.levels || []).map((e) => [e.m ?? "", num(e.E, 4), num(e.dE, 2)]),
      },
    ];
  },

  spectra(S) {
    const mat = materialOf(S.material);
    const d = dielectric(mat, { nG: 26, mesh: S.specMesh ?? 8, gamma: S.specGamma ?? 0.1 });
    const rows = Array.from(d.w).map((w, i) => [num(w, 4), num(d.eps1[i], 4), num(d.eps2[i], 5), num(d.alpha?.[i] ?? 0, 5)]);
    return [
      {
        name: "epsilon",
        note: `ε(ω)：nG=26、mesh=${S.specMesh ?? 8}、γ=${S.specGamma ?? 0.1} eV；ε₁(0) 已按查表 ε∞ 标定（标定量，非预测）`,
        columns: ["photon_eV", "eps1", "eps2", "alpha_cm-1"],
        rows,
      },
    ];
  },

  liquid(S) {
    const cfg = liquidConfig({ N: S.liqN ?? 150, seed: S.liqSeed ?? 7 });
    const pc = pairCorrelation(cfg, { bins: 90, rMax: 6 });
    const sf = structureFactor(pc, { qMax: 6, n: 140 });
    return [
      {
        name: "gr",
        note: `g(r)：N=${S.liqN ?? 150}，seed=${S.liqSeed ?? 7}，ρ=0.0164 Å⁻³`,
        columns: ["r_A", "g_r"],
        rows: Array.from(pc.r).map((r, i) => [num(r, 4), num(pc.g[i], 5)]),
      },
      {
        name: "Sk",
        note: "S(k)",
        columns: ["q_A-1", "S_q"],
        rows: Array.from(sf.q).map((q, i) => [num(q, 4), num(sf.S[i], 5)]),
      },
    ];
  },

  bloch(S) {
    const frames = integrateBloch({
      B0: S.B0 ?? 0.4,
      T1: S.T1 ?? 6,
      T2: S.T2 ?? 2.2,
      Omega: S.rabi ?? 1,
      detuning: S.detune ?? 0,
      tMax: S.tMax ?? 8,
      n: 400,
    });
    return [
      {
        name: "bloch",
        note: `Bloch 方程 RK4：B0=${S.B0} T，T₁=${S.T1} µs，T₂=${S.T2} µs`,
        columns: ["t_us", "Mx", "My", "Mz", "|M|"],
        rows: (frames.t ? frames.t.map((t, i) => [t, frames.M[i]]) : frames.map((f) => [f.t, f.M])).map(([t, M]) => [
          num(t, 5), num(M[0], 5), num(M[1], 5), num(M[2], 5), num(Math.hypot(M[0], M[1], M[2]), 5),
        ]),
      },
    ];
  },

  device(S) {
    const m = deviceModel(S.devMat || "CZT", {
      d: S.devThick ?? 1,
      V: S.devBias ?? 300,
      E: S.devE ?? 60,
      T: S.devT ?? 300,
    });
    const flat = Object.entries(m).filter(([, v]) => typeof v === "number");
    const ts = tempScan(S.devMat || "CZT", { d: S.devThick ?? 1, V: S.devBias ?? 300 });
    const fs = fluxScan(S.devMat || "CZT", { d: S.devThick ?? 1, V: S.devBias ?? 300 });
    return [
      { name: "model", note: "器件模型标量结果", columns: ["key", "value"], rows: flat.map(([k, v]) => [k, num(v, 6)]) },
      tableOf("tempScan", "温度扫描（T / FWHM / 暗电流）", ts),
      tableOf("fluxScan", "通量扫描", fs),
    ];
  },
};

/* 其余图层：导出读数与参数（没有独立数组可给） */
function generic(S, readout) {
  const rows = (readout || []).map((r) => [r.k, String(r.v).replace(/<[^>]+>/g, "")]);
  return [{ name: "readout", note: "该图层没有连续数据集，导出读数与参数", columns: ["key", "value"], rows }];
}

/* ── 对外接口 ─────────────────────────────────────────── */
export function collect(sceneId, S, readout) {
  const fn = EXPORTERS[sceneId];
  const datasets = fn ? fn(S) : generic(S, readout);
  return { meta: metaOf(sceneId, S), datasets };
}

export function toCSV(bundle) {
  const lines = [];
  lines.push(`# Quantum Atlas export`);
  for (const [k, v] of Object.entries(bundle.meta)) {
    if (k === "params") continue;
    lines.push(`# ${k}: ${v}`);
  }
  for (const [k, v] of Object.entries(bundle.meta.params || {})) lines.push(`# param.${k}: ${v}`);
  for (const d of bundle.datasets) {
    lines.push("");
    lines.push(`# dataset: ${d.name}${d.note ? " — " + d.note : ""}`);
    lines.push(d.columns.join(","));
    for (const row of d.rows) lines.push(row.join(","));
  }
  return lines.join("\n") + "\n";
}

export function filenameFor(sceneId, ext) {
  const t = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  return `quantum-atlas_${sceneId}_${t}.${ext}`;
}
