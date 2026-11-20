// 参数标定：以实验带隙为目标，微调 Si / Ge / C(金刚石) 的形状因子
import { solveAt, recipVectors, structure } from "../src/phys/epm.js";
import { materialOf } from "../src/phys/materials.js";

const scale2 = (m) => (m.a / (2 * Math.PI)) ** 2;

function gapOf(mat, vtab) {
  const clone = { ...mat, vtab, _phase: mat._phase };
  const B = recipVectors(mat.A);
  // Γ–X 路径（间接带隙所在）
  const s = (2 * Math.PI) / mat.a;
  const pts = [];
  for (let i = 0; i <= 24; i++) {
    const t = i / 24;
    pts.push([0, t * s, 0]);
  }
  // 需要基组：用较密的 G
  const basis = structure(mat, { nG: 46, mesh: 2 }).basis;
  let vbm = -Infinity, cbm = Infinity, direct = Infinity;
  for (const k of pts) {
    const { values } = solveAt(clone, basis, k);
    const v = values[3], c = values[4];
    vbm = Math.max(vbm, v);
    cbm = Math.min(cbm, c);
    direct = Math.min(direct, c - v);
  }
  return { indirect: cbm - vbm, direct };
}

const targets = { Si: 1.12, Ge: 0.66, C: 5.47 };

for (const id of ["Si"]) {
  const mat = materialOf(id);
  let best = null;
  for (let v3 = -0.26; v3 <= -0.16; v3 += 0.02) {
    for (let v8 = 0.0; v8 <= 0.10; v8 += 0.02) {
      const g = gapOf(mat, { ...mat.vtab, 3: v3, 8: v8 });
      const err = Math.abs(g.indirect - targets[id]);
      if (!best || err < best.err) best = { v3, v8, ...g, err };
    }
  }
  console.log(`[${id}] best v3=${best.v3.toFixed(2)} v8=${best.v8.toFixed(2)} → gap ${best.indirect.toFixed(3)} eV, direct ${best.direct.toFixed(3)} (exp ${targets[id]})`);
}
