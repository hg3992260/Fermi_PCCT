/* 场景共用绘制工具 */
import { C, V3, ramp, hex } from "../engine.js";
import { brillouinZone } from "../phys/epm.js";

export const PAL = {
  realspace: C.amber,
  kspace: C.ice,
  statespace: C.violet,
  param: C.lime,
  hot: [hex("#ff6b6b"), hex("#ffb84d"), hex("#fff1a8")],
  cool: [hex("#1b3a6b"), hex("#4c9aff"), hex("#b8e0ff")],
  thermal: [hex("#02787d"), hex("#009999"), hex("#3abfed"), hex("#ec6602"), hex("#7a162d")],
};

/** 能带能量 → 颜色（冷→热） */
export const energyColor = (dE, range = 12) => ramp((dE + range * 0.35) / range, PAL.thermal);

/** 半透明水平面 */
export function plane(v, y, size = 6, color = C.ice, alpha = 0.06, grid = 0) {
  const s = size;
  v.quad([-s, y, -s], [s, y, -s], [s, y, s], [-s, y, s], color, alpha);
  if (grid) {
    for (let i = -size; i <= size; i += grid) {
      v.line([i, y, -size], [i, y, size], color, 0.08);
      v.line([-size, y, i], [size, y, i], color, 0.08);
    }
  }
}

/** 布里渊区线框（缩放到 r） */
export function drawBZ(v, mat, { offset = [0, 0, 0], r = 2.2, color = C.ice, alpha = 0.5, labels = false, width = 0 } = {}) {
  const bz = brillouinZone(mat);
  const maxR = Math.max(...bz.verts.map((p) => Math.hypot(...p)));
  const s = r / maxR;
  const P = (p) => [p[0] * s + offset[0], p[1] * s + offset[1], p[2] * s + offset[2]];
  for (const f of bz.faces) {
    const pts = f.verts.map(P);
    v.poly(pts, color, { close: true, a: alpha, width });
  }
  if (labels) {
    // 只标 Γ（原点）与顶点最多处；高对称点的完整标注留给能带场景
    v.point(P([0, 0, 0]), color, 4, 0.9);
    v.label(P([0, 0, 0]), "Γ", { cls: "dim", offset: [0, -10] });
    const top = bz.verts.reduce((a, b) => (b[1] > a[1] ? b : a), bz.verts[0]);
    const q = P(top);
    v.point(q, color, 3, 0.7);
  }
  return { s, P };
}

/** 磁矩箭头（归一化到 len） */
export function spinArrow(v, p, m, len = 0.5, color = C.pink, alpha = 1) {
  const tip = V3.add(p, V3.mul(m, len));
  // 杆做成带状、尖端放一个受光小球：比 1px 线更像"一根磁矩"
  v.point(p, color, 3.4, alpha * 0.85);
  v.thick([p, tip], color, 0.015, alpha);
  v.point(tip, color, 7.5, alpha);
}

/** 坐标轴 + 网格 */
export function frame(v, { size = 4, grid = 1, axes = true, labels = ["x", "y", "z"] } = {}) {
  if (grid) v.grid(size, grid, C.slate, 0.07);
  if (axes) v.axes(size * 0.9, labels);
}

/** 三维曲线（带渐变） */
export function curve3(v, pts, colorA, colorB, alpha = 0.9, width = 0) {
  const n = pts.length;
  for (let i = 0; i < n - 1; i++) {
    const t = i / Math.max(1, n - 2);
    const c = Array.isArray(colorA) && Array.isArray(colorB) && colorA.length === 3 && colorB.length === 3
      ? V3.lerp(colorA, colorB, t)
      : colorA;
    if (width > 0) v.thick([pts[i], pts[i + 1]], c, width, alpha);
    else v.line(pts[i], pts[i + 1], c, alpha);
  }
}

/** 竖直虚线（能级标记） */
export function dashLine(v, a, b, color, dash = 0.12, alpha = 0.5) {
  const d = V3.sub(b, a), L = V3.len(d);
  const u = V3.mul(d, 1 / L);
  const n = Math.max(1, Math.floor(L / (dash * 2)));
  for (let i = 0; i < n; i++) {
    const s = i * dash * 2;
    v.line(V3.add(a, V3.mul(u, s)), V3.add(a, V3.mul(u, Math.min(L, s + dash))), color, alpha);
  }
}

/** 柱状（用于态概率、演化塔） */
export function bar(v, base, w, h, color, alpha = 0.75) {
  const [x, y, z] = base;
  const y2 = y + h;
  const a = [x - w, y, z - w], b = [x + w, y, z - w], c = [x + w, y, z + w], d = [x - w, y, z + w];
  const A = [x - w, y2, z - w], Bq = [x + w, y2, z - w], Cq = [x + w, y2, z + w], D = [x - w, y2, z + w];
  v.quad(a, b, c, d, color, alpha * 0.5);
  v.quad(A, Bq, Cq, D, color, alpha);
  v.quad(a, b, Bq, A, color, alpha * 0.8);
  v.quad(b, c, Cq, Bq, color, alpha * 0.8);
  v.quad(c, d, D, Cq, color, alpha * 0.8);
  v.quad(d, a, A, D, color, alpha * 0.8);
}
