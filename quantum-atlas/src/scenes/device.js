/* L9 · 器件：半导体探测器（移植自 Fermi / PyQt 的八个结果页） */
import { C, V3, hex } from "../engine.js";
import { DETECTORS, monteCarlo, tempScan, fluxScan, polarization, attenuation, score, K_EDGES } from "../phys/device.js";
import { curve3 } from "./util.js";
import { solve } from "../solver/solve.js";

const VIEWS = ["器件剖面", "谱响应", "电荷收集", "内电场", "极化", "衰减", "通量", "温度扫描", "材料对比"];
const MATS = Object.keys(DETECTORS);
const MAT_LABELS = MATS.map((k) => DETECTORS[k].name);
export function refreshMaterialOptions() { MATS.length = 0; MATS.push(...Object.keys(DETECTORS)); MAT_LABELS.length = 0; MAT_LABELS.push(...MATS.map((k) => DETECTORS[k].name)); }

function chart(v, { x0 = -3, y0 = 0, w = 6, h = 2.4, samples, color = C.ice, color2 = null, samples2,
  title = "", xLabel = "", yLabel = "", log = false, fill = true, xTicks = 4, yTicks = 4, yMin = 0, yMax = null }) {
  const vals = samples.map((p) => p.y);
  const lo = yMin;
  const hi = yMax ?? Math.max(...vals, ...(samples2 ? samples2.map((p) => p.y) : [0])) * 1.08;
  const xlo = samples[0].x, xhi = samples[samples.length - 1].x;
  const nx = (x) => x0 + ((x - xlo) / (xhi - xlo || 1)) * w;
  const ny = (y) => y0 + (log
    ? (Math.log10(Math.max(y, 1e-6)) - Math.log10(Math.max(lo, 1e-6))) / (Math.log10(hi) - Math.log10(Math.max(lo, 1e-6)))
    : (y - lo) / (hi - lo || 1)) * h;

  // 轴与网格
  v.line([x0, y0, 0], [x0 + w, y0, 0], C.slate, 0.8);
  v.line([x0, y0, 0], [x0, y0 + h, 0], C.slate, 0.8);
  if (log) {
    // 对数轴按"十进位"打刻度：0.1 / 1 / 10 / 100 / 1e3
    const k0 = Math.floor(Math.log10(Math.max(1e-9, lo)));
    const k1 = Math.ceil(Math.log10(hi));
    for (let k = k0; k <= k1; k++) {
      const val = Math.pow(10, k);
      if (val < lo * 0.999 || val > hi * 1.001) continue;
      const yy = ny(val);
      v.line([x0, yy, 0], [x0 + w, yy, 0], C.slate, 0.07);
      const txt = k >= 3 ? `1e${k}` : k <= -3 ? `1e${k}` : String(val);
      v.label([x0 - 0.06, yy, 0], txt, { cls: "dim", offset: [-20, 0] });
    }
  } else {
    for (let i = 0; i <= yTicks; i++) {
      const yy = y0 + (i / yTicks) * h;
      v.line([x0, yy, 0], [x0 + w, yy, 0], C.slate, i === 0 ? 0 : 0.07);
      const yv = lo + (i / yTicks) * (hi - lo);
      const txt = Math.abs(yv) >= 100 ? yv.toFixed(0) : Math.abs(yv) >= 10 ? yv.toFixed(0) : yv.toFixed(yv < 1 ? 2 : 1);
      v.label([x0 - 0.06, yy, 0], txt, { cls: "dim", offset: [-20, 0] });
    }
  }
  for (let i = 0; i <= xTicks; i++) {
    const xx = x0 + (i / xTicks) * w;
    v.line([xx, y0, 0], [xx, y0 - 0.08, 0], C.slate, 0.6);
    // 横轴刻度值
    const xv = xlo + (i / xTicks) * (xhi - xlo);
    const txt = Math.abs(xv) >= 100 ? xv.toFixed(0) : Math.abs(xv) >= 10 ? xv.toFixed(0) : xv.toFixed(1);
    v.label([xx, y0 - 0.14, 0], txt, { cls: "dim", offset: [0, 10] });
  }
  v.label([x0 + w / 2, y0 - 0.26, 0], xLabel, { cls: "dim", offset: [0, 12] });
  v.label([x0 - 0.16, y0 + h * 0.55, 0], yLabel, { cls: "dim", offset: [-6, 0] });
  v.label([x0 + w / 2, y0 + h + 0.3, 0], title, { cls: "tag", offset: [0, 0] });

  const pts = samples.map((p) => [nx(p.x), ny(p.y), 0]);
  if (fill) {
    for (let i = 0; i < pts.length - 1; i++) {
      v.quad([pts[i][0], y0, 0], [pts[i + 1][0], y0, 0], pts[i + 1], pts[i], color, 0.10);
    }
  }
  curve3(v, pts, color, color, 0.95, 0.022);
  if (samples2) {
    const pts2 = samples2.map((p) => [nx(p.x), ny(p.y), 0.02]);
    curve3(v, pts2, color2 || C.amber, color2 || C.amber, 0.9, 0.018);
  }
  return { nx, ny, hi, lo };
}

export default {
  id: "device",
  name: "L9 · 器件",
  space: "实空间 · 器件尺度",
  spaceNote: "半导体探测器 · 电荷收集与谱响应",
  note:
    "把前面八层收束到一件真实器件：<b>半导体探测器</b>。光子在这里被光电吸收或康普顿散射，产生电子—空穴对；载流子在内建电场里漂移，能走多远由 <b>µτE</b> 决定（Hecht 方程），走不到电极的部分就变成谱的低能拖尾。漏电流、Fano 噪声、陷阱俘获（极化）共同决定能量分辨率。",
  params: [
    { key: "devMat", label: "探测器材料", type: "select", options: MATS, labels: MAT_LABELS },
    { key: "devView", label: "视图", type: "select", options: VIEWS },
    { key: "devThick", label: "厚度", type: "slider", min: 0.2, max: 5, step: 0.1, unit: "mm", hint: "越厚吸收越高，但收集变难" },
    { key: "devBias", label: "偏压", type: "slider", min: 10, max: 2000, step: 10, unit: "V", hint: "E = V/d，直接决定 µτE 与漏电流" },
    { key: "devT", label: "温度", type: "slider", min: 200, max: 500, step: 5, unit: "K", hint: "ni ∝ exp(−Eg/2kT)：温度一高噪声就上来" },
    { key: "devE", label: "光子能量", type: "slider", min: 10, max: 150, step: 1, unit: "keV", hint: "决定吸收系数与 K 边位置" },
    { key: "attnSrc", label: "衰减数据", type: "seg", options: ["精确 NIST", "快速近似"], hint: "精确档用 NIST XCOM 质量衰减表（14 元素，log–log 插值）；快速档是原程序的解析近似" },
  ],

  /** 剖面图用 3/4 视角；其余都是 2D 图表，必须正对相机才能读 */
  camera(S) {
    const view = S.devView || VIEWS[0];
    if (view === "器件剖面") return { dist: 11.2, pitch: 0.30, yaw: 0.62, target: [0, 0.55, 0] };
    return { dist: 8.9, pitch: 0.02, yaw: 0.0, target: [0, 1.35, 0] };
  },

  build(v, S) {
    const id = S.devMat || "CZT";
    const sol = solve({ kind: "device", spec: { material: id, d: S.devThick ?? 1, V: S.devBias ?? 300, E: S.devE ?? 60, T: S.devT ?? 300 } });
    const m = sol.data.raw.model;
    const view = S.devView || VIEWS[0];

    if (view === "器件剖面") this.drawDevice(v, S, m);
    else if (view === "温度扫描") this.drawTempScan(v, S, m);
    else if (view === "谱响应") this.drawSpectrum(v, S, m);
    else if (view === "电荷收集") this.drawTransport(v, S, m);
    else if (view === "内电场") this.drawField(v, S, m);
    else if (view === "极化") this.drawPolarization(v, S, m);
    else if (view === "衰减") this.drawAttenuation(v, S, m);
    else if (view === "通量") this.drawFlux(v, S, m);
    else this.drawCompare(v, S, m);
  },

  /* ── 1. 器件剖面：平板、电极、电场箭头、光子径迹、载流子漂移 ── */
  drawDevice(v, S, m) {
    const H = 1.2 + Math.min(2.2, m.thick * 0.55);   // 视觉厚度
    const W = 2.6, D = 1.5;
    const y0 = 0, y1 = H;

    // 体材料（半透明板）
    v.quad([-W, y0, -D], [W, y0, -D], [W, y1, -D], [-W, y1, -D], C.ice, 0.06);
    v.quad([-W, y0, D], [W, y0, D], [W, y1, D], [-W, y1, D], C.ice, 0.06);
    v.quad([-W, y0, -D], [-W, y0, D], [-W, y1, D], [-W, y1, -D], C.ice, 0.05);
    v.quad([W, y0, -D], [W, y0, D], [W, y1, D], [W, y1, -D], C.ice, 0.05);
    // 电极
    v.quad([-W, y0, -D], [W, y0, -D], [W, y0, D], [-W, y0, D], C.slate, 0.22);
    v.quad([-W, y1, -D], [W, y1, -D], [W, y1, D], [-W, y1, D], C.amber, 0.22);
    v.label([-W - 0.1, y0, D], "阴极 0 V", { cls: "dim", offset: [-8, 0] });
    v.label([-W - 0.1, y1, D], `阳极 +${Math.round(m.V)} V`, { cls: "dim", offset: [-14, 0] });

    // 电场：|E(z)| 决定箭头长度；俘获电荷让场畸变
    const prof = m.field;
    const nA = 9;
    for (let i = 0; i < nA; i++) {
      const zz = ((i + 0.5) / nA) * 2 * D - D;
      for (let j = 0; j < 7; j++) {
        const xx = -W + ((j + 0.5) / 7) * 2 * W;
        const yi = y0 + 0.12 + ((i + 0.5) / nA) * (H - 0.24);
        const idx = Math.min(prof.E.length - 1, Math.round((yi / H) * (prof.E.length - 1)));
        const Emag = Math.abs(prof.E[idx]) / Math.max(1e-9, Math.abs(prof.E[Math.floor(prof.E.length * 0.5)]));
        const len = Math.min(0.42, 0.10 + 0.3 * Emag);
        v.arrow([xx, yi, zz], [xx, yi + len, zz], C.ice, { head: 0.06, a: 0.30 + 0.35 * Math.min(1, Emag) });
      }
    }

    // 光子径迹 + 相互作用点
    const xin = -0.9, zin = 0.2;
    const hitY = y1 * 0.72;
    v.line([xin, y1 + 1.6, zin + 0.4], [xin, hitY, zin], C.amber, 0.9);
    v.thick([[xin, y1 + 1.6, zin + 0.4], [xin, hitY, zin]], C.amber, 0.016, 0.9);
    v.point([xin, hitY, zin], C.amber, 12, 1);
    v.label([xin + 0.2, y1 + 1.5, zin + 0.4], `${m.Einc} keV`, { cls: "tag" });

    // 载流子：电子向上（阳极），空穴向下（阴极），长度 ∝ 漂移长度
    const zH = m.d - (hitY / H) * m.d;
    const lamE = m.lamE, lamH = m.lamH;
    const unitY = H / m.d;
    const eLen = Math.min(H, lamE * unitY);
    const hLen = Math.min(H, lamH * unitY * 0.9);
    v.thick([[xin, hitY, zin], [xin, Math.min(y1 - 0.05, hitY + eLen), zin]], C.ice, 0.02, 0.95);
    v.point([xin, Math.min(y1 - 0.05, hitY + eLen), zin], C.ice, 8, 1);
    v.thick([[xin, hitY, zin], [xin, Math.max(0.05, hitY - hLen), zin]], C.pink, 0.02, 0.95);
    v.point([xin, Math.max(0.05, hitY - hLen), zin], C.pink, 8, 1);
    v.label([xin + 0.25, hitY + eLen * 0.6, zin], `e⁻  λ=${(lamE * 10).toFixed(2)} mm`, { cls: "dim", offset: [40, 0] });
    v.label([xin + 0.25, hitY - hLen * 0.6, zin], `h⁺  λ=${(lamH * 10).toFixed(2)} mm`, { cls: "dim", offset: [40, 0] });

    // 厚度标尺（左侧）
    const xr = -W - 0.32;
    v.line([xr, y0, -D], [xr, y1, -D], C.slate, 0.7);
    v.line([xr - 0.08, y0, -D], [xr + 0.08, y0, -D], C.slate, 0.9);
    v.line([xr - 0.08, y1, -D], [xr + 0.08, y1, -D], C.slate, 0.9);
    v.label([xr, (y0 + y1) / 2, -D], `d = ${m.thick.toFixed(1)} mm`, { cls: "tag", offset: [-34, 0] });
  },

  /* ── 2. 谱响应：MC 直方图 ───────────────────────────── */
  drawSpectrum(v, S, m) {
    const mc = monteCarlo(m.mat.id, { thickness: m.thick, voltage: m.V, energy: m.Einc, n: 3000, bins: 90 });
    const N = mc.counts.length;
    const x0 = -3.2, w = 6.4, h = 2.6;
    for (let i = 0; i < N; i++) {
      const x = x0 + (i / (N - 1)) * w;
      const y = mc.counts[i] * h;
      if (y < 0.004) continue;
      const col = mc.counts[i] > 0.75 ? C.amber : C.ice;
      v.quad([x - w / N / 2, 0, 0], [x + w / N / 2, 0, 0], [x + w / N / 2, y, 0], [x - w / N / 2, y, 0], col, 0.55);
    }
    v.line([x0, 0, 0], [x0 + w, 0, 0], C.slate, 0.8);
    v.label([x0 + w / 2, -0.3, 0], "沉积能量 (keV)", { cls: "dim" });
    v.label([x0, h + 0.2, 0], `谱响应 · ${m.mat.name} · ${m.Einc} keV`, { cls: "tag" });
    // 全能峰位置
    const xp = x0 + (mc.photopeak.energy / (m.Einc * 1.1)) * w;
    v.line([xp, 0, 0.02], [xp, mc.photopeak.frac * h, 0.02], C.amber, 0.9);
    v.label([xp, mc.photopeak.frac * h + 0.16, 0], `全能峰 ${mc.photopeak.energy.toFixed(1)} keV`, { cls: "tag" });
    // 低能拖尾标注
    v.label([x0 + w * 0.18, h * 0.55, 0], "CCE 低能拖尾", { cls: "dim" });
    v.label([x0 + w * 0.92, h * 0.42, 0], `吸收效率 ${(mc.efficiency * 100).toFixed(0)}%`, { cls: "dim" });
  },

  /* ── 3. 电荷收集：CCE(z) + 电场叠加 ─────────────────── */
  drawTransport(v, S, m) {
    const samples = [];
    const eSamp = [];
    const prof = m.field;
    for (let i = 0; i <= 60; i++) {
      const z = (i / 60) * m.d;
      samples.push({ x: z, y: hechtLocal(z, m) });
      const idx = Math.min(prof.E.length - 1, Math.round((i / 60) * (prof.E.length - 1)));
      eSamp.push({ x: z, y: Math.abs(prof.E[idx]) / Math.max(1, Math.abs(prof.E[0])) });
    }
    const c = chart(v, {
      x0: -3, y0: 0, w: 6, h: 2.6, samples, color: C.ice,
      samples2: eSamp, color2: C.amber,
      title: "电荷收集效率与内电场", xLabel: "距阴极深度 z (cm)", yLabel: "CCE / E(z)", yMin: 0, yMax: 1.15,
    });
    // 平均 CCE 参考线
    const yAvg = c.ny(m.cceAvg);
    v.line([-3, yAvg, 0.01], [3, yAvg, 0.01], C.lime, 0.55);
    v.label([2.2, yAvg + 0.1, 0.01], `⟨CCE⟩ = ${(m.cceAvg * 100).toFixed(1)}%`, { cls: "tag" });
    v.label([-2.9, 2.44, 0], "— CCE(z)", { cls: "dim" });
    v.label([-1.7, 2.44, 0], "— |E(z)|（归一）", { cls: "dim" });
  },

  /* ── 4. 内电场：V(z) 与 E(z) ────────────────────────── */
  drawField(v, S, m) {
    const prof = m.field;
    const vS = prof.z.map((z, i) => ({ x: z, y: prof.V[i] / Math.max(1e-9, m.V) }));
    const eS = prof.z.map((z, i) => ({ x: z, y: Math.abs(prof.E[i]) / Math.max(1, Math.abs(prof.E[0])) }));
    chart(v, {
      x0: -3, y0: 0, w: 6, h: 2.6, samples: vS, color: C.violet,
      samples2: eS, color2: C.ice,
      title: `内电场（偏压 ${Math.round(m.V)} V，E ≈ ${(m.E / 1000).toFixed(1)} kV/cm）`,
      xLabel: "深度 z (cm)", yLabel: "V/V₀ · |E|/E₀", yMin: 0, yMax: 1.15,
    });
    v.label([-2.9, 2.44, 0], "— 电位 V(z)", { cls: "dim" });
    v.label([-1.7, 2.44, 0], "— 电场 |E(z)|", { cls: "dim" });
    if (m.mat.pol > 0.15) v.label([0.2, 2.44, 0], "俘获电荷导致场畸变", { cls: "tag" });
  },

  /* ── 5. 极化：内场随时间的塌陷 ──────────────────────── */
  drawPolarization(v, S, m) {
    const pol = polarization(m.mat.id, { thickness: m.thick, voltage: m.V, temp: m.T });
    chart(v, {
      x0: -3, y0: 0, w: 6, h: 2.6, samples: pol.map((p) => ({ x: p.t, y: p.f })), color: C.pink,
      title: `极化效应（${m.mat.name}，陷阱俘获使内场衰减）`,
      xLabel: "时间 t (s)", yLabel: "归一化内场", yMin: 0.3, yMax: 1.05,
    });
    const last = pol[pol.length - 1].f;
    v.label([1.6, 1.9, 0], `100 s 后降至 ${(last * 100).toFixed(0)}%`, { cls: "tag" });
  },

  /* ── 6. 衰减系数：μ(E) + K 边 ───────────────────────── */
  drawAttenuation(v, S, m) {
    const Es = [];
    for (let e = 10; e <= 150; e += 1) Es.push(e);
    const mus = attenuation(m.mat, Es, S.attnSrc === "快速近似" ? "mock" : "nist");
    const samples = Es.map((e, i) => ({ x: e, y: Math.max(1e-3, mus[i]) }));
    chart(v, {
      x0: -3, y0: 0, w: 6, h: 2.6, samples, color: C.ice, log: true, yMin: 0.1,
      title: `线衰减系数 μ(E) · ${m.mat.name}`,
      xLabel: "光子能量 (keV)", yLabel: "μ (1/cm，对数)", xTicks: 5,
    });
    // K 边标注
    for (const [el, frac] of Object.entries(m.mat.comp)) {
      const ke = K_EDGES[el];
      if (!ke || ke < 10 || ke > 150) continue;
      const x = -3 + ((ke - 10) / 140) * 6;
      v.line([x, 0, 0.02], [x, 2.6, 0.02], C.amber, 0.45);
      v.label([x, 2.75, 0], `${el} K ${ke} keV`, { cls: "dim" });
    }
  },

  /* ── 7. 通量：分辨率与吞吐 ──────────────────────────── */
  drawFlux(v, S, m) {
    const fl = fluxScan(m.mat.id, { thickness: m.thick, voltage: m.V, temp: m.T, energy: m.Einc });
    const fw = fl.map((p) => ({ x: p.flux, y: p.fwhm }));
    const tp = fl.map((p) => ({ x: p.flux, y: p.throughput }));
    chart(v, {
      x0: -3, y0: 0, w: 6, h: 2.6, samples: fw, color: C.amber, samples2: tp, color2: C.ice,
      title: "高流强性能：分辨率退化与吞吐饱和",
      xLabel: "光子通量 (Mcps/mm²)", yLabel: "FWHM (keV) / 吞吐 (Mcps/mm²)",
    });
    v.label([-2.9, 2.44, 0], "— 能量分辨率 FWHM", { cls: "dim" });
    v.label([-1.2, 2.44, 0], "— 最大吞吐", { cls: "dim" });
  },

  /* ── 温度扫描：FWHM 与暗电流随 T 的两条曲线 ─────────── */
  drawTempScan(v, S, m) {
    const ts = tempScan(m.mat.id, {
      thickness: m.thick, voltage: m.V, energy: m.Einc,
      attn: S.attnSrc === "快速近似" ? "mock" : "nist",
    });
    const upper = ts.map((p) => ({ x: p.T, y: p.fwhm }));
    const lower = ts.map((p) => ({ x: p.T, y: Math.max(1e-6, p.dark) }));

    chart(v, {
      x0: -3, y0: 1.35, w: 6, h: 1.55, samples: upper, color: C.amber,
      title: "能量分辨率 FWHM vs 温度", yLabel: "FWHM (keV)", yMin: 0,
    });
    chart(v, {
      x0: -3, y0: -1.35, w: 6, h: 1.35, samples: lower, color: C.ice, log: true,
      title: "暗电流 vs 温度", xLabel: "温度 (K)", yLabel: "I_dark (nA，对数)",
      yMin: Math.min(...lower.map((p) => p.y)) * 0.7, xTicks: 6,
    });

    // 当前温度标记：两个面板各一条竖线
    const xT = -3 + ((m.T - 200) / 300) * 6;
    v.line([xT, -1.35, 0.02], [xT, 2.9, 0.02], C.slate, 0.35);
    v.label([xT, 2.98, 0], `${m.T} K`, { cls: "tag", offset: [0, -4] });
    v.label([-2.9, 2.6, 0], `ni(${m.T} K) = ${m.ni.toExponential(1)} cm⁻³`, { cls: "dim" });
  },

  /* ── 8. 材料对比：PCCT 评分柱状 ─────────────────────── */
  drawCompare(v, S, m) {
    const list = MATS.map((id) => ({ id, ...score(id, { thickness: m.thick, voltage: m.V, temp: m.T, energy: m.Einc, attn: S.attnSrc === "快速近似" ? "mock" : "nist" }) }));
    list.sort((a, b) => b.score - a.score);
    const x0 = -3.2, w = 6.4, h = 2.6;
    const bw = (w / list.length) * 0.62;
    v.line([x0, 0, 0], [x0 + w, 0, 0], C.slate, 0.8);
    list.forEach((it, i) => {
      const cx = x0 + (i + 0.5) * (w / list.length);
      const y = it.score * h;
      const col = it.id === m.mat.id ? C.amber : C.ice;
      v.quad([cx - bw / 2, 0, 0], [cx + bw / 2, 0, 0], [cx + bw / 2, y, 0], [cx - bw / 2, y, 0], col, 0.55);
      v.thick([[cx - bw / 2, y, 0.01], [cx + bw / 2, y, 0.01]], col, 0.014, 0.95);
      v.label([cx, -0.52, 0], it.mat.name, { cls: it.id === m.mat.id ? "tag" : "dim", offset: [0, 0] });
      v.label([cx, y + 0.12, 0], it.score.toFixed(2), { cls: "dim" });
    });
    v.label([0, h + 0.42, 0], "PCCT 适用性评分（吸收 34% · 分辨率 28% · 收集 22% · 通量 16%）", { cls: "tag" });
  },

  readout(S) {
    const id = S.devMat || "CZT";
    const sol = solve({ kind: "device", spec: { material: id, d: S.devThick ?? 1, V: S.devBias ?? 300, E: S.devE ?? 60, T: S.devT ?? 300 } });
    const m = sol.data.raw.model;
    const rows = [
      { k: "带隙 E_g", v: `${m.mat.Eg.toFixed(2)} eV` },
      { k: "µτ_e / µτ_h", v: `${(m.mat.mu_e * m.mat.tau_e).toExponential(1)} / ${(m.mat.mu_h * m.mat.tau_h).toExponential(1)} cm²/V`, cls: "good" },
      { k: "内建载流子 n_i", v: m.ni.toExponential(2) + " cm⁻³" },
      { k: "电阻率 ρ", v: m.rho > 1e12 ? "> 10¹² Ω·cm" : m.rho.toExponential(2) + " Ω·cm" },
      { k: "漏电流", v: `${(m.current * 1e9).toFixed(3)} nA` },
      { k: "电子学噪声", v: `${m.fwhmElec.toFixed(3)} keV` },
      { k: "平均 CCE", v: `${(m.cceAvg * 100).toFixed(1)} %`, cls: m.cceAvg > 0.8 ? "good" : "" },
      { k: "吸收效率", v: `${(m.eff * 100).toFixed(1)} %` },
      { k: "能量分辨率", v: `${m.fwhmTotal.toFixed(2)} keV FWHM` },
    ];
    return rows;
  },
};

/** 与 device.js 的 hecht 同式，避免在场景里再算一次模型 */
function hechtLocal(z, m) {
  const le = Math.max(m.lamE / m.d, 1e-9);
  const lh = Math.max(m.lamH / m.d, 1e-9);
  return le * (1 - Math.exp(-(m.d - z) / (m.lamE + 1e-12))) + lh * (1 - Math.exp(-z / (m.lamH + 1e-12)));
}
