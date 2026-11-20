/* L3 · 自旋：单原子 → 局域点阵 → 宏观磁化 */
import { C, V3, hex } from "../engine.js";
import { spinArrow, curve3, PAL, frame, plane } from "./util.js";
import { ramp } from "../engine.js";
import { atomicLevels, spinLattice, magnon, brillouin, hysteresis, precess } from "../phys/spin.js";
import { solve } from "../solver/solve.js";

const KINDS = [
  { v: "FM", label: "铁磁 FM" },
  { v: "AF", label: "反铁磁 AF" },
  { v: "stripe", label: "条纹" },
  { v: "spiral", label: "螺旋" },
  { v: "skyrmion", label: "斯格明子" },
];

export default {
  id: "spin",
  name: "L3 · 自旋",
  space: "实空间 + 参数空间",
  spaceNote: "S_i → M(r) → M(T)",
  note:
    "同一枚箭头在三个尺度上是三种东西：<b>单原子</b>里它是 SU(2) 的一个不可约表示（能级由 Zeeman + 各向异性劈裂）；<b>点阵</b>里它是局域矩，交换作用 J 把基态钉成磁序、把涨落量子化成磁振子；<b>宏观</b>里它粗粒化成 M(r,t)，由 Bloch/LLG 方程驱动。",
  time: {
    label: "时间 t",
    min: 0, max: 1, step: 0.002,
    get: (S) => S.spinT ?? 0,
    set: (S, val) => { S.spinT = val; },
    fmt: (S) => (S.spinT ?? 0).toFixed(2),
  },
  params: [
    { key: "spinMode", label: "尺度", type: "seg", options: ["单原子", "点阵", "宏观"], hint: "同一物理量的三种粗粒化层级" },
    { key: "spinKind", label: "磁序", type: "select", options: KINDS.map((k) => k.v), labels: KINDS.map((k) => k.label) },
    { key: "J", label: "交换 J", type: "slider", min: 0.2, max: 3, step: 0.05, unit: "meV" },
    { key: "J2", label: "次近邻 J₂", type: "slider", min: -0.5, max: 1, step: 0.05, unit: "meV", hint: "解析式写不出 J₂，这里靠动力学矩阵对角化" },
    { key: "Danis", label: "各向异性 D", type: "slider", min: 0, max: 0.6, step: 0.01, unit: "meV" },
    { key: "Bz", label: "磁场 B", type: "slider", min: 0, max: 3, step: 0.05, unit: "T" },
    { key: "Temp", label: "温度 T", type: "slider", min: 1, max: 600, step: 1, unit: "K" },
  ],

  build(v, S) {
    const mode = S.spinMode || "点阵";
    const t = (S.spinT ?? 0) * 6;
    if (mode === "单原子") return this.buildAtom(v, S, t);
    if (mode === "宏观") return this.buildMacro(v, S, t);
    return this.buildLattice(v, S, t);
  },

  buildAtom(v, S, t) {
    const S_ = 1;
    const lv = atomicLevels({ S: S_, D: S.Danis ?? 0.3, g: 2, B: S.Bz ?? 1 });
    const dMax = Math.max(...lv.map((l) => l.dE), 1e-3);
    const sc = 2.6 / dMax;

    // 原子
    v.point([0, 0, 0], C.amber, 16, 1);
    v.sphereWire([0, 0, 0], 0.55, C.slate, { a: 0.16, seg: 8 });

    // 进动锥
    const th = 0.5;
    const omega = 1.6 * (S.Bz ?? 1);
    const m = precess(th, omega, t);
    const tip = V3.mul(m, 1.9);
    v.arrow([0, 0, 0], tip, C.pink, { head: 0.26, a: 1 });
    const cone = [];
    for (let i = 0; i <= 40; i++) {
      const ph = (i / 40) * Math.PI * 2;
      cone.push([1.9 * Math.sin(th) * Math.cos(ph), 1.9 * Math.cos(th), 1.9 * Math.sin(th) * Math.sin(ph)]);
    }
    v.poly(cone, C.pink, { close: true, a: 0.18 });
    v.pick(tip, "spin", 0.3, { m });

    // 能级标尺（右侧）
    const x0 = 3.2;
    v.line([x0, -0.4, 0], [x0, 3.0, 0], C.slate, 0.25);
    lv.forEach((l, i) => {
      const y = 0.2 + l.dE * sc;
      v.line([x0 - 0.5, y, 0], [x0 + 1.3, y, 0], l.m === -S_ || l.m === S_ ? C.cyan : C.violet, 0.7);
      v.label([x0 + 1.45, y, 0], `m = ${l.m}   ${l.dE.toFixed(2)} meV`, { cls: "dim", offset: [0, -8] });
      v.point([x0 - 0.5, y, 0], C.white, 4, 0.8);
    });
    v.label([x0 + 0.4, 3.2, 0], "Zeeman + 单轴各向异性", { cls: "tag" });
    v.label([0, -1.9, 0], `S = ${S_} · g = 2 · B = ${(S.Bz ?? 1).toFixed(2)} T`, { cls: "dim" });
  },

  buildLattice(v, S, t) {
    const N = 9;
    const spins = spinLattice({
      kind: S.spinKind || "FM", N, a: 0.55, q: 0.55, R: 1.9, t: (S.spinT ?? 0) * 2.2,
    });
    const J = S.J ?? 1, D = S.Danis ?? 0.05;
    // 交换键（仅最近邻，示意）
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        const a = spins[i * N + j];
        if (j < N - 1) v.line(a.p, spins[i * N + j + 1].p, C.slate, 0.20);
        if (i < N - 1) v.line(a.p, spins[(i + 1) * N + j].p, C.slate, 0.20);
      }
    for (const sp of spins) {
      const col = V3.lerp(C.pink, C.ice, (sp.m[1] + 1) / 2);
      spinArrow(v, sp.p, sp.m, 0.40, col, 0.95);
    }
    v.label([0, 0.15, -2.9], `${S.spinKind || "FM"} · ${N}×${N} 经典矩`, { cls: "tag" });

    // 磁振子：S(q,ħω) 强度图（中子散射惯例）+ 色散脊线
    // 动力学矩阵对角化（支持 J2 与双子晶格）；取最高支作为脊线
    const brA = solve({ kind: "spin", spec: { kind: S.spinKind === "AF" ? "AF" : "FM", S: 1, J1: J, J2: S.J2 ?? 0, D, n: 90 } }).data;
    const mg = { q: brA.q, w: brA.q.map((_, i) => Math.max(...brA.w.map((b) => b[i]))) };
    const x0 = 3.05, wSpan = 2.2, hSpan = 1.25;
    const wMax = Math.max(...mg.w, 1e-6);
    const kT = 0.08617 * (S.Temp ?? 300);        // meV
    const gamma = Math.max(wMax * 0.035, 0.05);  // 仪器分辨率（Lorentzian 半宽）

    // 强度图：每个 q 上按 Lorentzian 展宽色散，乘玻色占据因子；先算再按峰值归一
    const NQ = 54, NE = 30;
    const cells = [];
    let Imax = 1e-9;
    for (let iq = 0; iq < NQ; iq++) {
      const qf = iq / (NQ - 1);
      const qi = Math.min(mg.q.length - 1, Math.round(qf * (mg.q.length - 1)));
      const w0 = (mg.w[qi] / wMax) * hSpan;
      const x = x0 + qf * wSpan;
      for (let ie = 0; ie < NE; ie++) {
        const E = ((ie + 0.5) / NE) * hSpan * 1.08;
        const lor = (gamma / Math.PI) / ((E - w0) ** 2 + gamma * gamma);
        const occ = Math.min(6, 1 / (1 - Math.exp(-(E / Math.max(kT, 1e-3)))));
        const I = lor * occ * wMax * 0.35;
        if (I > Imax) Imax = I;
        cells.push({ x, ie, I });
      }
    }
    const dy = (hSpan * 1.08) / NE;
    for (const c of cells) {
      const In = Math.pow(Math.min(1, c.I / Imax), 0.42);
      if (In < 0.03) continue;
      const col = ramp(In, [C.ice, C.violet, C.amber, C.pink]);
      v.quad([c.x, c.ie * dy, 0], [c.x + wSpan / NQ, c.ie * dy, 0],
        [c.x + wSpan / NQ, c.ie * dy + dy, 0], [c.x, c.ie * dy + dy, 0], col, 0.10 + 0.55 * In);
    }

    v.line([x0, 0, 0.01], [x0 + wSpan, 0, 0.01], C.slate, 0.45);
    v.line([x0, 0, 0.01], [x0, hSpan + 0.15, 0.01], C.slate, 0.35);
    const pts = mg.q.map((q, i) => [x0 + (wSpan * i) / (mg.q.length - 1), (mg.w[i] / wMax) * hSpan, 0.02]);
    curve3(v, pts, C.ice, C.violet, 0.95, 0.014);
    v.label([x0 + wSpan * 0.5, hSpan + 0.44, 0], "磁振子 S(q, ħω)", { cls: "tag" });
    v.label([x0 + wSpan + 0.1, -0.02, 0], "q", { cls: "dim", offset: [10, 0] });
    v.label([x0 - 0.18, hSpan * 0.62, 0], `${wMax.toFixed(1)} meV`, { cls: "dim", offset: [-26, 0] });
    v.label([x0 + wSpan * 0.02, hSpan * 1.06, 0], "强度 ↑", { cls: "dim", offset: [8, 0] });
    // Γ–X–M–Γ 标记
    ["Γ", "X", "M", "Γ"].forEach((lab, i) => {
      const x = x0 + (wSpan * i) / 3;
      v.line([x, -0.05, 0], [x, 0.05, 0], C.slate, 0.5);
      v.label([x, -0.18, 0], lab, { cls: "dim", offset: [0, 8] });
    });
  },

  buildMacro(v, S, t) {
    const T = S.Temp ?? 300;
    const Tc = 320;
    // M(T) 曲线
    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const tt = (i / 60) * 650;
      const m = brillouin(tt, { S: 1, Tc });
      pts.push([-3.2 + (6.4 * i) / 60, m * 1.9, 0]);
    }
    curve3(v, pts, C.cyan, C.violet, 0.95);
    v.line([-3.2, 0, 0], [3.2, 0, 0], C.slate, 0.3);
    v.line([-3.2, 0, 0], [-3.2, 2.2, 0], C.slate, 0.3);
    v.label([3.25, 0.05, 0], "T", { cls: "dim" });
    v.label([-3.5, 2.1, 0], "M/M₀", { cls: "dim" });

    // 当前温度标记
    const mNow = brillouin(T, { S: 1, Tc });
    const xT = -3.2 + (6.4 * Math.min(650, T)) / 650;
    v.line([xT, 0, 0], [xT, mNow * 1.9, 0], C.amber, 0.5);
    v.point([xT, mNow * 1.9, 0], C.amber, 9, 1);
    v.label([xT, mNow * 1.9 + 0.2, 0], `T = ${T} K → M = ${mNow.toFixed(2)}`, { cls: "tag" });

    // 磁滞回线（右侧平面 z = -2.4）
    const z = -2.4;
    const Hs = Array.from({ length: 120 }, (_, i) => -180 + (360 * i) / 119);
    const up = hysteresis(Hs, { Hc: 80, Ms: 1 });
    const dn = hysteresis([...Hs].reverse(), { Hc: -80, Ms: 1 }).reverse();
    const ox = 0.2;
    v.line([ox, 0, z], [ox, 0, z - 0.01], C.slate, 0.2);
    const p1 = up.map((m, i) => [ox + (Hs[i] / 200) * 2.6, m * 1.2, z]);
    const p2 = dn.map((m, i) => [ox + (Hs[i] / 200) * 2.6, m * 1.2, z]);
    curve3(v, p1, C.pink, C.pink, 0.9);
    curve3(v, p2, C.amber, C.amber, 0.75);
    v.label([ox + 1.3, 1.5, z], "磁滞回线 M(H)", { cls: "tag" });

    // 磁化纹理
    for (let i = 0; i < 26; i++) {
      const x = -3 + (6 * i) / 25;
      const ph = 0.6 * x + t * 0.8;
      const mv = [0, Math.tanh(2.2 * Math.sin(ph)), 0];
      spinArrow(v, [x, -1.6, 0], V3.norm(mv), 0.32, V3.lerp(C.pink, C.cyan, (mv[1] + 1) / 2), 0.9);
    }
    v.label([0, -2.05, 0], "畴壁 / 纹理 M(r)", { cls: "dim" });
  },

  readout(S) {
    const mode = S.spinMode || "点阵";
    if (mode === "单原子") {
      const lv = atomicLevels({ S: 1, D: S.Danis ?? 0.3, g: 2, B: S.Bz ?? 1 });
      const split = Math.max(...lv.map((l) => l.dE));
      return [
        { k: "自旋 S", v: "1" },
        { k: "Zeeman 劈裂", v: `${(Math.abs(lv[0].dE - Math.max(...lv.map((l) => l.dE)))).toFixed(3)} meV` },
        { k: "最大能级差", v: `${split.toFixed(3)} meV` },
        { k: "Larmor ω", v: `${(2 * Math.PI * 28 * (S.Bz ?? 1)).toFixed(1)} GHz` },
      ];
    }
    if (mode === "宏观") {
      const T = S.Temp ?? 300;
      const m = brillouin(T, { S: 1, Tc: 320 });
      return [
        { k: "约化磁化 m", v: m.toFixed(3), cls: m > 0.5 ? "good" : "bad" },
        { k: "居里温度 Tc", v: "320 K" },
        { k: "相", v: m > 0.01 ? "有序（铁磁）" : "顺磁" },
        { k: "矫顽场 Hc", v: "80 mT" },
      ];
    }
    const model = {
      kind: S.spinKind === "AF" ? "AF" : "FM",
      S: 1,
      J1: S.J ?? 1,
      J2: S.J2 ?? 0,
      D: S.Danis ?? 0.05,
      n: 60,
    };
    const br = solve({ kind: "spin", spec: model }).data;
    const wMax = Math.max(...br.w.flat());
    const w0 = Math.max(...br.w.map((b) => b[0]));
    return [
      { k: "交换 J", v: `${(S.J ?? 1).toFixed(2)} meV` },
      { k: "各向异性 D", v: `${(S.Danis ?? 0.05).toFixed(2)} meV` },
      { k: "磁振子带宽", v: `${wMax.toFixed(2)} meV` },
      { k: "能隙 ω(Γ)", v: `${w0.toFixed(2)} meV` },
      { k: "ω(X)", v: `${Math.max(...br.w.map((b) => b[Math.round(b.length / 3)])).toFixed(2)} meV` },
      { k: "动力学矩阵", v: `${br.w.length} 支 · ${br.w.length * 2}×${br.w.length * 2}` },
      { k: "自旋波速", v: `${(2 * (S.J ?? 1) * Math.SQRT2).toFixed(2)} (meV·a)` },
      { k: "基态", v: S.spinKind || "FM" },
    ];
  },
};
