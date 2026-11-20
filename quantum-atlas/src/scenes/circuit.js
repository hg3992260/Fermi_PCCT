/* L8 · 量子电路：门序列 → 态演化塔（3D） + 电路编辑器（DOM） */
import { C, V3, hex } from "../engine.js";
import { bar, plane } from "./util.js";
import { GATE_LIST, countBits } from "../phys/quantum.js";
import { solve } from "../solver/solve.js";

const NSTEPS = 6;

function defaultOps() {
  return [
    { gate: "H", qubit: 0, step: 0 },
    { gate: "H", qubit: 1, step: 0 },
    { gate: "CNOT", control: 0, qubit: 1, step: 1 },
    { gate: "Ry(π/2)", qubit: 0, step: 2 },
    { gate: "Z", qubit: 1, step: 3 },
  ];
}

export default {
  id: "circuit",
  name: "L8 · 量子电路",
  space: "门空间 + 时间",
  spaceNote: "酉序列 / 演化塔",
  note:
    "一个电路就是一段<b>分段常数的哈密顿量</b>：每个门 = 一次酉演化 U = e^{−iHτ}，整条电路 = 这些 U 的乘积。右侧的演化塔把「每一步之后、每个基态上的概率幅」同时画出来——纵向是时间，横向是希尔伯特空间，高度是 |amplitude|，颜色是相位。",
  time: {
    label: "步进",
    min: 0, max: 1, step: 0.001,
    get: (S) => (S.step ?? 0) / NSTEPS,
    set: (S, val) => { S.step = Math.round(val * NSTEPS); },
    fmt: (S) => `第 ${S.step ?? 0} / ${NSTEPS} 步`,
  },
  params: [
    { key: "nq2", label: "比特数", type: "slider", min: 1, max: 3, step: 1 },
    { key: "circuitPreset", label: "示例", type: "select", options: ["Bell", "GHZ", "旋转+测量", "清空"], labels: ["Bell 对", "GHZ 态", "旋转 + 测量", "清空"] },
    { key: "noisy", label: "含噪声（密度矩阵）", type: "switch" },
    { key: "qT1", label: "T₁", type: "slider", min: 1, max: 40, step: 0.5, unit: "µs" },
    { key: "qT2", label: "T₂", type: "slider", min: 0.5, max: 20, step: 0.5, unit: "µs" },
    { key: "qGateNs", label: "单门耗时", type: "slider", min: 5, max: 500, step: 5, unit: "ns" },
  ],

  ensure(S) {
    if (!S.ops) S.ops = defaultOps();
    if (S.nSteps == null) S.nSteps = NSTEPS;
    if (S.step == null) S.step = 0;
    if (S.tool == null) S.tool = "H";
    if (S.nq2 == null) S.nq2 = 2;
    return S;
  },

  applyPreset(S) {
    const p = S.circuitPreset || "Bell";
    if (p === "Bell") { S.nq2 = 2; S.ops = defaultOps(); }
    else if (p === "GHZ") {
      S.nq2 = 3;
      S.ops = [
        { gate: "H", qubit: 0, step: 0 },
        { gate: "CNOT", control: 0, qubit: 1, step: 1 },
        { gate: "CNOT", control: 1, qubit: 2, step: 2 },
      ];
    } else if (p === "旋转+测量") {
      S.nq2 = 1;
      S.ops = [
        { gate: "H", qubit: 0, step: 0 },
        { gate: "Rz(π/2)", qubit: 0, step: 1 },
        { gate: "Ry(π/2)", qubit: 0, step: 2 },
        { gate: "M", qubit: 0, step: 3 },
      ];
    } else S.ops = [];
    S.step = S.nSteps;
  },

  build(v, S) {
    this.ensure(S);
    const n = Math.max(1, Math.round(S.nq2));
    const sol = solve({ kind: "circuit", spec: { n, ops: S.ops, steps: S.nSteps } });
    const timeline = sol.data.raw.timeline;
    const dim = 1 << n;
    const ns = timeline.length;

    // 演化塔
    const sx = Math.min(0.72, 5.4 / dim);
    const sz = 0.62;
    const x0 = -(dim * sx) / 2 + sx / 2;
    plane(v, -0.02, 3.4, C.slate, 0.03, 0);

    for (let s = 0; s < ns; s++) {
      const st = timeline[s];
      const isNow = s === S.step;
      for (let i = 0; i < dim; i++) {
        const amp = Math.hypot(st[i][0], st[i][1]);
        if (amp < 1e-4) continue;
        const ph = Math.atan2(st[i][1], st[i][0]);
        const col = V3.lerp(C.ice, C.amber, (ph + Math.PI) / (2 * Math.PI));
        bar(v, [x0 + i * sx, 0, (s - ns / 2) * sz], sx * 0.3, amp * 2.0, col, isNow ? 0.9 : 0.5);
      }
      v.line([x0 - sx, 0.01, (s - ns / 2) * sz], [x0 + (dim - 0.5) * sx, 0.01, (s - ns / 2) * sz], isNow ? C.ice : C.slate, isNow ? 0.4 : 0.12);
      v.label([x0 + (dim - 0.5) * sx + 0.15, 0.05, (s - ns / 2) * sz], `t${s}`, { cls: isNow ? "tag" : "dim", offset: [10, 0] });
    }

    // 基矢标签
    for (let i = 0; i < dim; i++) {
      v.label([x0 + i * sx, -0.12, (ns / 2) * sz + 0.25], i.toString(2).padStart(n, "0"), { cls: "dim", offset: [0, 8] });
    }
    v.label([0, 2.35, 0], `|ψ(t)⟩ 演化塔 · ${n} qubit`, { cls: "tag" });

    // 相位图例
    const lg = [];
    for (let i = 0; i <= 12; i++) {
      const t = i / 12;
      lg.push([-2.6 + 1.2 * t, -0.15, (ns / 2) * sz + 0.9]);
    }
    for (let i = 0; i < lg.length - 1; i++) {
      v.line(lg[i], lg[i + 1], V3.lerp(C.ice, C.amber, i / 12), 0.9);
    }
    v.label([-2.0, 0.05, (ns / 2) * sz + 0.9], "相位 −π → +π", { cls: "dim" });
  },

  /* ── DOM：电路编辑器 ─────────────────────────────── */
  mount(host, S, api) {
    this.ensure(S);
    this.unmount();
    const wrap = document.createElement("div");
    wrap.className = "circuit-wrap";
    wrap.id = "circuit-ui";

    const palette = document.createElement("div");
    palette.className = "circuit-palette";
    GATE_LIST.forEach((g) => {
      const b = document.createElement("button");
      b.className = "gate-chip" + (S.tool === g ? " on" : "");
      b.textContent = g;
      b.onclick = () => {
        S.tool = g;
        api.refresh();
      };
      palette.append(b);
    });

    const canvas = document.createElement("div");
    canvas.className = "circuit-canvas";
    const grid = document.createElement("div");
    grid.className = "circuit-grid";
    grid.style.gridTemplateColumns = `54px repeat(${S.nSteps}, minmax(38px, 1fr))`;

    const n = S.nq2;
    for (let q = 0; q < n; q++) {
      const lab = document.createElement("div");
      lab.className = "wire-label";
      lab.textContent = `q${q}`;
      grid.append(lab);
      for (let s = 0; s < S.nSteps; s++) {
        const cell = document.createElement("div");
        cell.className = "cell";
        const op = S.ops.find((o) => o.step === s && (o.qubit === q || (o.control === q && ["CNOT", "CZ"].includes(o.gate))));
        if (op) {
          cell.classList.add("filled");
          if (op.gate === "CNOT" || op.gate === "CZ") {
            cell.textContent = op.control === q ? "●" : op.gate === "CNOT" ? "⊕" : "◎";
            if (op.control === q) cell.classList.add("ctrl");
          } else {
            cell.textContent = op.gate;
            cell.classList.add("gate-1q");
          }
        }
        if (s === S.step) cell.classList.add("now");
        cell.onclick = () => {
          if (S.tool === "清空") return;
          const existing = S.ops.findIndex((o) => o.step === s && (o.qubit === q || o.control === q));
          if (existing >= 0) S.ops.splice(existing, 1);
          else if (["CNOT", "CZ"].includes(S.tool)) S.ops.push({ gate: S.tool, control: q, qubit: (q + 1) % n, step: s });
          else S.ops.push({ gate: S.tool, qubit: q, step: s });
          api.refresh();
        };
        cell.oncontextmenu = (e) => {
          e.preventDefault();
          S.ops = S.ops.filter((o) => !(o.step === s && (o.qubit === q || o.control === q)));
          api.refresh();
        };
        grid.append(cell);
      }
    }
    canvas.append(grid);
    const hint = document.createElement("div");
    hint.className = "circuit-hint";
    hint.innerHTML = `当前工具：<b>${S.tool}</b> · 左键放置/替换 · 右键删除 · 双比特门自动把下一根线当作目标。拖底部时间轴可逐步查看演化。`;
    canvas.append(hint);

    wrap.append(palette, canvas);
    host.append(wrap);
    this._wrap = wrap;
  },

  unmount() {
    if (this._wrap) { this._wrap.remove(); this._wrap = null; }
  },

  readout(S) {
    this.ensure(S);
    const n = Math.max(1, Math.round(S.nq2));
    const sol = solve({ kind: "circuit", spec: { n, ops: S.ops, steps: S.nSteps } });
    const timeline = sol.data.raw.timeline;
    const st = timeline[Math.min(S.step ?? 0, timeline.length - 1)];
    const probs = st.map((a) => a[0] * a[0] + a[1] * a[1]);
    const top = probs
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)[0];
    const rows = [
      { k: "门数", v: `${S.ops.length}` },
      { k: "比特数", v: `${n}` },
      { k: "当前步", v: `${S.step ?? 0} / ${S.nSteps}` },
      { k: "最概然态", v: `|${top.i.toString(2).padStart(n, "0")}⟩  ${(top.p * 100).toFixed(1)}%`, cls: "good" },
      { k: "参与基态", v: `${probs.filter((p) => p > 1e-6).length} / ${1 << n}` },
    ];
    // 含噪：密度矩阵 + 振幅/相位阻尼，与 L6 的 T1/T2 共用同一组参数
    if (S.noisy) {
      try {
        const noiseSol = solve({ kind: "circuit", spec: { n, ops: S.ops, steps: S.nSteps, noise: { T1: S.qT1 ?? 6, T2: S.qT2 ?? 2.2, gateTime: (S.qGateNs ?? 20) / 1000 } } });
        const frames = noiseSol.data.frames;
        const f = frames[Math.min(S.step ?? 0, frames.length - 1)];
        rows.push({ k: "保真度 F", v: f.F.toFixed(4), cls: f.F > 0.95 ? "good" : undefined });
        rows.push({ k: "纯度 Tr(ρ²)", v: f.purity.toFixed(3) });
        if (n === 2) rows.push({ k: "concurrence", v: f.concurrence.toFixed(3) });
        rows.push({ k: "T₁ / T₂", v: `${(S.qT1 ?? 6).toFixed(1)} / ${(S.qT2 ?? 2.2).toFixed(1)} µs` });
        rows.push({ k: "单门耗时", v: `${S.qGateNs ?? 20} ns` });
      } catch (e) {
        rows.push({ k: "噪声", v: "求解失败：" + e.message.slice(0, 30) });
      }
    }
    return rows;
  },
};
