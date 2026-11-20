/* ═══════════════════════════════════════════════════════════════
   main.js — 外壳（v2）
   职责：把场景参数写进 S、请求一帧、把读数渲染成表。
   场景只实现 build(v,S) / readout(S)，外壳不关心任何物理。
   ═══════════════════════════════════════════════════════════════ */

import { View3D, setPalette, paletteMode, C } from "./engine.js";
import { DETECTORS, saveCustomMaterial, removeCustomMaterial } from "./phys/device.js";
import { refreshMaterialOptions } from "./scenes/device.js";
import { glyphSVG } from "./glyphs.js";
import { CASES, BOUNDARY } from "./cases.js";
import { initCasebook, openCasebook, closeCasebook } from "./casebook.js";
import { initHud, showLayer, hudContextAt } from "./hud.js";
import { initMcpDock } from "./mcp-dock.js";
import { collect as collectExport, toCSV, filenameFor } from "./export.js";
import { kindOf, noticeOf, KINDS } from "./provenance.js";
import { solve, kinds as solverKinds } from "./solver/solve.js";
import atlas from "./scenes/atlas.js";
import crystal from "./scenes/crystal.js";
import bands from "./scenes/bands.js";
import spin from "./scenes/spin.js";
import spectra from "./scenes/spectra.js";
import liquid from "./scenes/liquid.js";
import bloch from "./scenes/bloch.js";
import qubit from "./scenes/qubit.js";
import circuit from "./scenes/circuit.js";
import device from "./scenes/device.js";
import { materialOf } from "./phys/materials.js";
import { structure } from "./phys/epm.js";

/* 开发期错误可见化 */
window.__errs = [];
function noteError(msg) {
  window.__errs.push(msg);
  if (!/[?&]audit=/.test(location.search)) document.title = "ERR: " + window.__errs.slice(-2).join(" | ");
  const tip = document.getElementById("status-tip");
  if (tip) tip.textContent = "⚠ " + msg;
}
window.addEventListener("error", (e) => noteError(`${e.message} @ ${(e.filename || "").split("/").pop()}:${e.lineno}`));
window.addEventListener("unhandledrejection", (e) => noteError(String(e.reason)));

const SCENES = { atlas, crystal, bands, spin, spectra, liquid, bloch, qubit, circuit, device };
const ORDER = ["atlas", "crystal", "bands", "spin", "spectra", "liquid", "bloch", "qubit", "circuit", "device"];

const CAM = {
  atlas: { dist: 13.5, pitch: 0.34, yaw: 0.55, target: [0, 0, 0] },
  crystal: { dist: 12.2, pitch: 0.42, yaw: 0.95, target: [0, 0.35, 0] },
  bands: { dist: 13, pitch: 0.34, yaw: 0.55, target: [1.4, 0, 0] },
  spin: { dist: 13.5, pitch: 0.46, yaw: 0.62, target: [0.9, 0.2, 0] },
  spectra: { dist: 9.8, pitch: 0.33, yaw: 0.58, target: [-0.45, 0.0, -0.75] },
  liquid: { dist: 13.5, pitch: 0.36, yaw: 0.68, target: [0.4, 0, 0] },
  bloch: { dist: 8.4, pitch: 0.28, yaw: 0.68, target: [0, 0, 0] },
  qubit: { dist: 8.8, pitch: 0.26, yaw: 0.55, target: [0.4, 0, 0] },
  circuit: { dist: 10.5, pitch: 0.34, yaw: 0.60, target: [0, -0.1, 0] },
  device: { dist: 11.2, pitch: 0.30, yaw: 0.62, target: [0, 0.55, 0] },
};

const LAYER_META = {
  atlas: ["复合空间", "10⁻¹⁰ → 10⁻³ m"],
  crystal: ["实空间", "10⁻¹⁰ m"],
  bands: ["倒空间", "10⁻⁹ m⁻¹"],
  spin: ["实空间", "10⁻⁹ m"],
  spectra: ["参数空间", "10⁻⁷ m"],
  liquid: ["实空间", "10⁻¹⁰ m"],
  bloch: ["状态空间", "10⁻⁶ m"],
  qubit: ["状态空间", "10⁻³ m"],
  circuit: ["门空间", "10⁻³ m"],
  device: ["实空间", "10⁻³ m"],
};

const SPACE_HUE = [
  ["实空间", "#ffb84d"],
  ["倒空间", "#37d6c0"],
  ["状态空间", "#a78bfa"],
  ["参数空间", "#5b9dff"],
  ["门空间", "#9ae66e"],
  ["复合空间", "#c7d3e6"],
];

/* ── 全局状态 ───────────────────────────────────────────── */
const S = {
  scene: "atlas",
  material: "Si",
  rep: 1, showBZ: true, showBonds: true, showCell: true,
  bandView: "带状", band: 4, kIndex: 45, nBandsShown: 10,
  spinMode: "点阵", spinKind: "FM", J: 1, J2: 0, Danis: 0.12, Bz: 1, Temp: 300, temp: 300, spinB: 0, spinDex: 0, spinT: 0,
  omega: 2.4, gamma: 0.1, quality: "快速 9³", showEps1: true, showAlpha: false,
  liqN: 150, liqSeed: 7, Ds: 1.6, showSqw: true, showGr: true, liqT: 0,
  blochMode: "Bloch 球", B0: 0.4, B1: 0.25, T1: 6, T2: 2.2, bt: 0,
  nq: 2, statePreset: "Bell (Φ⁺)", qview: "Q-sphere", showRho: false,
  nq2: 2, step: 4, nSteps: 6, ops: null, tool: "H", circuitPreset: "Bell",
  noisy: false, qT1: 6, qT2: 2.2, qGateNs: 20,
  devMat: "CZT", devView: "器件剖面", devThick: 1, devBias: 300, devT: 300, devE: 60,
};

/* ── 视口 ───────────────────────────────────────────────── */
const canvas = document.getElementById("gl");
const labelLayer = document.getElementById("labels");
const viewport = document.getElementById("viewport");
const v = new View3D(canvas, labelLayer);

let current = SCENES[S.scene];
let playing = false;
let lastT = 0;

v.onRender = () => {
  const sc = SCENES[S.scene];
  v.frame();
  try {
    sc.build(v, S);
  } catch (err) {
    console.error(err);
    v.label([0, 1.6, 0], "渲染出错：" + err.message, { cls: "tag" });
  }
  v.draw();
};

/* ── DOM 引用 ───────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const nav = $("nav");
const el = {
  chipLayer: $("chip-layer"),
  inspEyebrow: $("insp-eyebrow"),
  inspTitle: $("insp-title"),
  inspSub: $("insp-sub"),
  params: $("params"),
  paramCount: $("param-count"),
  paramsMenu: $("params-menu"),
  readout: $("readout"),
  readoutNote: $("readout-note"),
  trend: $("trend"),
  trendSvg: $("trend-svg"),
  trendCap: $("trend-cap"),
  note: $("note"),
  legend: $("legend"),
  tlLabel: $("tl-label"),
  tlHint: $("tl-hint"),
  tlValue: $("tl-value"),
  tlUnit: $("tl-unit"),
  timeline: $("timeline"),
  btnPlay: $("btn-play"),
  playGlyph: $("play-glyph"),
  btnBack: $("btn-step-back"),
  btnFwd: $("btn-step-fwd"),
  statusTip: $("status-tip"),
  statusLink: $("status-link"),
  statusSpace: $("status-space"),
  spaceName: $("space-name"),
  spaceNote: $("space-note"),
  spaceBadge: $("space-badge"),
  ladderFill: $("ladder-fill"),
  ladderDot: $("ladder-dot"),
  ladderVal: $("ladder-val"),
  busy: $("busy"),
  busyLabel: $("busy-label"),
  toast: $("toast"),
  toastLabel: $("toast-label"),
};

const TIPS = {
  atlas: "点击平台进入该尺度；拖动旋转整座塔。",
  crystal: "点击原子查看物种与坐标；下方多面体是倒空间的布里渊区。",
  bands: "拖时间轴沿高对称路径移动 k；右上切换三种视图。",
  spin: "切换尺度：同一枚箭头在原子 / 点阵 / 宏观下的三种身份。",
  spectra: "拖动光子能量，与能带层的跃迁箭头联动。",
  liquid: "改变种子重建无序构型；打开 S(q,ω) 看自旋扩散。",
  bloch: "拖动时间看态矢量轨迹；三个视角对应 Bloch 的三重身份。",
  qubit: "改变比特数与预设态，观察 Q-sphere 与纠缠。",
  circuit: "左侧放置门，右侧演化塔同步；右键删除。",
};

const TIME_HINTS = {
  bands: "沿 Γ–X–W–K–Γ–L–U… 路径",
  spin: "进动 / 畴壁演化",
  spectra: "光子能量扫描",
  liquid: "自旋扩散",
  bloch: "Bloch 方程积分",
  circuit: "逐门步进",
};

/* ── 工具 ───────────────────────────────────────────────── */
function hueOf(space) {
  const hit = SPACE_HUE.find(([k]) => space.includes(k));
  return hit ? hit[1] : "#c7d3e6";
}

function splitUnit(text) {
  const m = String(text).match(/^([\s\S]*?[\d.]+)\s*(.*)$/);
  if (!m) return [String(text), ""];
  return [m[1], m[2]];
}

function toast(msg) {
  el.toastLabel.textContent = msg;
  el.toast.setAttribute("opened", "");
}

/** 重活前先让浏览器画一帧，避免"卡住没反馈" */
/** 参数变更的统一出口：贵的重算先亮 busy 遮罩，其余直接重绘 */
function afterParamChange(key) {
  TREND.cache.clear();
  const label = current && current.expensive ? current.expensive(key, S) : null;
  if (!label) {
    renderReadout();
    v.request();
    return;
  }
  withBusy(label, () => current.precompute?.(S)).then(() => {
    renderReadout();
    v.request();
  });
}

function withBusy(label, fn) {
  el.busyLabel.textContent = label;
  el.busy.hidden = false;
  return new Promise((resolve) => {
    requestAnimationFrame(() =>
      setTimeout(() => {
        try {
          resolve(fn());
        } finally {
          el.busy.hidden = true;
        }
      }, 24)
    );
});
}

/* ── 场景切换 ───────────────────────────────────────────── */
function setScene(id) {
  if (!SCENES[id]) return;
  const prev = SCENES[S.scene];
  if (prev && prev.unmount) prev.unmount();
  S.scene = id;
  current = SCENES[id];
  const idx = ORDER.indexOf(id);

  nav.value = id;
  document.querySelectorAll("x-navitem").forEach((it) => {
    const on = it.getAttribute("value") === id;
    it.toggleAttribute("selected", on);
    it.toggleAttribute("toggled", on);
});
  document.querySelectorAll("#layer-menu x-menuitem").forEach((mi) => {
    mi.toggleAttribute("toggled", mi.getAttribute("value") === id);
});
  const sc = current;
  const [spaceKind, lenScale] = LAYER_META[id] || ["—", "—"];
  el.chipLayer.textContent = `L${idx} · ${shortName(sc.name)}`;
  el.inspEyebrow.textContent = `LAYER ${idx} · ${spaceKind}`;
  el.inspTitle.textContent = shortName(sc.name);
  el.inspSub.textContent = `${sc.space} · ${sc.spaceNote}`;
  el.spaceName.textContent = sc.space;
  el.spaceNote.textContent = sc.spaceNote;
  el.spaceBadge.firstElementChild.style.background = hueOf(sc.space);
  el.spaceBadge.firstElementChild.style.boxShadow = `0 0 10px ${hueOf(sc.space)}`;
  el.note.innerHTML = sc.note || "";
  el.statusTip.textContent = TIPS[id] || "";
  el.statusSpace.textContent = `${spaceKind} · ${lenScale}`;
  el.ladderVal.textContent = lenScale;
  el.ladderFill.style.width = `${((idx + 1) / ORDER.length) * 100}%`;
  el.ladderDot.style.left = `${((idx + 1) / ORDER.length) * 100}%`;

  const c = (current.camera && current.camera(S)) || CAM[id];
  if (c) {
    v.cam.dist = c.dist;
    v.cam.pitch = c.pitch;
    v.cam.yaw = c.yaw;
    v.cam.target = [...(c.target || [0, 0, 0])];
    v.camHome = { ...v.cam, target: [...v.cam.target] };
  }

  viewport.classList.toggle("split", id === "circuit");
  if (sc.ensure) sc.ensure(S);
  // 若该层保存过参数预设，进入时恢复
  const saved = savedPreset(id);
  if (saved) {
    for (const k of Object.keys(BASE[id] || {})) if (saved[k] !== undefined) S[k] = saved[k];
    TREND.cache.clear();
  }
  if (sc.mount) sc.mount(viewport, S, api);

  renderInspector();
  renderTimeline();
  updateLegend();
  // 先同步重绘一帧：CSS 换主题是异步的，画布不能等它，否则会出现"面板已变、三维没变"的空窗
  try { v.onRender(); } catch { v.request(); }
}

function shortName(name) {
  return name.replace(/^L\d+\s*·\s*/, "");
}

/* ── 检查器：参数 ───────────────────────────────────────── */
function sliderRow(p) {
  const row = document.createElement("div");
  row.className = "prow";

  const top = document.createElement("div");
  top.className = "prow-top";
  const lab = document.createElement("span");
  lab.className = "prow-label";
  lab.textContent = p.label;
  const val = document.createElement("span");
  val.className = "prow-val";

  const fmt = (x) => {
    const n = Number(x);
    const d = p.step >= 1 ? 0 : p.step >= 0.1 ? 1 : 2;
    return `${n.toFixed(d)}${p.unit ? " " + p.unit : ""}`;
  };
  val.textContent = fmt(S[p.key] ?? p.min);
  top.append(lab, val);

  const sl = document.createElement("x-slider");
  sl.min = p.min; sl.max = p.max; sl.step = p.step;
  sl.value = String(S[p.key] ?? p.min);
  sl.addEventListener("change", () => {
    const nv = Number(sl.value);
    S[p.key] = nv;
    val.textContent = fmt(nv);
    afterParamChange(p.key);
});
  row.append(top, sl);
  if (p.hint) {
    const h = document.createElement("div");
    h.className = "prow-hint";
    h.textContent = p.hint;
    row.append(h);
  }
  return row;
}

function switchRow(p) {
  const row = document.createElement("div");
  row.className = "prow-inline";
  const lab = document.createElement("span");
  lab.className = "prow-label";
  lab.textContent = p.label;
  const sw = document.createElement("x-switch");
  sw.size = "small";
  if (S[p.key] !== false) sw.setAttribute("toggled", "");
  sw.addEventListener("toggle", () => {
    S[p.key] = sw.hasAttribute("toggled");
    v.request();
});
  row.append(lab, sw);
  return row;
}

function selectRow(p) {
  const row = document.createElement("div");
  row.className = "prow";
  const lab = document.createElement("div");
  lab.className = "prow-label";
  lab.textContent = p.label;
  const sel = document.createElement("x-select");
  const menu = document.createElement("x-menu");
  p.options.forEach((opt, i) => {
    const it = document.createElement("x-menuitem");
    it.setAttribute("value", String(opt));
    const l = document.createElement("x-label");
    l.textContent = p.labels ? p.labels[i] : String(opt);
    it.append(l);
    if (String(S[p.key]) === String(opt)) it.setAttribute("toggled", "");
    menu.append(it);
});
  sel.append(menu);
  sel.setAttribute("value", String(S[p.key] ?? p.options[0]));
  sel.addEventListener("change", (e) => {
    S[p.key] = e.detail?.newValue ?? sel.value ?? sel.getAttribute("value");
    if (p.key === "circuitPreset") SCENES.circuit.applyPreset(S);
    afterParamChange(p.key);
    if (p.key === "circuitPreset") api.refresh();
});
  row.append(lab, sel);
  return row;
}

function segRow(p) {
  const row = document.createElement("div");
  row.className = "prow";
  const lab = document.createElement("div");
  lab.className = "prow-label";
  lab.textContent = p.label;
  const seg = document.createElement("div");
  seg.className = "seg";
  p.options.forEach((opt) => {
    const b = document.createElement("button");
    b.textContent = String(opt);
    if (String(S[p.key]) === String(opt)) b.classList.add("on");
    b.onclick = () => {
      S[p.key] = opt;
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      afterParamChange(p.key);
    };
    seg.append(b);
});
  row.append(lab, seg);
  if (p.hint) {
    const h = document.createElement("div");
    h.className = "prow-hint";
    h.textContent = p.hint;
    row.append(h);
  }
  return row;
}

function renderInspector() {
  el.params.replaceChildren();
  const list = current.params || [];
  el.paramCount.textContent = list.length ? `${list.length} 项` : "";
  if (!list.length) {
    const p = document.createElement("div");
    p.className = "prow-hint";
    p.textContent = "本层无参数：直接点击三维对象。";
    el.params.append(p);
  }
  for (const p of list) {
    let row;
    if (p.type === "slider") row = sliderRow(p);
    else if (p.type === "switch") row = switchRow(p);
    else if (p.type === "select") row = selectRow(p);
    else if (p.type === "seg") row = segRow(p);
    if (row) el.params.append(row);
  }
  renderReadout();
}

function renderReadout() {
  el.readout.replaceChildren();
  const rows = current.readout ? current.readout(S) : [];
  el.readoutNote.textContent = rows.length ? `${rows.length} 项` : "";
  for (const r of rows) {
    const d = document.createElement("div");
    d.className = "rrow " + (r.cls || "");
    const k = document.createElement("span");
    k.className = "rk";
    const kind = kindOf(S.scene, r.k);
    const dot = document.createElement("i");
    dot.dataset.kind = kind;
    dot.title = KINDS[kind].label;
    k.append(dot, document.createTextNode(r.k));
    const val = document.createElement("span");
    val.className = "rv";
    val.innerHTML = r.v;
    d.append(k, val);
    el.readout.append(d);
  }
  renderTrend(rows);
  renderNotice();
}

/* 图层级能力边界提示 */
function renderNotice() {
  const box = $("prov-notice");
  if (!box) return;
  const n = noticeOf(S.scene);
  if (!n) {
    box.hidden = true;
    box.textContent = "";
    return;
  }
  box.hidden = false;
  box.textContent = n.text;
}

/* ── 读数趋势条 ─────────────────────────────────────────── */
const TREND = { cache: new Map(), off: new Set() };

function numOf(text) {
  const m = String(text).match(/^-?\d+(?:\.\d+)?/);
  return m ? Number(m[0]) : null;
}

function paramSignature() {
  const sc = current;
  const parts = [S.scene, S.material];
  for (const p of sc.params || []) parts.push(`${p.key}=${S[p.key]}`);
  return parts.join("|");
}

function renderTrend(rows) {
  const t = current.time;
  el.trend.hidden = true;
  if (!t || !rows.length) return;

  // 候选：前 4 行"纯数值"读数
  const cand = [];
  rows.forEach((r, i) => {
    const v = numOf(r.v);
    if (v !== null && /^[-−]?\d/.test(String(r.v).trim())) cand.push({ i, v });
  });
  if (!cand.length) return;
  const cands = cand.slice(0, 4);

  const sig = `${paramSignature()}#${cands.map((c) => c.i).join(",")}`;
  if (TREND.off.has(S.scene)) return;

  let data = TREND.cache.get(sig);
  if (!data) {
    const saved = t.get(S);
    const sample = (n) => {
      const out = cands.map(() => []);
      for (let k = 0; k < n; k++) {
        t.set(S, t.min + ((t.max - t.min) * k) / (n - 1), { nPts: current.time.nPts });
        const rr = current.readout ? current.readout(S) : [];
        cands.forEach((c, ci) => {
          const v = rr[c.i] ? numOf(rr[c.i].v) : null;
          out[ci].push(Number.isFinite(v) ? v : null);
        });
      }
      return out;
    };
    const t0 = performance.now();
    let coarse;
    try {
      coarse = sample(6);
    } finally {
      t.set(S, saved, { nPts: current.time.nPts });
    }
    const cost = performance.now() - t0;
    if (cost > 120) { TREND.off.add(S.scene); return; }

    // 选变化最明显的那个量（相对幅度）
    let best = 0, bestScore = -1;
    coarse.forEach((vals, ci) => {
      const clean = vals.filter((v) => v !== null);
      if (clean.length < 3) return;
      const mn = Math.min(...clean), mx = Math.max(...clean);
      const mean = clean.reduce((a, b) => a + b, 0) / clean.length;
      const score = (mx - mn) / (Math.abs(mean) + 1e-9);
      if (score > bestScore) { bestScore = score; best = ci; }
    });
    if (bestScore < 1e-4) {
      // 所有候选都不随扫描参数变化 → 不显示趋势，避免误导
      TREND.cache.set(sig, null);
      return;
    }

    // 只对选中的量做 16 点采样（如果 6 点已够就复用）
    let vals = coarse[best];
    if (coarse[best].length < 16) {
      const saved2 = t.get(S);
      try {
        vals = [];
        for (let k = 0; k < 16; k++) {
          t.set(S, t.min + ((t.max - t.min) * k) / 15, { nPts: current.time.nPts });
          const rr = current.readout ? current.readout(S) : [];
          const v = rr[cands[best].i] ? numOf(rr[cands[best].i].v) : null;
          vals.push(Number.isFinite(v) ? v : null);
        }
      } finally {
        t.set(S, saved2, { nPts: current.time.nPts });
      }
    }
    const clean = vals.filter((v) => v !== null);
    if (clean.length < 3) { TREND.cache.set(sig, null); return; }
    data = {
      idx: cands[best].i,
      vals,
      min: Math.min(...clean),
      max: Math.max(...clean),
      label: rows[cands[best].i].k,
    };
    TREND.cache.set(sig, data);
    if (TREND.cache.size > 60) TREND.cache.clear();
  }

  if (!data) return;
  const { vals, min, max } = data;
  const span = max - min || 1;
  const W = 100, H = 26, pad = 3;
  const pts = [];
  vals.forEach((v, i) => {
    if (v === null) return;
    const x = pad + (i / (vals.length - 1)) * (W - pad * 2);
    const y = H - pad - ((v - min) / span) * (H - pad * 2);
    pts.push([x, y]);
  });
  if (pts.length < 2) return;

  const line = pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const area = `M${pts[0][0].toFixed(2)},${H} L${line.replaceAll(" ", " L")} L${pts[pts.length - 1][0].toFixed(2)},${H} Z`;

  const cur = t.get(S);
  const frac = Math.min(1, Math.max(0, (cur - t.min) / (t.max - t.min || 1)));
  const mx = pad + frac * (W - pad * 2);
  const cv = vals[Math.round(frac * (vals.length - 1))];
  const my = H - pad - (((cv ?? min) - min) / span) * (H - pad * 2);

  el.trendSvg.innerHTML =
    `<path class="area" d="${area}"></path>` +
    `<polyline class="line" points="${line}"></polyline>` +
    `<circle class="marker" cx="${mx.toFixed(2)}" cy="${my.toFixed(2)}" r="1.8"></circle>`;
  el.trendCap.innerHTML =
    `<span>↑ 随 ${t.label}：${data.label}</span>` +
    `<b>${min.toFixed(3)} – ${max.toFixed(3)}</b>`;
  el.trend.hidden = false;
}

/* ── 图例 ───────────────────────────────────────────────── */
function updateLegend() {
  el.legend.replaceChildren();
  const items = legendOf(S.scene);
  for (const it of items) {
    const d = document.createElement("div");
    d.className = "lg";
    const sw = document.createElement("span");
    sw.className = "sw" + (it.dot ? " dot" : "");
    sw.style.background = it.color;
    const t = document.createElement("span");
    t.textContent = it.label;
    d.append(sw, t);
    el.legend.append(d);
  }
}

function legendOf(id) {
  switch (id) {
    case "crystal":
      return [
        { label: "原子（可点击）", color: "#009999", dot: true },
        { label: "化学键", color: "#8fa3bf" },
        { label: "惯用胞 / 布里渊区", color: "#37d6c0" },
        { label: "原胞（红）", color: "#7a162d" },
      ];
    case "bands":
      return [
        { label: "占据带", color: "#5b7fb8" },
        { label: "空带", color: "#8a6bd0" },
        { label: "选中带 / 当前 k", color: "#ffb84d" },
        { label: "跃迁 ħω", color: "#ff7ac8" },
      ];
    case "spectra":
      return [
        { label: "ε(ω) = ε₁ + iε₂", color: "#37d6c0" },
        { label: "ε₂ 投影", color: "#a78bfa" },
        { label: "吸收 α", color: "#ffb84d" },
        { label: "带隙", color: "#9ae66e" },
      ];
    case "spin":
      return [
        { label: "磁矩 ↑", color: "#37d6c0", dot: true },
        { label: "磁矩 ↓", color: "#ff7ac8", dot: true },
        { label: "磁振子色散", color: "#a78bfa" },
      ];
    case "liquid":
      return [
        { label: "自旋旋量（色相=相位）", color: "#ff7ac8", dot: true },
        { label: "g(r)", color: "#37d6c0" },
        { label: "S(q,ω)", color: "#4c9aff" },
      ];
    case "bloch":
      return [
        { label: "态矢量", color: "#ff7ac8" },
        { label: "轨迹", color: "#a78bfa" },
        { label: "球面 / 主轴", color: "#8fa3bf" },
      ];
    case "qubit":
      return [
        { label: "相位 0（同相）", color: "#009999", dot: true },
        { label: "相位 ±π（反相）", color: "#ec6602", dot: true },
      ];
    case "circuit":
      return [
        { label: "|amplitude|", color: "#4c9aff" },
        { label: "相位", color: "#ff7ac8" },
        { label: "当前步", color: "#37d6c0" },
      ];
    default:
      return [
        { label: "实空间", color: "#ffb84d" },
        { label: "倒空间", color: "#37d6c0" },
        { label: "状态空间", color: "#a78bfa" },
      ];
  }
}

/* ── 传送带 ─────────────────────────────────────────────── */
function renderTimeline() {
  const t = current.time;
  if (!t) {
    el.timeline.setAttribute("disabled", "");
    el.timeline.value = "0";
    el.tlLabel.textContent = "静态视图";
    el.tlHint.textContent = "本层没有时间轴";
    el.tlValue.textContent = "—";
    el.tlUnit.textContent = "";
    el.btnPlay.setAttribute("disabled", "");
    el.btnBack.setAttribute("disabled", "");
    el.btnFwd.setAttribute("disabled", "");
    return;
  }
  el.timeline.removeAttribute("disabled");
  el.btnPlay.removeAttribute("disabled");
  el.btnBack.removeAttribute("disabled");
  el.btnFwd.removeAttribute("disabled");
  el.timeline.setAttribute("min", String(t.min));
  el.timeline.setAttribute("max", String(t.max));
  el.timeline.setAttribute("step", String(t.step));
  el.timeline.value = String(t.get(S));
  el.tlLabel.textContent = t.label;
  el.tlHint.textContent = TIME_HINTS[S.scene] || "";
  const [num, unit] = splitUnit(t.fmt ? t.fmt(S) : String(t.get(S)));
  el.tlValue.textContent = num;
  el.tlUnit.textContent = unit;
}

const onTimeline = () => {
  const t = current.time;
  if (!t) return;
  t.set(S, Number(el.timeline.value), { nPts: current.time.nPts });
  syncTimelineReadout();
  renderReadout();
  v.request();
};
el.timeline.addEventListener("input", onTimeline);
el.timeline.addEventListener("change", onTimeline);
function syncTimelineReadout() {
  const t = current.time;
  if (!t) return;
  const [num, unit] = splitUnit(t.fmt ? t.fmt(S) : String(el.timeline.value));
  el.tlValue.textContent = num;
  el.tlUnit.textContent = unit;
}

function nudge(dir) {
  const t = current.time;
  if (!t) return;
  const span = t.max - t.min;
  const step = dir * (S.scene === "circuit" ? span / 6 : Math.max(t.step, span / 120) * (S.scene === "bands" ? 40 : 6));
  let val = t.get(S) + step;
  if (!Number.isFinite(val)) val = t.min;
  if (val > t.max) val = t.max;
  if (val < t.min) val = t.min;
  t.set(S, val, { nPts: current.time.nPts });
  el.timeline.value = String(val);
  syncTimelineReadout();
  v.request();
  renderReadout();
}

el.btnBack.addEventListener("click", () => nudge(-1));
el.btnFwd.addEventListener("click", () => nudge(1));

function tick(now) {
  if (!playing) return;
  const t = current.time;
  if (!t) { setPlaying(false); return; }
  const ts = typeof now === "number" && Number.isFinite(now) ? now : performance.now();
  const dt = Math.min(0.05, Math.max(0, (ts - lastT) / 1000));
  lastT = ts;
  let val = t.get(S) + dt * 0.32;
  if (!Number.isFinite(val)) val = t.min;
  if (val > t.max) val = t.min;
  t.set(S, val, { nPts: current.time.nPts });
  el.timeline.value = String(val);
  syncTimelineReadout();
  renderReadout();
  v.request();
  requestAnimationFrame(tick);
}

function setPlaying(on) {
  playing = on;
  el.btnPlay.toggleAttribute("toggled", on);
  el.playGlyph.textContent = on ? "❚❚" : "▶";
  if (on) { lastT = performance.now(); requestAnimationFrame(tick); }
}
el.btnPlay.addEventListener("click", () => setPlaying(!playing));
window.__tickGuard = (n) => { if (!Number.isFinite(S.kIndex) && S.scene === "bands") S.kIndex = 45; };

/* ── 参数预设 ───────────────────────────────────────────── */
const BASE = {};
for (const id of ORDER) {
  BASE[id] = {};
  for (const p of SCENES[id].params || []) BASE[id][p.key] = S[p.key];
}
const presetKey = (id) => `qa:preset:${id}`;
function savedPreset(id) {
  try { return JSON.parse(localStorage.getItem(presetKey(id)) || "null"); } catch { return null; }
}
function applyPreset(id, useSaved) {
  const base = BASE[id] || {};
  const saved = useSaved ? savedPreset(id) : null;
  for (const k of Object.keys(base)) {
    S[k] = saved && saved[k] !== undefined ? saved[k] : base[k];
  }
  TREND.cache.clear();
  renderInspector();
  renderTimeline();
  v.request();
}

$("params-menu").addEventListener("click", (e) => {
  const mi = e.target.closest("x-menuitem[value]");
  if (!mi) return;
  const action = mi.getAttribute("value");
  if (action === "reset") {
    applyPreset(S.scene, true);
    toast("已恢复本层默认参数");
  } else if (action === "save") {
    const payload = {};
    for (const p of current.params || []) payload[p.key] = S[p.key];
    try {
      localStorage.setItem(presetKey(S.scene), JSON.stringify(payload));
      toast(`已保存「${shortName(current.name)}」的默认参数`);
    } catch {
      toast("保存失败：本地存储不可用");
    }
  } else if (action === "copy") {
    const payload = { scene: S.scene, material: S.material };
    for (const p of current.params || []) payload[p.key] = S[p.key];
    copyText(JSON.stringify(payload, null, 2), "已复制参数 JSON");
  }
});

/* ── 材料库（对应 PyQt 的 Material Repository） ─────────── */
function parseComposition(text) {
  const out = {};
  const t = String(text || "").trim();
  if (!t) return out;
  if (t.includes(":") || t.includes("=")) {
    for (const part of t.split(/[,;]/)) {
      const kv = part.split(/[:=]/);
      if (kv.length >= 2) {
        const el = kv[0].trim();
        const v = Number(kv[1]);
        if (el && Number.isFinite(v)) out[el] = v;
      }
    }
  } else {
    const re = /([A-Z][a-z]?)(\d*\.?\d*)/g;
    let m;
    while ((m = re.exec(t))) {
      if (!m[1]) continue;
      out[m[1]] = m[2] ? Number(m[2]) : 1;
    }
  }
  const sum = Object.values(out).reduce((a, b) => a + b, 0);
  if (sum > 0) for (const k of Object.keys(out)) out[k] /= sum;
  return out;
}

function fillMaterialsTable() {
  const dlg = $("dialog-materials");
  const tb = dlg.querySelector("tbody");
  tb.replaceChildren();
  for (const m of Object.values(DETECTORS)) {
    const tr = document.createElement("tr");
    const cells = [
      m.name + (m.custom ? " ·自定义" : ""),
      m.Eg.toFixed(2),
      (m.mu_e * m.tau_e).toExponential(1),
      (m.mu_h * m.tau_h).toExponential(1),
      m.W.toFixed(2),
      m.rho_m.toFixed(2),
      m.pol.toFixed(2),
    ];
    cells.forEach((c, i) => {
      const td = document.createElement("td");
      td.textContent = c;
      if (i > 0) { td.style.textAlign = "right"; td.style.fontFamily = "var(--font-mono)"; }
      tr.append(td);
    });
    if (m.custom) {
      const td = document.createElement("td");
      const del = document.createElement("button");
      del.textContent = "删除";
      del.className = "mat-del";
      del.onclick = (ev) => {
        ev.stopPropagation();
        removeCustomMaterial(m.id);
        refreshMaterialOptions();
        if (S.devMat === m.id) S.devMat = Object.keys(DETECTORS)[0];
        fillMaterialsTable();
        if (S.scene === "device") renderInspector();
        toast(`已删除 ${m.name}`);
      };
      td.append(del);
      tr.append(td);
    } else {
      tr.append(document.createElement("td"));
    }
    tr.onclick = () => {
      S.devMat = m.id;
      if (S.scene !== "device") setScene("device");
      else { renderInspector(); v.request(); }
      closeDialog(dlg);
      toast(`已切换到 ${m.name}`);
    };
    tb.append(tr);
  }
}

function openMaterials() {
  fillMaterialsTable();
  $("dialog-materials").showModal();
}

function addCustomMaterial() {
  const name = ($("cm-name").value || "").trim();
  const num = (id, dflt) => {
    const v = Number($(id).value);
    return Number.isFinite(v) && v !== 0 ? v : dflt;
  };
  if (!name) { toast("请先填材料名称"); return; }
  const comp = parseComposition($("cm-comp").value);
  if (!Object.keys(comp).length) { toast("成分写法看不懂，试试 Cd0.45Zn0.05Te0.5"); return; }
  const def = {
    id: "custom-" + name,
    name,
    Eg: num("cm-eg", 1.5),
    mu_e: num("cm-mue", 1000),
    mu_h: num("cm-muh", 80),
    tau_e: num("cm-taue", 5e-6),
    tau_h: num("cm-tauh", 5e-7),
    W: num("cm-w", 4.6),
    rho_m: num("cm-rho", 5.8),
    comp,
    pol: 0.1,
    tau_pol: 12,
  };
  saveCustomMaterial(def);
  refreshMaterialOptions();
  fillMaterialsTable();
  S.devMat = def.id;
  if (S.scene !== "device") setScene("device");
  else renderInspector();
  toast(`已添加 ${name}`);
}

/* ── 拾取联动 ───────────────────────────────────────────── */
const api = { refresh: () => { renderInspector(); renderTimeline(); v.request(); } };

v.onPick = (pk) => {
  if (!pk) { el.statusLink.textContent = "联动：无"; return; }
  if (S.scene === "atlas" && SCENES[pk.id]) { setScene(pk.id); return; }
  if (S.scene === "spectra" && pk.id === "eps" && pk.data) {
    S.omega = pk.data.w;
    el.timeline.value = String(S.omega / 8);
    syncTimelineReadout();
    el.statusLink.textContent = "联动：ħω → 能带跃迁";
    renderReadout();
    v.request();
    return;
  }
  if (S.scene === "crystal" && String(pk.id).startsWith("atom:")) {
    el.statusLink.textContent = `${pk.data.species} @ (${pk.data.pos.map((x) => x.toFixed(2)).join(", ")}) Å`;
    return;
  }
  if (S.scene === "bands" && pk.id === "k") {
    el.statusLink.textContent = `k 点 #${pk.data.kIdx}`;
    return;
  }
  el.statusLink.textContent = "联动：" + (pk.id || "—");
};

/* ── 主题 ───────────────────────────────────────────────── */
const THEME = {
  light: "theme/atlas-light.css",
  dark: "theme/atlas-dark.css",
};
let isDark = false;
function applyTheme(dark) {
  isDark = dark;
  document.body.classList.toggle("dark", dark);
  setPalette(dark ? "dark" : "light");
  const url = dark ? THEME.dark : THEME.light;
  const Xel = window.Xel;
  if (Xel) {
    Xel.theme = url;
    Xel.accentColor = dark ? "orange" : "orange";
  } else {
    document.querySelector('meta[name="xel-theme"]').content = url;
  }
  const mi = document.querySelector('#menubar x-menuitem[value="toggle-theme"]');
  if (mi) mi.toggleAttribute("toggled", dark);
  updateLegend();
  // 先同步重绘一帧：CSS 换主题是异步的，画布不能等它，否则会出现"面板已变、三维没变"的空窗
  try { v.onRender(); } catch { v.request(); }
}

/* ── 菜单与按钮 ─────────────────────────────────────────── */
function dispatch(action) {
  switch (action) {
    case "export-png": exportPng(); break;
    case "export-state": copyLink(); break;
    case "quit": if (window.shell) window.shell.command("close"); else toast("浏览器预览模式：请在 Electron 中退出"); break;
    case "cam-iso": v.setPreset("iso"); v.request(); break;
    case "cam-top": v.setPreset("top"); v.request(); break;
    case "cam-front": v.setPreset("front"); v.request(); break;
    case "cam-reset": v.setPreset("reset"); v.request(); break;
    case "toggle-theme": applyTheme(!isDark); break;
    case "toggle-legend":
      document.body.classList.toggle("hide-legend");
      break;
    case "materials": openMaterials(); break;
    case "theory": openCasebook(); break;
    case "export": exportCurrent(); break;
    case "shortcuts": toast("1–9 图层 · Space 播放 · ←/→ 步进 · R 相机 · T 主题 · F1 说明"); break;
    case "about": $("dialog-about").showModal?.(); break;
    default:
      if (SCENES[action]) setScene(action);
  }
}

$("menubar").addEventListener("click", (e) => {
  const mi = e.target.closest("x-menuitem[value]");
  if (!mi) return;
  if (mi.querySelector(":scope > x-menu")) return;   // 顶级菜单
  dispatch(mi.getAttribute("value"));
});
document.querySelectorAll("#layer-menu x-menuitem").forEach((mi) => {
  mi.addEventListener("click", () => setScene(mi.getAttribute("value")));
});
for (const [id, preset] of [["cam-iso", "iso"], ["cam-top", "top"], ["cam-front", "front"], ["cam-reset", "reset"]]) {
  $(id).addEventListener("click", () => { v.setPreset(preset); v.request(); });
}

const drawer = $("drawer");
/** 通用剪贴板：cliboard API 不可用/超时/被拒时自动退化为 execCommand */
function copyText(text, okMsg) {
  let settled = false;
  const done = () => { if (!settled) { settled = true; toast(okMsg); } };
  const fallback = () => {
    if (settled) return;
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.cssText = "position:fixed;opacity:0;left:-9999px";
      document.body.append(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      done();
    } catch {
      settled = true;
      toast(text.slice(0, 90));
    }
  };
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, fallback);
      setTimeout(fallback, 700);   // 有些环境会一直挂起
    } else fallback();
  } catch { fallback(); }
}

function closeDialog(d) {
  if (!d) return;
  try { d.close?.(); } catch { /* ignore */ }
  if (d.open) d.removeAttribute("open");   // 某些环境里 close() 是空操作
}
function drawerIsOpen() {
  return !!(drawer.open || drawer.opened || drawer.hasAttribute("opened"));
}
/* 导出当前图层的数值结果（CSV）：文件菜单 / Ctrl+E / MCP 三条路都走这里 */
function exportCurrent() {
  try {
    const { csv, filename, datasets } = window.__atlas.exportData("csv");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    toast(已导出 );
  } catch (e) {
    toast("导出失败：" + e.message);
  }
}

function openDrawer() {
  if (drawerIsOpen()) {
    drawer.close?.();
    drawer.removeAttribute("opened");
  } else if (drawer.showModal) {
    drawer.showModal();
  } else {
    drawer.setAttribute("opened", "");
  }
}
$("btn-help").addEventListener("click", openCasebook);
$("cb-close").addEventListener("click", closeCasebook);
initCasebook({ $, setScene, setMaterial, S, renderInspector, renderTimeline, v, toast });

/* 全局悬浮面板：任何地方右键都能唤出当前图层速览 */
initHud({
  getState: () => window.__atlas.state(),
  cases: () => CASES,
  applyCase: (c) => {
    closeCasebook();
    setScene(c.apply.scene);
    if (c.apply.material) setMaterial(c.apply.material);
    Object.assign(S, c.apply.params || {});
    renderInspector();
    renderTimeline();
    v.request();
  },
  openCasebook,
});
v.onContext = (x, y) => hudContextAt(x, y);
initMcpDock();
if (new URLSearchParams(location.search).get("page") === "cases") setTimeout(openCasebook, 400);
$("btn-theme").addEventListener("click", () => applyTheme(!isDark));
$("mat-close").addEventListener("click", () => closeDialog($("dialog-materials")));
$("cm-add").addEventListener("click", () => addCustomMaterial());
$("about-close").addEventListener("click", () => {
  const d = $("dialog-about");
  closeDialog(d);
});

/* 左栏图示：每个 glyph 对应论文配图的惯例（见 docs/图层图示对照.md） */
for (const it of document.querySelectorAll("#nav x-navitem")) {
  const id = it.getAttribute("value");
  const svg = `<span class="glyph">${glyphSVG(id, 20)}</span>`;
  const old = it.querySelector("x-icon");
  if (old) old.outerHTML = svg;
  else it.insertAdjacentHTML("afterbegin", svg);
}

/* 下拉菜单定位修正：Xel 把 x-menu 的 inline top/left 算在了带 transform/backdrop-filter
   的祖先里，结果菜单盖在触发按钮上（标题栏那块甚至会被 overflow 裁掉）。
   打开后统一改成 fixed，贴在触发控件下方。 */
document.addEventListener("click", (e) => {
  const sel = e.target && e.target.closest ? e.target.closest("x-select") : null;
  if (!sel) return;
  const menu = sel.querySelector("x-menu");
  if (!menu) return;
  requestAnimationFrame(() => {
    if (!menu.opened) return;
    const r = sel.getBoundingClientRect();
    menu.style.position = "fixed";
    menu.style.left = `${Math.round(r.left)}px`;
    menu.style.top = `${Math.round(r.bottom + 6)}px`;
    menu.style.zIndex = "9999";
    const mr = menu.getBoundingClientRect();
    if (mr.bottom > window.innerHeight - 8) {
      menu.style.top = `${Math.max(8, Math.round(window.innerHeight - 8 - mr.height))}px`;
    }
  });
});
/* ── 左栏导航（原来漏了绑定：点不动） ──────────────────── */
nav.addEventListener("change", () => {
  const id = nav.value;
  if (id && SCENES[id] && id !== S.scene) setScene(id);
});
nav.addEventListener("click", (e) => {
  const item = e.target.closest && e.target.closest("x-navitem");
  if (!item) return;
  const id = item.getAttribute("value");
  if (id && SCENES[id] && id !== S.scene) setScene(id);
});

/* 材料切换（标题栏 x-select） */
const selMat = $("sel-material");

async function setMaterial(id) {
  S.material = id;
  const mat = materialOf(id);
  el.statusLink.textContent = `体系 → ${mat.name}`;
  if (!mat.tb && S.scene !== "atlas") {
    await withBusy(`正在对角化 ${mat.name}…`, () => structure(mat, { nG: 46, mesh: 6 }));
  }
  renderInspector();
  v.request();
}
selMat.addEventListener("change", (e) => setMaterial(e.detail?.newValue || selMat.value || selMat.getAttribute("value")));

/* 窗口按钮 */
$("xelTitlebar").addEventListener("buttonclick", (e) => {
  const cmd = e.detail;
  if (window.shell) window.shell.command(cmd);
  else if (cmd === "close") toast("浏览器预览模式：窗口控制仅在 Electron 生效");
});
/* ── 导出 ───────────────────────────────────────────────── */
function exportPng() {
  try {
    const url = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = `quantum-atlas-${S.scene}-${Date.now()}.png`;
    a.click();
    toast("已导出当前视图 PNG");
  } catch (err) {
    toast("导出失败：" + err.message);
  }
}

function copyLink() {
  const url = new URL(location.href);
  url.search = `?scene=${S.scene}&material=${S.material}`;
  copyText(url.toString(), "已复制状态链接");
}

/* ── 键盘 ───────────────────────────────────────────────── */
window.addEventListener("keydown", (e) => {
  const tgt = e.target;
  if (tgt && tgt.matches && tgt.matches("input, textarea, x-slider, x-select, x-tagsinput, x-numberinput")) return;
  if (e.key >= "1" && e.key <= "9") setScene(ORDER[Number(e.key) - 1]);
  else if (e.code === "Space") { e.preventDefault(); if (current.time) setPlaying(!playing); }
  else if (e.key === "ArrowRight") { e.preventDefault(); nudge(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); nudge(-1); }
  else if (e.key === "r" || e.key === "R") { v.setPreset("reset"); v.request(); }
  else if (e.key === "t" || e.key === "T") applyTheme(!isDark);
  else if (e.key === "F1") { e.preventDefault(); openCasebook(); }
  else if ((e.ctrlKey || e.metaKey) && (e.key === "e" || e.key === "E")) { e.preventDefault(); exportCurrent(); }
  else if (e.key === "Escape" && !$("casebook").hidden) { closeCasebook(); }
  else if (e.key === "Escape") { if (drawer.opened) drawer.close?.(); }
});
/* ── 启动 ───────────────────────────────────────────────── */
const QS = new URLSearchParams(location.search);
applyTheme(QS.get("theme") === "dark");
if (QS.get("quality")) S.quality = QS.get("quality") === "exact" ? "精确 15³" : "快速 9³";
if (QS.get("devView")) S.devView = QS.get("devView");
if (QS.get("qview")) S.qview = QS.get("qview");
if (QS.get("preset")) S.statePreset = QS.get("preset");
if (QS.get("devMat")) S.devMat = QS.get("devMat");
/* 通用：任何已存在的数值参数都能从 URL 覆盖（?temp=500、?J=2 …），方便复现与截图 */
for (const [k, v] of QS.entries()) {
  if (!(k in S)) continue;
  const cur = S[k];
  if (typeof cur === "number" && Number.isFinite(Number(v))) S[k] = Number(v);
  else if (typeof cur === "boolean") S[k] = v === "1" || v === "true";
  else if (typeof cur === "string") S[k] = v;
}
setScene(QS.get("scene") && SCENES[QS.get("scene")] ? QS.get("scene") : "atlas");

if (QS.get("dialog") === "materials") setTimeout(() => openMaterials(), 500);
/* 自检钩子：?selftest=custom 走一遍"添加自定义材料"的完整链路 */
if (QS.get("selftest") === "custom") {
  setTimeout(() => {
    $("cm-name").value = "CdZnTe-2";
    $("cm-comp").value = "Cd0.45Zn0.05Te0.5";
    $("cm-eg").value = "1.60";
    addCustomMaterial();
    document.title = "SELFTEST " + JSON.stringify({
      ids: Object.keys(DETECTORS),
      current: S.devMat,
      stored: JSON.parse(localStorage.getItem("qa:detectors") || "[]").map((d) => d.id),
      opts: SCENES.device.params[0].options,
    });
  }, 700);
}

if (QS.get("material")) {
  const mid = QS.get("material");
  if (materialOf(mid).id === mid) {
    S.material = mid;
    selMat.setAttribute("value", mid);
    selMat.querySelectorAll("x-menuitem").forEach((mi) => {
      mi.toggleAttribute("toggled", mi.getAttribute("value") === mid);
    });
  }
}

window.addEventListener("resize", () => v.request());
setTimeout(() => v.request(), 120);

/* 布局自检（?debug=1）：把关键元素的矩形写进标题，便于无头核查 */
if (QS.get("debug")) {
  setTimeout(() => {
    const r = (sel) => {
      const e = document.querySelector(sel);
      if (!e) return null;
      const b = e.getBoundingClientRect();
      const cs = getComputedStyle(e);
      return [Math.round(b.x), Math.round(b.y), Math.round(b.width), Math.round(b.height), cs.display, cs.visibility, cs.opacity, cs.backgroundColor].join(",");
    };
    document.title = "DBG " + JSON.stringify({
      w: innerWidth,
      brand: r(".brand"),
      menubar: r(".menubar"),
      tail: r(".titlebar-tail"),
      chip: r("#chip-layer"),
      sel: r("#sel-material"),
      cam: r("#cam-iso"),
      vp: r("#viewport"),
      tb: r("#titlebar"),
      rail: r(".rail"),
      insp: r(".inspector"),
      hud: r(".hud"),
      gl: r("#gl"),
      glass: getComputedStyle(document.querySelector(".viewport")).getPropertyValue("--glass-bg").trim(),
    });
  }, 900);
}

/* 无头截图：把 WebGL 后备缓冲转成 <img>（部分 headless 合成器不写 canvas 层） */
if (QS.get("snap")) {
  const shoot = () => {
    try {
      v.onRender();
      const img = document.createElement("img");
      img.src = canvas.toDataURL("image/png");
      img.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:fill;z-index:1;pointer-events:none";
      viewport.append(img);
    } catch { /* ignore */ }
  };
  setTimeout(shoot, 700);
  setTimeout(shoot, 2600);
}

/* 诊断：?probe=theme 量一次主题切换的耗时与画布平均亮度 */
if (QS.get("probe") === "theme") {
  setTimeout(async () => {
    const cv = document.getElementById("gl");
    const mean = () => new Promise((res) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = 32; c.height = 32;
        const g2 = c.getContext("2d");
        g2.drawImage(img, 0, 0, 32, 32);
        const d = g2.getImageData(0, 0, 32, 32).data;
        let s = 0;
        for (let i = 0; i < d.length; i += 4) s += (d[i] + d[i + 1] + d[i + 2]) / 3;
        res(Math.round(s / (d.length / 4)));
      };
      img.src = cv.toDataURL("image/png");
    });
    const out = [];
    v.onRender();
    out.push("dark=" + (await mean()));
    const t0 = performance.now();
    applyTheme(true);
    const tSync = (performance.now() - t0).toFixed(1);
    out.push("syncMs=" + tSync);
    out.push("afterToggleNoRender=" + (await mean()));
    v.onRender();
    out.push("afterRender=" + (await mean()));
    applyTheme(false);
    v.onRender();
    out.push("backLight=" + (await mean()));
    document.title = "PROBE " + out.join(" | ");
  }, 1200);
}

/* 诊断：?probe=menu 检查菜单项 → 对话框的链路 */
if (QS.get("probe") === "menu") {
  setTimeout(() => {
    const out = [];
    try {
      out.push("before=" + $("dialog-about").open);
      dispatch("about");
      out.push("afterDispatch=" + $("dialog-about").open);
      dispatch("materials");
      out.push("mats=" + $("dialog-materials").open);
    } catch (e) { out.push("ERR " + e.name + ":" + e.message); }
    out.push("errs=" + (window.__errs || []).length);
    document.title = "PROBE " + out.join(" | ") + " || " + (window.__errs || []).slice(-2).join(" ; ").slice(0, 160);
  }, 1200);
}
/* 诊断：?probe=about 检查关于对话框的关闭链路 */
if (QS.get("probe") === "about") {
  setTimeout(() => {
    const d = $("dialog-about");
    const out = [];
    try {
      d.showModal();
      out.push("open=" + d.open);
      out.push("closeType=" + typeof d.close);
      const btn = d.querySelector("x-button");
      out.push("btn=" + (btn ? btn.tagName + "#" + btn.id : "none"));
      out.push("shadow=" + !!(btn && btn.shadowRoot));
      out.push("isDialog=" + (d instanceof HTMLDialogElement));
      out.push("hasOpenAttr=" + d.hasAttribute("open"));
      out.push("display=" + getComputedStyle(d).display);
      try {
        d.close();
        out.push("jsClose=" + d.open + "/" + d.hasAttribute("open"));
      } catch (e) { out.push("jsCloseErr=" + e.name + ":" + e.message); }
      d.removeAttribute("open");
      out.push("afterRemoveAttr=" + d.open);
      if (!d.open) { d.showModal(); }
      const inner = btn && btn.shadowRoot && btn.shadowRoot.querySelector("button");
      if (inner) inner.dispatchEvent(new MouseEvent("click", { bubbles: true, composed: true }));
      else if (btn) btn.click();
      out.push("afterClick=" + d.open);
      document.title = "PROBE " + out.join(" | ");
    } catch (e) {
      document.title = "PROBE_ERR " + e.name + ": " + e.message + " | " + out.join(" | ");
    }
  }, 900);
}

/* ── MCP 桥的渲染端 API ─────────────────────────────────────
   只暴露"读状态 + 受控操作"，不暴露任意 eval。
   主进程的本地桥（main.cjs）通过这些方法驱动界面。 */
window.__atlas = {
  state() {
    const t = current.time;
    return {
      scene: S.scene,
      sceneName: current.name,
      space: current.space,
      material: S.material,
      palette: paletteMode(),
      time: t ? { label: t.label, value: t.get(S), min: t.min, max: t.max } : null,
      params: Object.fromEntries((current.params || []).map((p) => [p.key, S[p.key]])),
      readout: (current.readout ? current.readout(S) : []).map((r) => ({
        k: r.k,
        v: String(r.v).replace(/<[^>]+>/g, ""),
      })),
    };
  },
  layers() {
    return ORDER.map((id, i) => ({
      index: i,
      id,
      name: SCENES[id].name,
      space: SCENES[id].space,
      params: (SCENES[id].params || []).map((p) => p.key),
    }));
  },
  async scene(id) {
    if (!SCENES[id]) throw new Error("未知图层：" + id);
    setScene(id);
    await new Promise((r) => setTimeout(r, 120));
    return this.state();
  },
  params(obj) {
    for (const [k, val] of Object.entries(obj || {})) S[k] = val;
    renderInspector();
    renderTimeline();
    v.request();
    return this.state();
  },
  async material(id) {
    await setMaterial(id);
    return this.state();
  },
  time(value) {
    const t = current.time;
    if (!t) throw new Error("当前图层没有扫描轴");
    const val = t.min + (t.max - t.min) * Math.max(0, Math.min(1, Number(value) || 0));
    t.set(S, val, { nPts: t.nPts });
    renderTimeline();
    renderReadout();
    v.request();
    return this.state();
  },
  shot() {
    v.onRender();
    return canvas.toDataURL("image/png");
  },
  openCasebook() { openCasebook(); return { ok: true }; },
  showHud(x = 320, y = 240) { showLayer(x, y); return { ok: true }; },
  exportData(format = "csv") {
    const bundle = collectExport(S.scene, S, current.readout ? current.readout(S) : []);
    if (format === "json") return bundle;
    const csv = toCSV(bundle);
    return { csv, filename: filenameFor(S.scene, "csv"), datasets: bundle.datasets.map((d) => ({ name: d.name, rows: d.rows.length })) };
  },
  cases() { return CASES.map((c) => ({ layer: c.layer, tag: c.tag, title: c.title, apply: c.apply })); },
  /* 通用求解器：GUI 与 MCP 共用同一个数据源 */
  solverKinds() { return solverKinds(); },
  solve(kind, spec, params) {
    // JSON 往返：去掉 data.raw 里缓存函数（如 mat._phase）等不可克隆字段，保证能过 Electron IPC
    return JSON.parse(JSON.stringify(solve({ kind, spec: spec ?? {}, params: params ?? {} })));
  },
  openCase(i) {
    const c = CASES[i];
    if (!c) throw new Error("没有第 " + i + " 个案例");
    openCasebook();
    if (c.apply.scene) setScene(c.apply.scene);
    if (c.apply.material) setMaterial(c.apply.material);
    Object.assign(S, c.apply.params || {});
    renderInspector(); renderTimeline(); v.request();
    return { ok: true, title: c.title };
  },
};

/* 控件自检：?audit=1 逐个模拟操作，结果写进 document.title */
if (QS.get("audit") === "1") {
  window.__auditHooks = { tick: () => tick(performance.now()), nudge, setPlaying };
  import("./audit.js").then(({ runAudit }) => {
    setTimeout(() => {
      runAudit({
        S, SCENES, ORDER, setScene, applyTheme, dispatch, v, el, $,
        palette: C, paletteMode,
        DETECTORS,
        renderInspector,
        baseOf: (id) => BASE[id],
      }).catch((e) => { document.title = "AUDIT_ERR " + String(e.stack || e.message).replace(/\n/g, " ≤ ").slice(0, 300); });
    }, 900);
  });
}
