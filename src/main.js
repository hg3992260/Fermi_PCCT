/* ═══════════════════════════════════════════════════════════════
   main.js — 外壳（v2）
   职责：把场景参数写进 S、请求一帧、把读数渲染成表。
   场景只实现 build(v,S) / readout(S)，外壳不关心任何物理。
   ═══════════════════════════════════════════════════════════════ */

import { View3D } from "./engine.js";
import atlas from "./scenes/atlas.js";
import crystal from "./scenes/crystal.js";
import bands from "./scenes/bands.js";
import spin from "./scenes/spin.js";
import spectra from "./scenes/spectra.js";
import liquid from "./scenes/liquid.js";
import bloch from "./scenes/bloch.js";
import qubit from "./scenes/qubit.js";
import circuit from "./scenes/circuit.js";
import { materialOf } from "./phys/materials.js";
import { structure } from "./phys/epm.js";

/* 开发期错误可见化 */
window.__errs = [];
function noteError(msg) {
  window.__errs.push(msg);
  document.title = "ERR: " + window.__errs.slice(-2).join(" | ");
  const tip = document.getElementById("status-tip");
  if (tip) tip.textContent = "⚠ " + msg;
}
window.addEventListener("error", (e) => noteError(`${e.message} @ ${(e.filename || "").split("/").pop()}:${e.lineno}`));
window.addEventListener("unhandledrejection", (e) => noteError(String(e.reason)));

const SCENES = { atlas, crystal, bands, spin, spectra, liquid, bloch, qubit, circuit };
const ORDER = ["atlas", "crystal", "bands", "spin", "spectra", "liquid", "bloch", "qubit", "circuit"];

const CAM = {
  atlas: { dist: 13.5, pitch: 0.34, yaw: 0.55, target: [0, 0, 0] },
  crystal: { dist: 12.5, pitch: 0.40, yaw: 0.72, target: [0, 0.2, 0] },
  bands: { dist: 13, pitch: 0.34, yaw: 0.55, target: [1.4, 0, 0] },
  spin: { dist: 13.5, pitch: 0.46, yaw: 0.62, target: [0.9, 0.2, 0] },
  spectra: { dist: 12, pitch: 0.30, yaw: 0.62, target: [0, 0, 0.6] },
  liquid: { dist: 13.5, pitch: 0.36, yaw: 0.68, target: [0.4, 0, 0] },
  bloch: { dist: 8.4, pitch: 0.28, yaw: 0.68, target: [0, 0, 0] },
  qubit: { dist: 8.8, pitch: 0.26, yaw: 0.55, target: [0.4, 0, 0] },
  circuit: { dist: 10.5, pitch: 0.34, yaw: 0.60, target: [0, -0.1, 0] },
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
  spinMode: "点阵", spinKind: "FM", J: 1, Danis: 0.12, Bz: 1, Temp: 300, spinT: 0,
  omega: 2.4, gamma: 0.1, showEps1: true, showAlpha: false,
  liqN: 150, liqSeed: 7, Ds: 1.6, showSqw: true, showGr: true, liqT: 0,
  blochMode: "Bloch 球", B0: 0.4, B1: 0.25, T1: 6, T2: 2.2, bt: 0,
  nq: 2, statePreset: "Bell (Φ⁺)", showRho: false,
  nq2: 2, step: 4, nSteps: 6, ops: null, tool: "H", circuitPreset: "Bell",
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
  readout: $("readout"),
  readoutNote: $("readout-note"),
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

  const c = CAM[id];
  if (c) {
    v.cam.dist = c.dist;
    v.cam.pitch = c.pitch;
    v.cam.yaw = c.yaw;
    v.cam.target = [...(c.target || [0, 0, 0])];
    v.camHome = { ...v.cam, target: [...v.cam.target] };
  }

  viewport.classList.toggle("split", id === "circuit");
  if (sc.ensure) sc.ensure(S);
  if (sc.mount) sc.mount(viewport, S, api);

  renderInspector();
  renderTimeline();
  updateLegend();
  v.request();
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
    v.request();
    renderReadout();
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
    S[p.key] = e.detail?.newValue ?? sel.getAttribute("value");
    if (p.key === "circuitPreset") SCENES.circuit.applyPreset(S);
    v.request();
    renderReadout();
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
      v.request();
      renderReadout();
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
    k.textContent = r.k;
    const val = document.createElement("span");
    val.className = "rv";
    val.innerHTML = r.v;
    d.append(k, val);
    el.readout.append(d);
  }
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
        { label: "原子（可点击）", color: "#8fb8ff", dot: true },
        { label: "化学键", color: "#8fa3bf" },
        { label: "布里渊区", color: "#37d6c0" },
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
        { label: "相位 −π", color: "#4c9aff", dot: true },
        { label: "相位 +π", color: "#ff7ac8", dot: true },
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

el.timeline.addEventListener("change", () => {
  const t = current.time;
  if (!t) return;
  t.set(S, Number(el.timeline.value), { nPts: current.time.nPts });
  syncTimelineReadout();
  v.request();
  renderReadout();
});
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
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  let val = t.get(S) + dt * 0.32;
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
  dark: "theme/atlas-dark.css",
  light: "theme/atlas-light.css",
};
let isDark = true;
function applyTheme(dark) {
  isDark = dark;
  document.body.classList.toggle("light", !dark);
  const url = dark ? THEME.dark : THEME.light;
  const Xel = window.Xel;
  if (Xel) {
    Xel.theme = url;
    Xel.accentColor = "blue";
  } else {
    document.querySelector('meta[name="xel-theme"]').content = url;
  }
  const mi = document.querySelector('#menubar x-menuitem[value="toggle-theme"]');
  if (mi) mi.toggleAttribute("toggled", !dark);
  v.request();
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
    case "theory": openDrawer(); break;
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
function openDrawer() {
  if (drawer.opened) drawer.close?.();
  else if (drawer.showModal) drawer.showModal();
  else drawer.setAttribute("opened", "");
}
$("btn-help").addEventListener("click", openDrawer);
$("btn-theme").addEventListener("click", () => applyTheme(!isDark));
$("dialog-about").querySelector("x-button").addEventListener("click", () => $("dialog-about").close?.());

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
selMat.addEventListener("change", (e) => setMaterial(e.detail?.newValue || selMat.getAttribute("value")));

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
  navigator.clipboard?.writeText(url.toString()).then(
    () => toast("已复制状态链接"),
    () => toast(url.toString())
  );
}

/* ── 键盘 ───────────────────────────────────────────────── */
window.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea, x-slider, x-select, x-tagsinput, x-numberinput")) return;
  if (e.key >= "1" && e.key <= "9") setScene(ORDER[Number(e.key) - 1]);
  else if (e.code === "Space") { e.preventDefault(); if (current.time) setPlaying(!playing); }
  else if (e.key === "ArrowRight") { e.preventDefault(); nudge(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); nudge(-1); }
  else if (e.key === "r" || e.key === "R") { v.setPreset("reset"); v.request(); }
  else if (e.key === "t" || e.key === "T") applyTheme(!isDark);
  else if (e.key === "F1") { e.preventDefault(); openDrawer(); }
  else if (e.key === "Escape") { if (drawer.opened) drawer.close?.(); }
});
/* ── 启动 ───────────────────────────────────────────────── */
const QS = new URLSearchParams(location.search);
applyTheme(QS.get("theme") !== "light");
setScene(QS.get("scene") && SCENES[QS.get("scene")] ? QS.get("scene") : "atlas");

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
      cam: r("#cam-iso"),
      glass: getComputedStyle(document.querySelector(".viewport")).getPropertyValue("--glass-bg"),
      sel: r("#sel-material"),
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
