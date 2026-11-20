/* hud.js — 全局悬浮面板（鼠标跟随 / 固定 / 最大化）
 *
 * 两种内容：
 *   · showLayer()  —— 任意位置右键：当前图层的实时读数（跟随参数变化自动刷新）
 *   · showCase(i)  —— 案例卡上右键/悬停：该案例的完整说明
 *
 * 模式：follow（跟随光标）/ pin（固定，可拖动）/ max（居中放大），
 * 外加 minimized。全部由右键菜单切换，Esc 逐层退出。
 */


let deps = null;
let fd = null;
let menu = null;
let mode = "follow";
let minimized = false;
let content = "layer";       // layer | case
let caseIndex = -1;
let lastX = 0;
let lastY = 0;
let timer = 0;

export function initHud(d) {
  deps = d;
  fd = document.getElementById("float-dialog");
  menu = document.getElementById("fd-menu");
  if (!fd || !menu) return;

  document.addEventListener("contextmenu", onContext, true);
  menu.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]");
    if (b) applyMode(b.dataset.mode);
  });
  document.addEventListener("pointerdown", (e) => {
    if (menu.hidden) return;
    if (e.target.closest("#fd-menu") || e.button === 2) return;
    closeMenu();
  }, true);

  const head = document.getElementById("fd-head");
  let drag = null;
  head.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    drag = { dx: e.clientX - fd.offsetLeft, dy: e.clientY - fd.offsetTop };
    head.setPointerCapture(e.pointerId);
    if (mode !== "pin") applyMode("pin");
  });
  head.addEventListener("pointermove", (e) => {
    if (!drag) return;
    fd.style.left = Math.max(8, e.clientX - drag.dx) + "px";
    fd.style.top = Math.max(8, e.clientY - drag.dy) + "px";
  });
  head.addEventListener("pointerup", () => (drag = null));
  head.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); openMenu(e.clientX, e.clientY); });

  let raf = 0;
  document.addEventListener("mousemove", (e) => {
    lastX = e.clientX;
    lastY = e.clientY;
    if (!fd || fd.hidden || mode !== "follow" || minimized) return;
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; fdPlace(lastX, lastY); });
  });

  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (menu && !menu.hidden) { closeMenu(); e.stopPropagation(); return; }
    if (fd && !fd.hidden) { hideHud(); e.stopPropagation(); }
  }, true);

  // 图层模式下自动刷新读数
  timer = setInterval(() => {
    if (!fd.hidden && content === "layer" && !minimized) renderLayer(true);
  }, 700);
}

/* 画布里的右键由引擎转过来（保住了右键拖拽平移） */
export function hudContextAt(x, y) {
  showLayer(x, y);
  openMenu(x, y);
}

function onContext(e) {
  const card = e.target.closest ? e.target.closest("#cb-list .cb-card") : null;
  const inFloat = e.target.closest ? e.target.closest("#float-dialog") : null;
  if (inFloat) {
    e.preventDefault();
    openMenu(e.clientX, e.clientY);
    return;
  }
  if (card) {
    e.preventDefault();
    showCase(Number(card.dataset.case || 0), e.clientX, e.clientY);
    openMenu(e.clientX, e.clientY);
    return;
  }
  // 其它地方：只要不在输入控件里，就唤出"当前图层速览"
  if (e.target.closest && e.target.closest("input, textarea, x-input, x-select, x-slider")) return;
  e.preventDefault();
  showLayer(e.clientX, e.clientY);
  openMenu(e.clientX, e.clientY);
}

/* ── 内容 ─────────────────────────────────────────── */
export function showLayer(x = lastX, y = lastY) {
  if (!fd) return;
  content = "layer";
  caseIndex = -1;
  renderLayer();
  fd.hidden = false;
  if (mode === "follow" && !minimized) fdPlace(x, y);
}

function renderLayer(quiet = false) {
  const st = deps.getState();
  if (!st) return;
  const sig = JSON.stringify(st);
  if (quiet && fd.dataset.sig === sig) return;
  fd.dataset.sig = sig;

  document.getElementById("fd-title").textContent = `${st.sceneName} · ${st.space}`;
  const rows = (st.readout || []).slice(0, 7)
    .map((r) => `<div class="fd-rrow"><span>${r.k}</span><b>${r.v}</b></div>`).join("");
  const params = Object.entries(st.params || {}).slice(0, 6)
    .map(([k, v]) => `<span class="cb-num">${k} = ${typeof v === "number" ? (Math.abs(v) < 1 ? v.toFixed(2) : v) : v}</span>`)
    .join("");
  const t = st.time ? `<div class="fd-time">${st.time.label} · ${Number(st.time.value).toFixed(2)} / ${st.time.max}</div>` : "";

  document.getElementById("fd-body").innerHTML = `
    <h4>${st.sceneName}</h4>
    ${t}
    <div class="fd-readout">${rows}</div>
    <div class="cb-nums">${params}</div>
    <button class="cb-open" data-act="casebook">打开案例库 →</button>`;
  const btn = document.getElementById("fd-body").querySelector('[data-act="casebook"]');
  if (btn) btn.onclick = () => deps.openCasebook();
}

export function showCase(i, x = lastX, y = lastY) {
  if (!fd) return;
  content = "case";
  caseIndex = i;
  const c = deps.cases()[i];
  if (!c) return;
  document.getElementById("fd-title").textContent = `${c.layer} · ${c.tag} · ${c.title.split("：")[0]}`;
  document.getElementById("fd-body").innerHTML = `
    <h4>${c.title}</h4>
    <p class="fd-q">${c.q}</p>
    <div class="fd-row"><span>怎么做</span><p>${c.do}</p></div>
    <div class="fd-row"><span>看哪里</span><p>${c.see}</p></div>
    <div class="cb-nums">${(c.num || []).map((n) => `<span class="cb-num">${n}</span>`).join("")}</div>
    <button class="cb-open" data-act="apply">在界面中打开 →</button>`;
  document.getElementById("fd-body").querySelector('[data-act="apply"]').onclick = () => deps.applyCase(c);
  fd.hidden = false;
  if (mode === "follow" && !minimized) fdPlace(x, y);
}

/* ── 模式与几何 ───────────────────────────────────── */
function openMenu(x, y) {
  menu.hidden = false;
  menu.querySelectorAll("button[data-mode]").forEach((b) => {
    const m = b.dataset.mode;
    b.classList.toggle("on",
      (m === "follow" && mode === "follow" && !minimized) ||
      (m === "pin" && mode === "pin") ||
      (m === "max" && mode === "max") ||
      (m === "min" && minimized));
  });
  const w = menu.offsetWidth || 170;
  const h = menu.offsetHeight || 200;
  menu.style.left = Math.min(x, window.innerWidth - w - 10) + "px";
  menu.style.top = Math.min(y, window.innerHeight - h - 10) + "px";
}

function closeMenu() {
  if (menu) menu.hidden = true;
}

function applyMode(m) {
  if (m === "hide") { hideHud(); return; }
  if (m === "min") {
    minimized = !minimized;
    fd.dataset.min = minimized ? "1" : "0";
    closeMenu();
    return;
  }
  mode = m;
  minimized = false;
  fd.dataset.min = "0";
  fd.dataset.mode = m;
  const label = document.getElementById("fd-mode");
  if (label) label.textContent = m === "pin" ? "固定" : m === "max" ? "最大化" : "跟随";
  if (m === "follow") fdPlace(lastX, lastY);
  closeMenu();
}

function fdPlace(x, y) {
  if (!fd || fd.hidden || mode === "max") return;
  const w = fd.offsetWidth || 340;
  const h = fd.offsetHeight || 240;
  let left = x + 18;
  let top = y + 18;
  if (left + w > window.innerWidth - 12) left = x - w - 18;
  if (top + h > window.innerHeight - 12) top = y - h - 18;
  fd.style.left = Math.max(10, left) + "px";
  fd.style.top = Math.max(10, top) + "px";
}

export function hideHud() {
  if (fd) { fd.hidden = true; fd.dataset.sig = ""; }
  closeMenu();
}
