/* casebook.js — 案例库子页面的构建与交互
   依赖由 main.js 注入，避免循环 import。 */
import { CASES, BOUNDARY } from "./cases.js";
import { showCase, hideHud } from "./hud.js";

let deps = null;

export function initCasebook(d) {
  deps = d;
  build();
}

function build() {
  const { $ } = deps;
  const toc = $("cb-toc");
  const list = $("cb-list");
  const foot = $("cb-foot");
  if (list.children.length) return;

  toc.insertAdjacentHTML("beforeend", `<div class="cb-toc-title">目录</div>`);

  CASES.forEach((c, i) => {
    const short = c.title.split("：")[0];
    const b = document.createElement("button");
    b.innerHTML = `<b>${c.layer}</b><span>${short}</span>`;
    b.onclick = () => {
      const card = document.getElementById(`cb-card-${i}`);
      card?.scrollIntoView({ block: "start", behavior: "smooth" });
      toc.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    };
    toc.append(b);

    const card = document.createElement("article");
    card.className = "cb-card";
    card.id = `cb-card-${i}`;
    card.innerHTML = `
      <div class="cb-card-h">
        <span class="cb-tag layer">${c.layer}</span>
        <span class="cb-tag">${c.tag}</span>
      </div>
      <h2>${c.title}</h2>
      <p class="cb-q">${c.q}</p>
      <div class="cb-row"><span>怎么做</span><p>${c.do}</p></div>
      <div class="cb-row"><span>看哪里</span><p>${c.see}</p></div>
      <div class="cb-nums">${(c.num || []).map((n) => `<span class="cb-num">${n}</span>`).join("")}</div>
      <button class="cb-open">在界面中打开 →</button>`;
    card.querySelector(".cb-open").onclick = () => applyCase(c);
    card.dataset.case = String(i);
    // 悬停卡片 → 浮窗预览（右键卡片可切换跟随/固定/最大化）
    card.addEventListener("mouseenter", (e) => showCase(i, e.clientX, e.clientY));
    card.addEventListener("focusin", () => showCase(i, 240, 240));
    list.append(card);
  });

  foot.innerHTML = `
    <h3>能力边界（诚实版）</h3>
    <table>${BOUNDARY.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>`;
}

async function applyCase(c) {
  const { setScene, setMaterial, S, renderInspector, renderTimeline, v, toast } = deps;
  closeCasebook();
  if (c.apply.scene) setScene(c.apply.scene);
  if (c.apply.material) await setMaterial(c.apply.material);
  if (c.apply.params) {
    for (const [k, val] of Object.entries(c.apply.params)) S[k] = val;
    renderInspector();
    renderTimeline();
    v.request();
  }
  toast(`已切到能复现该案例的状态：${c.title.split("：")[0]}`);
}

export function openCasebook() {
  const el = document.getElementById("casebook");
  if (!el || !el.hidden) return;
  el.hidden = false;
  document.body.classList.add("casebook-open");
  el.querySelector(".cb-list").scrollTop = 0;
}

export function closeCasebook() {
  const el = document.getElementById("casebook");
  if (!el || el.hidden) return;
  el.hidden = true;
  hideHud();
  document.body.classList.remove("casebook-open");
}
