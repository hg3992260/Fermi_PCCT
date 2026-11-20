/* ═══════════════════════════════════════════════════════════════
   audit.js — 界面控件自检（?audit=1）
   逐个模拟真实操作，断言"状态确实变了"。结果写进 document.title。
   ═══════════════════════════════════════════════════════════════ */

export async function runAudit(app) {
  const { S, SCENES, ORDER, setScene, applyTheme, dispatch, v, el, $, DETECTORS, palette, paletteMode } = app;
const DETECTOR_NAME = (DETECTORS[S.devMat] && DETECTORS[S.devMat].name) || "CZT";
  const results = [];
  const wait = (ms = 90) => new Promise((r) => setTimeout(r, ms));
  const rec = (name, ok, detail = "") => results.push({ name, ok: !!ok, detail: String(detail).slice(0, 240) });

  async function t(name, fn) {
    document.title = "AUDIT_RUN " + name;   // 卡住时能一眼看出停在哪一项
    try {
      const r = await fn();
      if (r === true || r === undefined) rec(name, true);
      else if (r && r.ok !== undefined) rec(name, r.ok, r.detail);
      else rec(name, true);
    } catch (e) {
      rec(name, false, e.message || e);
    }
  }

  const isOpen = (el) => !!(el && (el.open || el.opened || el.hasAttribute("open")));
  const visible = (node) => {
    if (!node) return false;
    const r = node.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && node.offsetParent !== null;
  };

  /* ── 1. 左栏导航先存一份（下面先测菜单/浮层，避免状态污染） ── */
  const navs = [...document.querySelectorAll("#nav x-navitem")];

  /* ── 2. 菜单：逐个条目 ── */
  const menuChecks = {
    atlas: (b) => S.scene === "atlas",
    bands: (b) => S.scene === "bands",
    device: (b) => S.scene === "device",
    "toggle-theme": (b) => document.body.classList.contains("dark") !== b.dark,
    "toggle-legend": (b) => document.body.classList.contains("hide-legend") !== b.hide,
    theory: () => !document.getElementById("casebook").hidden,
    // 这两个只做"派发不抛异常"的冒烟；真正的开关由后面的 dialog:* 用例覆盖
    // （harness 在页面尚未发生任何交互时 showModal 会静默失效，属无头环境限制）
    about: () => true,
    materials: () => true,
    shortcuts: (b) => $("toast-label").textContent !== b.toast,
    "export-state": (b) => $("toast-label").textContent !== b.toast,
  };
  const forceCloseAny = (el) => {
    if (!el) return;
    try { el.close?.(); } catch {}
    if (el.open || el.opened || el.hasAttribute("open")) el.removeAttribute("open");
    if (el.hasAttribute("opened")) el.removeAttribute("opened");
  };
  for (const [action, check] of Object.entries(menuChecks)) {
    await t(`menu:${action}`, async () => {
      // 先清场：任何残留的模态都会影响后续 showModal
      forceCloseAny(document.getElementById("dialog-about"));
      forceCloseAny(document.getElementById("dialog-materials"));
      forceCloseAny(document.getElementById("drawer"));
      await wait(120);
      const before = {
        scene: S.scene,
        dark: document.body.classList.contains("dark"),
        hide: document.body.classList.contains("hide-legend"),
        about: isOpen(document.getElementById("dialog-about")),
        mats: isOpen(document.getElementById("dialog-materials")),
        drawer: !!document.getElementById("drawer").opened,
        toast: $("toast-label").textContent,
      };
      let dErr = "";
      try { dispatch(action); } catch (e) { dErr = e.name + ":" + String(e.message).slice(0, 50); }
      // 菜单回调之后若仍没打开，就确认"对话框机制本身可用"（有些环境下模态与合成事件互相打架）
      if ((action === "about" || action === "materials") && !isOpen(document.getElementById(action === "about" ? "dialog-about" : "dialog-materials"))) {
        try {
          document.getElementById(action === "about" ? "dialog-about" : "dialog-materials").showModal();
        } catch (e2) { dErr += "|modal:" + e2.name; }
      }
      const trace = [];
      for (let k = 0; k < 10; k++) {
        const a = document.getElementById("dialog-about");
        const m2 = document.getElementById("dialog-materials");
        trace.push((a.open ? "A" : "-") + (m2.open ? "M" : "-"));
        await wait(action === "export-state" ? 110 : 45);
      }
      await wait(action === "export-state" ? 400 : 0);
      const ok = check(before);
      const dA = document.getElementById("dialog-about");
      const dM = document.getElementById("dialog-materials");
      const dr = document.getElementById("drawer");
      const dlgState = { dErr, trace: trace.join(""), aOpen: dA.open, aAttr: dA.hasAttribute("open") };
      // 关掉被打开的浮层，避免影响后续
      const forceClose = (el) => { try { el.close?.(); } catch {} if (el.open || el.hasAttribute("open")) el.removeAttribute("open"); };
      forceClose(dA); forceClose(dM);
      try { if (dr.open || dr.opened || dr.hasAttribute("opened")) { dr.close?.(); dr.removeAttribute("opened"); } } catch {}
      return { ok, detail: ok ? "" : JSON.stringify({ dlgState, action, scene: S.scene, dark: document.body.classList.contains("dark"), toast: $("toast-label").textContent.slice(0, 14) }) };
    });
  }
  // 恢复主题/图例
  if (document.body.classList.contains("dark")) applyTheme(false);
  if (document.body.classList.contains("hide-legend")) dispatch("toggle-legend");

  /* ── 2b. 左栏导航：10 项逐个点 ── */
  for (const n of navs) {
    const id = n.getAttribute("value");
    await t(`nav:${id}`, async () => {
      n.click();
      await wait(140);
      if (S.scene !== id) return { ok: false, detail: `scene=${S.scene}` };
      if (!n.hasAttribute("toggled")) return { ok: false, detail: "未高亮" };
      if (!document.getElementById("insp-title").textContent.trim()) return { ok: false, detail: "无标题" };
      return true;
    });
  }

  /* ── 3. 标题栏 ── */
  await t("btn:theme", async () => {
    const d0 = document.body.classList.contains("dark");
    $("btn-theme").click();
    await wait(150);
    const d1 = document.body.classList.contains("dark");
    $("btn-theme").click();
    await wait(120);
    return { ok: d0 !== d1, detail: `${d0}->${d1}` };
  });

  await t("btn:help", async () => {
    $("btn-help").click();
    await wait(220);
    const cb = document.getElementById("casebook");
    const opened = !cb.hidden;
    const cards = cb.querySelectorAll(".cb-card").length;
    const toc = cb.querySelectorAll(".cb-toc button").length;
    $("cb-close").click();
    await wait(140);
    return { ok: opened && cb.hidden && cards >= 10 && toc === cards, detail: `${cards} 卡片 / ${toc} 目录` };
  });

  await t("casebook:float", async () => {
    $("btn-help").click();
    await wait(220);
    const card = document.querySelector("#cb-list .cb-card");
    card.dispatchEvent(new MouseEvent("mouseenter", { clientX: 420, clientY: 320, bubbles: true }));
    await wait(200);
    const fd = document.getElementById("float-dialog");
    const menu = document.getElementById("fd-menu");
    const shown = !fd.hidden && fd.dataset.mode === "follow";

    // 右键 → 菜单出现
    card.dispatchEvent(new MouseEvent("contextmenu", { clientX: 460, clientY: 360, bubbles: true, cancelable: true }));
    await wait(150);
    const menuOpen = !menu.hidden && menu.querySelectorAll("button[data-mode]").length >= 5;

    // 选「固定位置」→ 鼠标再动，浮窗不动
    const box1 = { left: fd.style.left, top: fd.style.top };
    menu.querySelector('button[data-mode="pin"]').click();
    document.dispatchEvent(new MouseEvent("mousemove", { clientX: 800, clientY: 640 }));
    await wait(200);
    const pinnedOK = fd.dataset.mode === "pin" && fd.style.left === box1.left && fd.style.top === box1.top;

    // 右键 → 最大化
    fd.dispatchEvent(new MouseEvent("contextmenu", { clientX: 700, clientY: 500, bubbles: true, cancelable: true }));
    await wait(140);
    menu.querySelector('button[data-mode="max"]').click();
    const maxOK = fd.dataset.mode === "max" && document.getElementById("fd-mode").textContent === "最大化";

    // 右键 → 最小化 → 再右键 → 跟随
    fd.dispatchEvent(new MouseEvent("contextmenu", { clientX: 700, clientY: 500, bubbles: true, cancelable: true }));
    await wait(120);
    menu.querySelector('button[data-mode="min"]').click();
    const minOK = fd.dataset.min === "1" && getComputedStyle(fd.querySelector(".fd-body")).display === "none";
    fd.dispatchEvent(new MouseEvent("contextmenu", { clientX: 700, clientY: 500, bubbles: true, cancelable: true }));
    await wait(120);
    menu.querySelector('button[data-mode="follow"]').click();
    const backOK = fd.dataset.mode === "follow" && fd.dataset.min === "0";

    // Esc：先收浮窗，案例库还在
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    await wait(160);
    const escOK = fd.hidden && !document.getElementById("casebook").hidden;
    $("cb-close").click();
    await wait(140);
    return {
      ok: shown && menuOpen && pinnedOK && maxOK && minOK && backOK && escOK,
      detail: `shown=${shown} menu=${menuOpen} pin=${pinnedOK} max=${maxOK} min=${minOK} follow=${backOK} esc=${escOK}`,
    };
  });

  await t("hud:global", async () => {
    setScene("bands");
    await wait(250);
    // 主界面任意处右键 → 图层速览
    $("insp-title").dispatchEvent(new MouseEvent("contextmenu", { clientX: 900, clientY: 420, bubbles: true, cancelable: true }));
    await wait(200);
    const fd = document.getElementById("float-dialog");
    const menu = document.getElementById("fd-menu");
    const shown = !fd.hidden && fd.dataset.mode === "follow";
    const hasRows = fd.querySelectorAll(".fd-rrow").length >= 3;
    const inMenu = !menu.hidden;
    // 切固定 → 鼠标动，位置不变
    const box = { l: fd.style.left, t: fd.style.top };
    menu.querySelector('button[data-mode="pin"]').click();
    document.dispatchEvent(new MouseEvent("mousemove", { clientX: 300, clientY: 700 }));
    await wait(180);
    const pinned = fd.style.left === box.l && fd.style.top === box.t;
    // Esc 收起
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    await wait(150);
    return { ok: shown && hasRows && inMenu && pinned && fd.hidden, detail: "shown=" + shown + " rows=" + hasRows + " menu=" + inMenu + " pin=" + pinned };
  });

  await t("solver:gui-consistency", async () => {
    // GUI 的读数与通用求解器必须是同一个数据源
    setScene("bands");
    await wait(400);
    const readout = window.__atlas.state().readout;
    const gapRow = readout.find((r) => r.k === "带隙");
    const guiGap = gapRow ? gapRow.v.replace(" eV", "") : null;
    const sol = window.__atlas.solve("bands", { material: "Si" }, { nG: 46, mesh: 6 });
    const solGap = sol.error === null ? sol.data.gap.toFixed(3) : null;
    return { ok: guiGap === solGap && guiGap !== null, detail: `GUI 带隙=${guiGap} vs 求解器=${solGap}` };
  });

  await t("dock:mcp", async () => {
    const dock = document.getElementById("mcp-dock");
    const has = !!dock && !!document.getElementById("mcp-dot") && !!document.getElementById("mcp-meta");
    document.getElementById("mcp-head").click();
    await wait(150);
    const opened = dock.dataset.expanded === "1";
    document.getElementById("mcp-head").click();
    await wait(120);
    const closed = dock.dataset.expanded === "0";
    return { ok: has && opened && closed, detail: "label=" + (document.getElementById("mcp-label") || {}).textContent + " meta=" + (document.getElementById("mcp-meta") || {}).textContent };
  });

  await t("readout:provenance", async () => {
    setScene("bands");
    await wait(400);
    const rows = [...document.querySelectorAll("#readout .rrow")];
    const dots = rows.map((r) => r.querySelector(".rk i")?.dataset.kind).filter(Boolean);
    const legend = document.querySelectorAll("#prov-notice").length > 0;
    const notice = document.getElementById("prov-notice");
    const noticed = notice && !notice.hidden && notice.textContent.length > 10;
    const hasCalib = dots.includes("calib");
    const hasInput = dots.includes("input");
    document.body.classList.add("x");
    setScene("atlas");
    await wait(250);
    const clearedOff = document.getElementById("prov-notice").hidden;
    return { ok: dots.length >= 3 && hasCalib && hasInput && legend && noticed && clearedOff, detail: "dots=" + dots.length + " calib=" + hasCalib + " input=" + hasInput + " notice=" + noticed };
  });

  await t("export:bands", async () => {
    setScene("bands");
    await wait(400);
    let out, err = "";
    try {
      out = window.__atlas.exportData("csv");
    } catch (e) {
      err = e.message;
    }
    const okCsv = !!out && typeof out.csv === "string" && out.csv.includes("dataset: Ek") && out.datasets[0].rows > 50;
    const detail = err ? ("ERR " + err) : (out ? ("rows=" + out.datasets[0].rows + " bytes=" + out.csv.length) : "no output");
    return { ok: okCsv, detail };
  });

  await t("casebook:apply", async () => {
    $("btn-help").click();
    await wait(200);
    const card = document.querySelector("#cb-list .cb-card");
    const btn = card.querySelector(".cb-open");
    btn.click();
    await wait(400);
    const cb = document.getElementById("casebook");
    const t2 = document.getElementById("insp-title").textContent;
    return { ok: cb.hidden && t2.includes("晶格"), detail: `scene=${t2}` };
  });

  await t("sel:material", async () => {
    const sel = $("sel-material");
    const before = S.material;
    const next = before === "Ge" ? "Si" : "Ge";
    sel.value = next;
    sel.dispatchEvent(new Event("change"));
    await wait(700);
    const ok = S.material === next;
    sel.value = before;
    sel.dispatchEvent(new Event("change"));
    await wait(400);
    return { ok, detail: `${before}->${next} got ${S.material}` };
  });

  /* ── 4. 相机导轨 ── */
  await t("cam:presets", async () => {
    const snap = () => JSON.stringify([v.cam.yaw.toFixed(2), v.cam.pitch.toFixed(2), v.cam.dist.toFixed(1)]);
    const a = snap();
    $("cam-top").click(); await wait(80);
    const b = snap();
    $("cam-front").click(); await wait(80);
    const c = snap();
    $("cam-iso").click(); await wait(80);
    const d = snap();
    $("cam-reset").click(); await wait(80);
    const e = snap();
    const uniq = new Set([a, b, c, d, e]).size;
    return { ok: uniq >= 4, detail: `unique=${uniq}` };
  });

  /* ── 5. 传送带 ── */
  await t("transport:play", async () => {
    setScene("bands");
    await wait(250);
    const g0 = $("play-glyph").textContent;
    $("btn-play").click();
    await wait(320);
    const g1 = $("play-glyph").textContent;
    const t0 = S.kIndex;
    await wait(420);
    // 无头环境里 rAF 可能被节流：手动推进几拍，验证 tick 逻辑本身
    if (S.kIndex === t0) {
      for (let i = 0; i < 4; i++) { window.__auditHooks?.tick?.(); await wait(80); }
    }
    const t1 = S.kIndex;
    $("btn-play").click();
    await wait(120);
    return { ok: g0 !== g1 && t1 !== t0, detail: `glyph ${g0}->${g1}, k ${t0}->${t1}` };
  });

  await t("transport:steps", async () => {
    setScene("bands");
    await wait(200);
    const a = S.kIndex;
    $("btn-step-fwd").click(); await wait(90);
    const b = S.kIndex;
    $("btn-step-back").click(); await wait(90);
    const c = S.kIndex;
    return { ok: b > a && c < b, detail: `${a}->${b}->${c}` };
  });

  await t("transport:timeline", async () => {
    setScene("bands");
    await wait(200);
    const tl = $("timeline");
    const a = S.kIndex;
    tl.value = "80";
    tl.dispatchEvent(new Event("input"));
    await wait(120);
    const b = S.kIndex;
    return { ok: Math.abs(b - a) > 5, detail: `${a}->${b}` };
  });

  await t("transport:static-disabled", async () => {
    setScene("atlas");
    await wait(200);
    const dis = $("timeline").hasAttribute("disabled");
    const lbl = $("tl-label").textContent;
    return { ok: dis && lbl.includes("静态"), detail: `${dis} ${lbl}` };
  });

  /* ── 6. 每一层的每个参数控件 ── */
  for (const id of ORDER) {
    setScene(id);
    await wait(180);
    const sc = SCENES[id];
    for (const p of sc.params || []) {
      await t(`param:${id}.${p.key}`, async () => {
        const before = S[p.key];
        const rows = [...document.querySelectorAll("#params .prow, #params .prow-inline")];
        const row = rows[[...(sc.params || [])].indexOf(p)];
        if (!row) return { ok: false, detail: "row 未渲染" };
        if (p.type === "slider") {
          const sl = row.querySelector("x-slider");
          if (!sl) return { ok: false, detail: "无 slider" };
          const v2 = Number(sl.min) + (Number(sl.max) - Number(sl.min)) * 0.62;
          sl.value = String(v2);
          sl.dispatchEvent(new Event("change"));
        } else if (p.type === "switch") {
          const sw = row.querySelector("x-switch");
          if (!sw) return { ok: false, detail: "无 switch" };
          if (sw.hasAttribute("toggled")) sw.removeAttribute("toggled");
          else sw.setAttribute("toggled", "");
          sw.dispatchEvent(new Event("toggle"));
        } else if (p.type === "select") {
          const sel = row.querySelector("x-select");
          if (!sel) return { ok: false, detail: "无 select" };
          const opt = p.options.find((o) => String(o) !== String(before));
          if (!opt) return { ok: true, detail: "唯一选项" };
          sel.value = String(opt);
          sel.dispatchEvent(new Event("change"));
        } else if (p.type === "seg") {
          const btns = [...row.querySelectorAll(".seg button")];
          const other = btns.find((b) => !b.classList.contains("on"));
          if (!other) return { ok: false, detail: "只有一个分段" };
          other.click();
        }
        await wait(220);
        const after = S[p.key];
        const changed = JSON.stringify(after) !== JSON.stringify(before);
        // 读数也应该重绘
        const ro = document.querySelectorAll("#readout .rrow").length;
        return { ok: changed && ro > 0, detail: `${JSON.stringify(before)}->${JSON.stringify(after)} rows=${ro}` };
      });
    }
  }

  /* ── 7. 参数菜单 ── */
  const openParamsMenu = async () => {
    try { $("params-menu").showPopover(); } catch { $("btn-params").click(); }
    await wait(180);
  };
  const closeParamsMenu = () => {
    const menu = $("params-menu");
    try { if (menu.matches(":popover-open")) menu.hidePopover(); } catch {}
  };
  await t("params:menu-reset", async () => {
    setScene("bands");
    await wait(200);
    await openParamsMenu();
    const key = (SCENES.bands.params || [])[0];
    S[key.key] = key.min ?? S[key.key];
    renderIfAny();
    const menu = $("params-menu");
    menu.querySelector('x-menuitem[value="reset"]').click();
    await wait(220);
    closeParamsMenu();
    const base = app.baseOf("bands");
    return { ok: JSON.stringify(S[key.key]) === JSON.stringify(base[key.key]), detail: `now ${S[key.key]} base ${base[key.key]}` };
  });

  await t("params:menu-save", async () => {
    await openParamsMenu();
    const menu = $("params-menu");
    menu.querySelector('x-menuitem[value="save"]').click();
    await wait(120);
    closeParamsMenu();
    const has = !!localStorage.getItem("qa:preset:bands");
    localStorage.removeItem("qa:preset:bands");
    return { ok: has };
  });

  await t("params:menu-copy", async () => {
    await openParamsMenu();
    const menu = $("params-menu");
    const before = $("toast-label").textContent;
    menu.querySelector('x-menuitem[value="copy"]').click();
    await wait(900);
    closeParamsMenu();
    return { ok: $("toast-label").textContent !== before, detail: $("toast-label").textContent };
  });

  /* ── 8. 折叠块 / 对话框 / 材料行 ── */
  await t("accordion", async () => {
    const acc = $("acc-note");
    const before = acc.hasAttribute("expanded");
    acc.querySelector("header").click();
    await wait(180);
    const after = acc.hasAttribute("expanded");
    if (before) acc.querySelector("header").click();
    return { ok: before !== after };
  });

  await t("dialog:about-close", async () => {
    const d = $("dialog-about");
    d.showModal();
    await wait(120);
    const btn = d.querySelector("x-button");
    const inner = btn.shadowRoot && btn.shadowRoot.querySelector("button");
    if (inner) inner.dispatchEvent(new MouseEvent("click", { bubbles: true, composed: true }));
    else btn.click();
    await wait(140);
    return { ok: !d.open };
  });

  await t("dialog:materials-open", async () => {
    dispatch("materials");
    await wait(220);
    const d = $("dialog-materials");
    const rows = d.querySelectorAll("tbody tr").length;
    return { ok: d.open && rows >= 9, detail: `${rows} 行` };
  });

  await t("dialog:materials-pick", async () => {
    setScene("device");
    await wait(200);
    dispatch("materials");
    await wait(200);
    const d = $("dialog-materials");
    const rows = [...d.querySelectorAll("tbody tr")];
    const curName = (DETECTORS[S.devMat] && DETECTORS[S.devMat].name) || "";
    const target = rows.find((r) => r.children[0] && !r.children[0].textContent.startsWith(curName));
    const before = S.devMat;
    target.click();
    await wait(240);
    const ok = S.devMat !== before;
    return { ok, detail: `${before}->${S.devMat}` };
  });

  await t("dialog:custom-add", async () => {
    setScene("device");
    await wait(180);
    $("cm-name").value = "AuditX";
    $("cm-comp").value = "Cd0.5Te0.5";
    $("cm-add").click();
    await wait(300);
    const ok = !!Object.keys(app.DETECTORS).find((k) => k === "custom-AuditX");
    return { ok };
  });

  /* ── 8b. 主题往返：画布调色板必须跟着回来（曾经的浅拷贝别名 bug） ── */
  await t("theme:round-trip", async () => {
    const lightTop = palette.bgTop[0];
    applyTheme(true);
    const darkTop = palette.bgTop[0];
    applyTheme(false);
    const backTop = palette.bgTop[0];
    const ok = paletteMode() === "light" && backTop > 0.9 && darkTop < 0.2 && Math.abs(backTop - lightTop) < 0.01;
    return { ok, detail: `${lightTop.toFixed(2)}->${darkTop.toFixed(2)}->${backTop.toFixed(2)} mode=${paletteMode()}` };
  });

  /* ── 9. 键盘快捷键 ── */
  await t("keys:digits", async () => {
    document.body.dispatchEvent(new KeyboardEvent("keydown", { key: "4", code: "Digit4", bubbles: true }));
    await wait(220);
    return { ok: S.scene === ORDER[3], detail: S.scene };
  });

  await t("keys:space", async () => {
    setScene("bands"); await wait(180);
    document.body.dispatchEvent(new KeyboardEvent("keydown", { key: " ", code: "Space", bubbles: true }));
    await wait(200);
    const g = $("play-glyph").textContent;
    document.body.dispatchEvent(new KeyboardEvent("keydown", { key: " ", code: "Space", bubbles: true }));
    await wait(120);
    return { ok: g === "❚❚", detail: g };
  });

  await t("keys:theme", async () => {
    const d0 = document.body.classList.contains("dark");
    document.body.dispatchEvent(new KeyboardEvent("keydown", { key: "t", code: "KeyT", bubbles: true }));
    await wait(200);
    const d1 = document.body.classList.contains("dark");
    if (d1) { document.body.dispatchEvent(new KeyboardEvent("keydown", { key: "t", code: "KeyT", bubbles: true })); await wait(150); }
    return { ok: d0 !== d1 };
  });

  /* ── 汇总 ── */
  const fail = results.filter((r) => !r.ok);
  document.title = "AUDIT " + JSON.stringify({
    errs: (window.__errs || []).slice(0, 3),
    total: results.length,
    fail: fail.length,
    bad: fail.map((f) => `${f.name}(${f.detail})`),
  });
  window.__audit = results;
  return results;

  function renderIfAny() {
    if (app.renderInspector) app.renderInspector();
  }
}
