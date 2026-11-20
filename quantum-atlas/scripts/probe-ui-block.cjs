/* ═══════════════════════════════════════════════════════════════
   --probe-ui：用真实鼠标/键盘事件逐个点界面上"只有真机才暴露"的控件
   （拖拽区、overflow 裁切、popover 定位这些问题，DOM click() 测不出来）
   ═══════════════════════════════════════════════════════════════ */
const PROBE_UI = process.argv.includes("--probe-ui") || process.argv.includes("--probe-select");

if (PROBE_UI) {
  app.setPath("userData", path.join(app.getPath("temp"), `qa-probe-${process.pid}`));
  app.whenReady().then(() => {
    createWindow();
    win.webContents.once("did-finish-load", async () => {
      const wait = (ms) => new Promise((r) => setTimeout(r, ms));
      const run = (code) => win.webContents.executeJavaScript(code);
      const click = async (x, y) => {
        win.webContents.sendInputEvent({ type: "mouseDown", x, y, button: "left", clickCount: 1 });
        win.webContents.sendInputEvent({ type: "mouseUp", x, y, button: "left", clickCount: 1 });
        await wait(260);
      };
      // 取某个选择器的中心点
      const center = (sel) => run(`(() => {
        const e = document.querySelector(${JSON.stringify(sel)});
        if (!e) return null;
        const r = e.getBoundingClientRect();
        return { x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2), w: Math.round(r.width), h: Math.round(r.height) };
      })()`);

      const results = [];
      const check = async (name, fn) => {
        try {
          const r = await fn();
          results.push({ name, ok: !!r.ok, detail: r.detail || "" });
        } catch (e) {
          results.push({ name, ok: false, detail: "ERR " + e.message });
        }
      };

      await wait(2200);

      /* 1. 体系下拉：点开、选第二项 */
      await check("select:material", async () => {
        const c = await center("#sel-material");
        await click(c.x, c.y);
        const st = await run(`(() => {
          const s = document.getElementById("sel-material");
          const m = s.querySelector("x-menu");
          const r = m.getBoundingClientRect();
          return { opened: !!m.opened, x: Math.round(r.x), y: Math.round(r.y), h: Math.round(r.height) };
        })()`);
        if (!st.opened) return { ok: false, detail: "菜单没打开" };
        // 菜单必须落在按钮下方，而不是盖住按钮
        if (st.h < 40) return { ok: false, detail: `菜单被裁切 h=${st.h}` };
        const item = await center("#sel-material x-menuitem:nth-child(2)");
        await click(item.x, item.y);
        const val = await run(`document.getElementById("sel-material").value`);
        return { ok: val && val !== "Si", detail: `picked=${val}` };
      });

      /* 2. 主题按钮 */
      await check("btn:theme", async () => {
        const c = await center("#btn-theme");
        const before = await run(`document.body.classList.contains("dark")`);
        await click(c.x, c.y);
        const after = await run(`document.body.classList.contains("dark")`);
        await click(c.x, c.y);
        return { ok: before !== after, detail: `${before}->${after}` };
      });

      /* 3. 帮助按钮 → 抽屉 */
      await check("btn:help", async () => {
        const c = await center("#btn-help");
        await click(c.x, c.y);
        const op = await run(`(() => { const d = document.getElementById("drawer"); return !!(d.open || d.opened || d.hasAttribute("opened")); })()`);
        await run(`(() => { const d = document.getElementById("drawer"); d.close?.(); d.removeAttribute("opened"); })()`);
        return { ok: op };
      });

      /* 4. 菜单栏「文件」下拉 */
      await check("menubar:file", async () => {
        const c = await center("#menubar x-menu:first-of-type");
        await click(c.x, c.y);
        const st = await run(`(() => {
          const m = document.querySelector("#menubar x-menu");
          const r = m.getBoundingClientRect();
          return { opened: !!m.opened, h: Math.round(r.height) };
        })()`);
        win.webContents.sendInputEvent({ type: "keyDown", keyCode: "Escape" });
        await wait(200);
        return { ok: st.opened && st.h > 30, detail: JSON.stringify(st) };
      });

      /* 5. 左栏导航 */
      await check("nav:bands", async () => {
        const c = await center('#nav x-navitem[value="bands"]');
        await click(c.x, c.y);
        const ok = await run(`(() => {
          const it = document.querySelector('#nav x-navitem[value="bands"]');
          const t = document.getElementById("insp-title").textContent;
          return it.hasAttribute("toggled") && t.includes("能带");
        })()`);
        return { ok };
      });

      /* 6. 相机导轨：机位变了，迷你坐标轴应该跟着变 */
      await check("cam:rail", async () => {
        const snap = () => run(`document.getElementById("gizmo-axes").innerHTML`);
        const a = await snap();
        const c = await center("#cam-top");
        await click(c.x, c.y);
        const b = await snap();
        await click((await center("#cam-reset")).x, (await center("#cam-reset")).y);
        return { ok: a !== b };
      });

      /* 7. 传送带：播放 / 单步 / 时间轴 */
      await check("transport:play", async () => {
        const g0 = await run(`document.getElementById("play-glyph").textContent`);
        const c = await center("#btn-play");
        await click(c.x, c.y);
        const g1 = await run(`document.getElementById("play-glyph").textContent`);
        await click(c.x, c.y);
        return { ok: g0 !== g1, detail: `${g0}->${g1}` };
      });

      await check("transport:step", async () => {
        const v0 = await run(`document.getElementById("tl-value").textContent`);
        const c = await center("#btn-step-fwd");
        await click(c.x, c.y);
        const v1 = await run(`document.getElementById("tl-value").textContent`);
        return { ok: v0 !== v1, detail: `${v0}->${v1}` };
      });

      await check("transport:timeline", async () => {
        const t0 = await run(`(() => { const r = document.getElementById("timeline").getBoundingClientRect(); return Math.round(r.x + r.width * 0.18); })()`);
        const y = await run(`(() => { const r = document.getElementById("timeline").getBoundingClientRect(); return Math.round(r.y + r.height / 2); })()`);
        const v0 = await run(`document.getElementById("tl-value").textContent`);
        await click(t0, y);
        const v1 = await run(`document.getElementById("tl-value").textContent`);
        return { ok: v0 !== v1, detail: `${v0}->${v1}` };
      });

      /* 8. 检查器里的分段控件 */
      await check("seg:bandview", async () => {
        const c = await center("#params .seg button:nth-child(2)");
        await click(c.x, c.y);
        const on = await run(`(() => {
          const b = document.querySelectorAll("#params .seg button")[1];
          return b.classList.contains("on");
        })()`);
        return { ok: on };
      });

      /* 9. 标题栏可拖动（mousedown 在空白区不应报错，且窗口位置可读） */
      await check("titlebar:drag-region", async () => {
        const r = await run(`(() => {
          const t = document.querySelector(".brand");
          const s = document.getElementById("sel-material");
          return { brand: getComputedStyle(t).webkitAppRegion, sel: getComputedStyle(s).webkitAppRegion };
        })()`);
        return { ok: r.sel === "no-drag", detail: JSON.stringify(r) };
      });

      const bad = results.filter((r) => !r.ok);
      console.log("PROBE_UI " + JSON.stringify({ total: results.length, fail: bad.length, bad: bad.map((b) => `${b.name}(${b.detail})`) }));
      app.quit();
    });
  });
}
