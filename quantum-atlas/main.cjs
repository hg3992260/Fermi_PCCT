// Electron 主进程（CommonJS）：ESM 在这里会解析到 npm 包，必须用 require
// 走自定义协议 app:// 提供页面资源，否则 Xel 的图标/主题 fetch 在 file:// 下会失败
const { app, BrowserWindow, protocol, net, ipcMain, nativeTheme } = require("electron");
const path = require("node:path");
const fs = require("node:fs");
const { pathToFileURL } = require("node:url");

const ROOT = __dirname;
const SMOKE = process.argv.includes("--smoke");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".woff2": "font/woff2",
  ".ftl": "text/plain; charset=utf-8",
};

// Windows 任务栏分组 / 图标：必须在窗口创建前设置
app.setAppUserModelId?.("com.quantumatlas.app");

// 自检时用临时 profile，避免与正在运行的实例争抢 userData 缓存锁
if (SMOKE) {
  try {
    app.setPath("userData", path.join(app.getPath("temp"), `qa-smoke-${process.pid}`));
  } catch { /* ignore */ }
  console.log("SMOKE_START");
  setTimeout(() => {
    console.log("SMOKE_TIMEOUT");
    app.quit();
  }, 25000).unref?.();
}

protocol.registerSchemesAsPrivileged([
  { scheme: "app", privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true } },
]);

/** @type {BrowserWindow | null} */
let win = null;

function createWindow() {
  win = new BrowserWindow({
    width: 1480,
    height: 940,
    minWidth: 1180,
    minHeight: 720,
    frame: false,
    titleBarStyle: "hidden",
    backgroundColor: "#ffffff",
    icon: path.join(ROOT, "assets", "icon.ico"),
    show: false,
    webPreferences: {
      preload: path.join(ROOT, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  win.loadURL("app://local/index.html");
  win.once("ready-to-show", () => {
    if (!SMOKE) win.show();
  });

  const push = () => {
    if (!win || win.isDestroyed()) return;
    win.webContents.send("shell:state", {
      maximized: win.isMaximized(),
      dark: nativeTheme.shouldUseDarkColors,
    });
  };
  win.on("maximize", push);
  win.on("unmaximize", push);
  win.webContents.on("did-finish-load", push);

  win.webContents.on("before-input-event", (event, input) => {
    if (input.type === "keyDown" && input.key === "F12") win.webContents.toggleDevTools();
  });

  // 无头自检：不显示窗口，加载完成后回读运行时状态并抓帧，然后退出
  if (SMOKE) {
    win.webContents.once("did-finish-load", async () => {
      try {
        await new Promise((r) => setTimeout(r, 3500));
        const info = await win.webContents.executeJavaScript(
          `JSON.stringify({
             title: document.title,
             theme: (function(){ try { return getComputedStyle(document.documentElement).getPropertyValue("--theme-id").trim(); } catch(e) { return "?"; } })(),
             errs: (window.__errs || []).slice(0, 3),
             nav: document.querySelectorAll("x-navitem").length,
             icons: document.querySelectorAll('link[rel="icon"]').length,
             brandImg: !!document.querySelector(".brand-mark img") && document.querySelector(".brand-mark img").complete,
             labels: document.querySelectorAll("#labels .lab").length,
             canvas: (() => { const c = document.getElementById("gl"); return c ? c.width + "x" + c.height : "none"; })()
           })`
        );
        console.log("SMOKE_INFO " + info);
        win.webContents.setBackgroundThrottling(false);
        win.webContents.invalidate();
        await new Promise((r) => setTimeout(r, 400));
        const img = await win.webContents.capturePage();
        fs.writeFileSync(path.join(ROOT, "docs", "shot-electron.png"), img.toPNG());
        console.log("SMOKE_OK");
      } catch (e) {
        console.log("SMOKE_FAIL " + e.message);
      } finally {
        app.quit();
      }
    });
  }
}

app.whenReady().then(() => {
  nativeTheme.themeSource = "dark";

  protocol.handle("app", (request) => {
    const url = new URL(request.url);
    if (url.host !== "local") return new Response("not found", { status: 404 });
    if (url.pathname === "/favicon.ico") {
      try {
        const ico = fs.readFileSync(path.join(ROOT, "assets", "icon.ico"));
        return new Response(ico, { status: 200, headers: { "content-type": "image/x-icon", "cache-control": "no-store" } });
      } catch { return new Response(null, { status: 204 }); }
    }
    const rel = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    const filePath = path.normalize(path.join(ROOT, rel));
    if (!filePath.startsWith(ROOT)) return new Response("forbidden", { status: 403 });

    // 自己读盘并显式给 no-store：否则 Chromium 会缓存 app:// 响应，
    // 改完 CSS/JS 重启仍看到旧界面（这个坑很隐蔽）。
    try {
      const buf = fs.readFileSync(filePath);
      const type = MIME[path.extname(filePath).toLowerCase()] || "application/octet-stream";
      return new Response(buf, {
        status: 200,
        headers: { "content-type": type, "cache-control": "no-store" },
      });
    } catch {
      if (SMOKE) console.log("SMOKE_MISS " + url.pathname);
      return new Response("not found: " + rel, { status: 404, headers: { "cache-control": "no-store" } });
    }
  });

  createWindow();
  startBridge();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("shell:command", (_e, cmd) => {
  if (!win || win.isDestroyed()) return;
  if (cmd === "minimize") win.minimize();
  else if (cmd === "maximize") (win.isMaximized() ? win.unmaximize() : win.maximize());
  else if (cmd === "close") win.close();
});

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
      // 真实鼠标事件探针（--probe-ui）
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
        const st = await run(`(() => {
          const cb = document.getElementById("casebook");
          return { open: !cb.hidden, cards: cb.querySelectorAll(".cb-card").length };
        })()`);
        const close = await center("#cb-close");
        if (st.open && close) await click(close.x, close.y);
        const after = await run(`document.getElementById("casebook").hidden`);
        return { ok: st.open && st.cards >= 10 && after, detail: `${st.cards} 卡片` };
      });

      /* 4. 菜单栏「文件」下拉 */
      await check("menubar:file", async () => {
        const c = await center("#menubar x-menuitem");
        await click(c.x, c.y);
        const st = await run(`(() => {
          const ms = [...document.querySelectorAll("#menubar x-menu")];
          const m = ms.find((x) => x.opened) || ms[0];
          const r = m ? m.getBoundingClientRect() : { x: 0, y: 0, width: 0, height: 0 };
          return { opened: ms.some((x) => x.opened), h: Math.round(r.height) };
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
      /* 8b. 案例库里的鼠标跟随浮窗：真鼠标移动过去，浮窗应出现并跟随 */
      await check("float:follow", async () => {
        const help = await center("#btn-help");
        await click(help.x, help.y);
        await wait(320);
        const card = await center("#cb-list .cb-card");
        win.webContents.sendInputEvent({ type: "mouseMove", x: card.x, y: card.y });
        await wait(320);
        const s1 = await run(`(() => {
          const fd = document.getElementById("float-dialog");
          const r = fd.getBoundingClientRect();
          return { open: !fd.hidden, x: Math.round(r.x), y: Math.round(r.y), title: document.getElementById("fd-title").textContent };
        })()`);
        // 真右键点浮窗 → 模式菜单应弹出
        const fx = s1.x + 60, fy = s1.y + 12;
        win.webContents.sendInputEvent({ type: "mouseDown", x: fx, y: fy, button: "right", clickCount: 1 });
        win.webContents.sendInputEvent({ type: "mouseUp", x: fx, y: fy, button: "right", clickCount: 1 });
        await wait(320);
        const menuOpen = await run(`!document.getElementById("fd-menu").hidden`);
        win.webContents.sendInputEvent({ type: "keyDown", keyCode: "Escape" });
        win.webContents.sendInputEvent({ type: "keyUp", keyCode: "Escape" });
        await wait(200);
        win.webContents.sendInputEvent({ type: "mouseMove", x: card.x + 70, y: card.y + 46 });
        await wait(280);
        const s2 = await run(`(() => {
          const r = document.getElementById("float-dialog").getBoundingClientRect();
          return { x: Math.round(r.x), y: Math.round(r.y) };
        })()`);
        const shot = await win.webContents.capturePage();
        fs.writeFileSync(path.join(ROOT, "docs", "v17-float.png"), shot.toPNG());
        await run(`document.getElementById("cb-close").click()`);
        await wait(200);
        const follow = s2.x !== s1.x || s2.y !== s1.y;
        return { ok: s1.open && follow && menuOpen, detail: `跟随=${follow} 右键菜单=${menuOpen} (${s1.x},${s1.y})->(${s2.x},${s2.y})` };
      });

      /* 8c. 全局悬浮面板：在三维画布里右键（轻点，不是拖拽平移） */
      await check("hud:canvas", async () => {
        const vp = await center("#gl");
        win.webContents.sendInputEvent({ type: "mouseDown", x: vp.x, y: vp.y, button: "right", clickCount: 1 });
        win.webContents.sendInputEvent({ type: "mouseUp", x: vp.x, y: vp.y, button: "right", clickCount: 1 });
        await wait(420);
        const st = await run(`(() => {
          const fd = document.getElementById("float-dialog");
          return { open: !fd.hidden, rows: fd.querySelectorAll(".fd-rrow").length, title: document.getElementById("fd-title").textContent };
        })()`);
        const shot = await win.webContents.capturePage();
        fs.writeFileSync(path.join(ROOT, "docs", "v18-hud.png"), shot.toPNG());
        win.webContents.sendInputEvent({ type: "keyDown", keyCode: "Escape" });
        win.webContents.sendInputEvent({ type: "keyUp", keyCode: "Escape" });
        await wait(250);
        return { ok: st.open && st.rows >= 3, detail: `${st.title} rows=${st.rows}` };
      });

      /* 8e. 右下角 MCP 状态坞：桥被调用后要出现日志行 */
      /* 8e. 右下角 MCP 状态坞：桥被调用后出现日志行，点开看请求/响应体 */
      await check("mcp:dock", async () => {
        await run('document.getElementById("mcp-head").click()');
        await wait(300);
        await run('fetch("http://127.0.0.1:8701/layers").then(r => r.json()).then(() => true)');
        await wait(1200);
        await run('fetch("http://127.0.0.1:8701/params", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ params: { devBias: 650 } }) }).then(r => r.json()).then(() => true)');
        await wait(2200);

        const st = await run(`(() => {
          const dock = document.getElementById("mcp-dock");
          return {
            state: dock.dataset.state,
            label: document.getElementById("mcp-label").textContent,
            meta: document.getElementById("mcp-meta").textContent,
            rows: document.querySelectorAll("#mcp-log .mcp-row").length,
          };
        })()`);

        const detail = await run(`(() => {
          const rows = [...document.querySelectorAll("#mcp-log .mcp-row")];
          const row = rows.reverse().find((r) => r.textContent.includes("/params"));
          if (!row) return { found: false, isDetail: false, text: "" };
          row.click();
          const d = row.nextElementSibling;
          return { found: true, isDetail: !!d && d.classList.contains("mcp-detail"), hasClass: row.className, text: (d ? d.textContent : row.textContent).slice(0, 200) };
        })()`);

        await wait(400);
        const shot = await win.webContents.capturePage();
        fs.writeFileSync(path.join(ROOT, "docs", "v20-mcp-dock.png"), shot.toPNG());
        await run('document.getElementById("mcp-head").click()');
        await wait(200);
        const okDetail = detail.found && detail.isDetail && /devBias/.test(detail.text);
        return { ok: st.state === "on" && st.rows >= 1 && okDetail, detail: st.label + " | rows=" + st.rows + " | " + JSON.stringify(detail).slice(0, 220) };
      });

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

/* ═══════════════════════════════════════════════════════════════
   MCP 本地桥：127.0.0.1:8701（可用 ATLAS_MCP_PORT 改）
   MCP 服务器（mcp/atlas-mcp.mjs）是瘦客户端，所有状态都从这里取，
   因此界面永远是唯一真源。只监听回环地址，不对外暴露。
   ═══════════════════════════════════════════════════════════════ */
const bridgeLog = [];
let bridgeSeq = 0;
let lastClientAt = 0;

function startBridge() {
  const http = require("node:http");
  const port = Number(process.env.ATLAS_MCP_PORT || 8701);

  const run = (code) => {
    if (!win || win.isDestroyed()) throw new Error("窗口未就绪");
    return win.webContents.executeJavaScript(code);
  };

  const json = (res, obj, code = 200) => {
    const body = JSON.stringify(obj);
    res.__body = body.length > 4000 ? body.slice(0, 4000) + "…" : body;
    res.writeHead(code, {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      // 渲染进程的 origin 是 app://local，本地桥需要允许跨源
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "content-type",
      "access-control-allow-methods": "GET, POST, OPTIONS",
    });
    res.end(body);
  };

  const readBody = (req) =>
    new Promise((resolve, reject) => {
      let buf = "";
      req.on("data", (d) => {
        buf += d;
        if (buf.length > 1e6) reject(new Error("body too large"));
      });
      req.on("end", () => {
        try {
          const parsed = buf ? JSON.parse(buf) : {};
          req.__body = buf.length > 2000 ? buf.slice(0, 2000) + "…" : buf;
          resolve(parsed);
        } catch (e) { reject(e); }
      });
      req.on("error", reject);
    });

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://127.0.0.1");
    const t0 = Date.now();
    const internal = url.pathname === "/mcp/log";
    res.on("finish", () => {
      bridgeLog.push({
        id: ++bridgeSeq,
        t: t0,
        method: req.method,
        path: url.pathname,
        status: res.statusCode,
        ms: Date.now() - t0,
        internal,
        reqBody: req.__body || null,
        resBody: res.__body || null,
      });
      if (bridgeLog.length > 200) bridgeLog.shift();
      if (!internal) lastClientAt = Date.now();
    });
    if (req.method === "OPTIONS") {
      res.writeHead(204, {
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "content-type",
        "access-control-allow-methods": "GET, POST, OPTIONS",
      });
      return res.end();
    }
    try {
      if (url.pathname === "/mcp/log") {
        const since = Number(url.searchParams.get("since") || 0);
        const tools = (fs.readFileSync(path.join(ROOT, "mcp", "atlas-mcp.mjs"), "utf8").match(/name: "atlas_/g) || []).length;
        return json(res, {
          ok: true,
          port,
          tools,
          seq: bridgeSeq,
          clientActive: Date.now() - lastClientAt < 20000,
          lastClientAt,
          events: bridgeLog.filter((e) => e.id > since),
        });
      }
      if (url.pathname === "/health") return json(res, { ok: true, scene: await run("window.__atlas?.state().scene") });
      if (url.pathname === "/state") return json(res, await run("window.__atlas.state()"));
      if (url.pathname === "/layers") return json(res, await run("window.__atlas.layers()"));
      if (url.pathname === "/cases") return json(res, await run("window.__atlas.cases()"));
      if (url.pathname === "/solve" && req.method === "POST") {
        const { kind, spec, params } = await readBody(req).catch(() => ({}));
        return json(res, await run(`window.__atlas.solve(${JSON.stringify(kind)}, ${JSON.stringify(spec)}, ${JSON.stringify(params)})`));
      }
      if (url.pathname === "/solver-kinds") return json(res, await run("window.__atlas.solverKinds()"));
      if (url.pathname === "/export") {
        const fmt = url.searchParams.get("format") === "json" ? "json" : "csv";
        const out = await run(`window.__atlas.exportData(${JSON.stringify(fmt)})`);
        if (fmt === "json") return json(res, out);
        const dir = path.join(ROOT, "docs", "exports");
        if (url.searchParams.get("save")) {
          fs.mkdirSync(dir, { recursive: true });
          fs.writeFileSync(path.join(dir, out.filename), out.csv);
        }
        res.writeHead(200, {
          "content-type": "text/csv; charset=utf-8",
          "cache-control": "no-store",
          "access-control-allow-origin": "*",
          "content-disposition": `attachment; filename="${out.filename}"`,
        });
        return res.end(out.csv);
      }
      if (url.pathname === "/casebook" && req.method === "POST") {
        return json(res, await run("window.__atlas.openCasebook()"));
      }
      if (url.pathname === "/case" && req.method === "POST") {
        const { index } = await readBody(req);
        return json(res, await run(`window.__atlas.openCase(${JSON.stringify(Number(index))})`));
      }
      if (url.pathname === "/hud" && req.method === "POST") {
        const { x, y } = await readBody(req).catch(() => ({}));
        return json(res, await run(`window.__atlas.showHud(${JSON.stringify(Number(x) || 320)}, ${JSON.stringify(Number(y) || 240)})`));
      }
      if (url.pathname === "/scene" && req.method === "POST") {
        const { id } = await readBody(req);
        return json(res, await run(`window.__atlas.scene(${JSON.stringify(id)})`));
      }
      if (url.pathname === "/material" && req.method === "POST") {
        const { id } = await readBody(req);
        return json(res, await run(`window.__atlas.material(${JSON.stringify(id)})`));
      }
      if (url.pathname === "/params" && req.method === "POST") {
        const { params } = await readBody(req);
        return json(res, await run(`window.__atlas.params(${JSON.stringify(params || {})})`));
      }
      if (url.pathname === "/time" && req.method === "POST") {
        const { value } = await readBody(req);
        return json(res, await run(`window.__atlas.time(${JSON.stringify(Number(value))})`));
      }
      if (url.pathname === "/shot") {
        const dataUrl = await run("window.__atlas.shot()");
        const buf = Buffer.from(String(dataUrl).split(",")[1] || "", "base64");
        if (url.searchParams.get("save")) {
          fs.writeFileSync(path.join(ROOT, "docs", "mcp-shot.png"), buf);
        }
        res.__body = "<png " + buf.length + " bytes>";
        res.writeHead(200, { "content-type": "image/png", "content-length": buf.length, "cache-control": "no-store" });
        return res.end(buf);
      }
      json(res, { error: "未知端点 " + url.pathname }, 404);
    } catch (e) {
      json(res, { error: String(e.message || e) }, 500);
    }
  });

  server.on("error", (e) => {
    console.log("MCP bridge 未启动：" + e.message);
  });
  server.listen(port, "127.0.0.1", () => {
    console.log(`MCP bridge → http://127.0.0.1:${port}`);
  });
}
