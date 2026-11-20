/* mcp-dock.js — 右下角的 MCP 状态与运行日志
 *
 * 收起：一个胶囊（连接点 + 端口 + 工具数 + 最近一次调用）
 * 展开：最近 40 条桥调用；点某一行展开该次调用的请求体 / 响应体
 *
 * 数据来自主进程的 /mcp/log（环形缓冲 200 条，含 reqBody/resBody 截断文本）。
 */

const POLL = 1600;
const MAX_ROWS = 40;

let dock = null;
let list = null;
let seq = 0;
let expanded = false;

const fmtTime = (t) => new Date(t).toLocaleTimeString("zh-CN", { hour12: false });

export function initMcpDock() {
  dock = document.getElementById("mcp-dock");
  if (!dock) return;
  list = dock.querySelector("#mcp-log");

  dock.querySelector("#mcp-head").addEventListener("click", (e) => {
    if (e.target.closest(".mcp-act")) return;
    expanded = !expanded;
    dock.dataset.expanded = expanded ? "1" : "0";
    if (expanded) poll();
  });
  dock.querySelector("#mcp-clear").addEventListener("click", (e) => {
    e.stopPropagation();
    if (list) list.innerHTML = "";
  });
  dock.querySelector("#mcp-copy").addEventListener("click", async (e) => {
    e.stopPropagation();
    const rows = [...(list ? list.querySelectorAll(".mcp-row") : [])].map((r) => r.textContent.replace(/\s+/g, " ").trim());
    try {
      await navigator.clipboard.writeText(rows.join("\n"));
      e.currentTarget.classList.add("ok");
      setTimeout(() => e.currentTarget.classList.remove("ok"), 900);
    } catch { /* ignore */ }
  });

  poll();
  setInterval(poll, POLL);
}

async function poll() {
  if (!dock) return;
  try {
    const res = await fetch(`http://127.0.0.1:8701/mcp/log?since=${seq}`, { cache: "no-store" });
    if (!res.ok) throw new Error(String(res.status));
    render(await res.json());
  } catch {
    renderOffline();
  }
}

function render(data) {
  dock.dataset.state = "on";
  seq = data.seq;

  const dot = dock.querySelector("#mcp-dot");
  const last = dock.querySelector("#mcp-last");
  dot.dataset.live = data.clientActive ? "1" : "0";
  dock.querySelector("#mcp-label").textContent = `MCP · ${data.port}`;
  dock.querySelector("#mcp-meta").textContent = `${data.tools} 工具 · ${data.clientActive ? "客户端已连接" : "等待客户端"}`;

  const fresh = (data.events || []).filter((e) => !e.internal);
  if (fresh.length) {
    last.textContent = fresh[fresh.length - 1].path;
    dock.dataset.flash = "1";
    setTimeout(() => (dock.dataset.flash = "0"), 380);
  } else if (!last.textContent) {
    last.textContent = "尚无调用";
  }

  if (!expanded || !fresh.length) return;
  for (const e of fresh.slice(-MAX_ROWS)) addRow(e);
  while (list.children.length > MAX_ROWS * 2) list.firstChild.remove();
  list.scrollTop = list.scrollHeight;
}

function addRow(e) {
  const row = document.createElement("div");
  row.className = "mcp-row";
  row.dataset.ok = e.status < 400 ? "1" : "0";
  row.dataset.id = String(e.id);
  row.innerHTML = `
    <span class="mcp-t">${fmtTime(e.t).slice(3)}</span>
    <span class="mcp-m">${e.method}</span>
    <span class="mcp-p">${e.path}</span>
    <span class="mcp-d">${e.ms}ms</span>
    <span class="mcp-s">${e.status}</span>`;

  if (e.reqBody || e.resBody) {
    row.classList.add("has-detail");
    row.addEventListener("click", (ev) => {
      ev.stopPropagation();
      toggleDetail(row, e);
    });
  }
  list.append(row);
}

function pretty(text) {
  if (!text) return "（空）";
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

export function toggleDetail(row, e) {
  const next = row.nextElementSibling;
  if (next && next.classList.contains("mcp-detail")) {
    next.remove();
    row.classList.remove("open");
    return;
  }
  row.classList.add("open");
  const box = document.createElement("div");
  box.className = "mcp-detail";
  box.innerHTML = `
    <div class="mcp-sec">
      <div class="mcp-sec-h">请求 · ${e.method} ${e.path}</div>
      <pre>${escapeHtml(pretty(e.reqBody))}</pre>
    </div>
    <div class="mcp-sec">
      <div class="mcp-sec-h">响应 · ${e.status} · ${e.ms}ms</div>
      <pre>${escapeHtml(pretty(e.resBody))}</pre>
    </div>`;
  row.after(box);
  if (list) list.scrollTop = list.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function renderOffline() {
  dock.dataset.state = "off";
  dock.querySelector("#mcp-dot").dataset.live = "0";
  dock.querySelector("#mcp-label").textContent = "MCP · 未连接";
  dock.querySelector("#mcp-meta").textContent = "本地桥无响应";
}
