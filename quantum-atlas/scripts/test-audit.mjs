/* test-audit.mjs — 浏览器层：起临时静态服务器 + 无头 Edge 跑 ?audit=1，解析标题里的结果 */
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml",
  ".png": "image/png", ".woff2": "font/woff2", ".ftl": "text/plain; charset=utf-8",
};

const EDGE = process.env.EDGE_PATH || [
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
].find(existsSync);

if (!EDGE) {
  console.error("找不到 Edge，可用 EDGE_PATH 指定");
  process.exit(2);
}

const server = createServer(async (req, res) => {
  const rel = decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/^\/+/, "") || "index.html";
  const file = path.join(ROOT, rel);
  try {
    const buf = await readFile(file);
    res.writeHead(200, { "content-type": MIME[path.extname(file).toLowerCase()] || "application/octet-stream" });
    res.end(buf);
  } catch {
    res.writeHead(404).end("not found");
  }
});

const port = await new Promise((r) => server.listen(0, "127.0.0.1", () => r(server.address().port)));
const url = `http://127.0.0.1:${port}/?audit=1`;

const args = [
  "--headless=new", "--enable-unsafe-swiftshader", "--disable-gpu-sandbox",
  `--user-data-dir=${path.join(process.env.TEMP || "/tmp", `qa-audit-${Date.now()}`)}`,
  "--window-size=1400,900", "--virtual-time-budget=200000", "--dump-dom", url,
];

const dom = await new Promise((resolve, reject) => {
  const p = spawn(EDGE, args, { stdio: ["ignore", "pipe", "ignore"] });
  let out = "";
  p.stdout.on("data", (d) => (out += d));
  const timer = setTimeout(() => { p.kill(); reject(new Error("无头浏览器超时")); }, 300000);
  p.on("exit", () => { clearTimeout(timer); resolve(out); });
  p.on("error", reject);
});

server.close();

const m = dom.match(/<title>AUDIT (\{[\s\S]*?\})<\/title>/);
if (!m) {
  const err = dom.match(/<title>([^<]*)<\/title>/);
  console.error("✗ 浏览器自检未产出结果。title =", err ? err[1] : "(无)");
  process.exit(1);
}

let r;
try {
  r = JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">"));
} catch (e) {
  console.error("✗ 结果解析失败：", e.message);
  process.exit(1);
}

const errs = Array.isArray(r.errs) ? r.errs.length : 0;
console.log(`  控件自检：${r.total - r.fail}/${r.total} 通过${errs ? `（${errs} 条已知控件库噪声）` : ""}`);
if (r.fail) {
  for (const b of r.bad) console.error("  ✗ " + b);
  process.exit(1);
}
