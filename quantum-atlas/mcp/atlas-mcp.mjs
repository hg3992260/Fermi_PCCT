/* atlas-mcp.mjs — Quantum Atlas 的 MCP 服务器（stdio）
 *
 * 架构：本进程是"瘦 MCP 端"，所有状态与操作都转发给正在运行的 Electron 应用
 * 内置的本地桥（127.0.0.1:8701，见 main.cjs 的 startBridge）。
 * 这样 MCP 端不持有任何模型/工程状态，界面永远是唯一真源。
 *
 * 用法：
 *   node mcp/atlas-mcp.mjs            # 供 MCP 客户端以 stdio 方式拉起
 *   环境变量：ATLAS_BRIDGE=http://127.0.0.1:8701
 */

import { createInterface } from "node:readline";

const BRIDGE = process.env.ATLAS_BRIDGE || "http://127.0.0.1:8701";
const PROTOCOL = "2024-11-05";

async function bridge(path, body) {
  const res = await fetch(BRIDGE + path, {
    method: body ? "POST" : "GET",
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const type = res.headers.get("content-type") || "";
  if (!res.ok) throw new Error(`bridge ${path} → ${res.status} ${await res.text()}`);
  if (type.startsWith("image/")) {
    const buf = Buffer.from(await res.arrayBuffer());
    return { base64: buf.toString("base64"), bytes: buf.length };
  }
  return res.json();
}

const TOOLS = [
  {
    name: "atlas_state",
    description: "读取当前状态：所在图层、体系、主题、全部参数、读数行、空间标签。",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "atlas_scene",
    description:
      "切换图层。id ∈ atlas | crystal | bands | spin | spectra | liquid | bloch | qubit | circuit | device。",
    inputSchema: {
      type: "object",
      properties: { id: { type: "string" } },
      required: ["id"],
      additionalProperties: false,
    },
  },
  {
    name: "atlas_set_params",
    description:
      "设置参数（键值对），例如 {devBias: 600, devThick: 2} 或 {bandView: '曲面'}。只影响当前图层。",
    inputSchema: {
      type: "object",
      properties: { params: { type: "object", additionalProperties: true } },
      required: ["params"],
      additionalProperties: false,
    },
  },
  {
    name: "atlas_material",
    description: "切换体系（Si/Ge/C/Na/graphene）。图层 2（能带）与 4（光谱）会随之重算。",
    inputSchema: {
      type: "object",
      properties: { id: { type: "string" } },
      required: ["id"],
      additionalProperties: false,
    },
  },
  {
    name: "atlas_time",
    description: "把当前图层的扫描轴设到 0..1 的位置（k 路径 / 时间 / 光子能量 / 门序号）。",
    inputSchema: {
      type: "object",
      properties: { value: { type: "number", minimum: 0, maximum: 1 } },
      required: ["value"],
      additionalProperties: false,
    },
  },
  {
    name: "atlas_readout",
    description: "只取读数行（label/value），便于做数据抽取或对比。",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "atlas_screenshot",
    description: "截取三维视口，返回 PNG 的 base64（同一时间只能有一个返回，体积较大）。",
    inputSchema: {
      type: "object",
      properties: { save: { type: "boolean", description: "为 true 时同时写入 docs/mcp-shot.png" } },
      additionalProperties: false,
    },
  },
  {
    name: "atlas_export",
    description: "导出当前图层的数值结果（CSV 文本或 JSON 数据集），带材料/截断/网格等元数据；save=true 时写入应用目录 docs/exports/。",
    inputSchema: {
      type: "object",
      properties: {
        format: { type: "string", enum: ["csv", "json"] },
        save: { type: "boolean" },
      },
      additionalProperties: false,
    },
  },
  {
    name: "atlas_solve",
    description: "通用求解入口：对任意域（crystal/bands/spectra/spin/liquid/quantum/circuit/device）用统一 Problem→Solution 信封求解，返回 data/provenance/convergence。spec 是域相关的模型定义（如 {material: Si}、{kind: FM, S: 1, J1: 1, D: 0}、{n: 2, ops: [...]}）。",
    inputSchema: {
      type: "object",
      properties: {
        kind: { type: "string", description: "crystal|bands|spectra|spin|liquid|quantum|circuit|device" },
        spec: { type: "object", description: "域相关的模型定义" },
        params: { type: "object", description: "数值参数（截断/网格/…）" },
      },
      required: ["kind"],
      additionalProperties: false,
    },
  },
  {
    name: "atlas_solver_kinds",
    description: "列出通用求解器已注册的域。",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "atlas_cases",
    description: "列出案例库里的 12 个算例（layer / tag / title / 复现参数），用于回答「这个程序能算什么」。",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "atlas_casebook",
    description: "打开应用内的案例库子页面（全屏、可一键复现案例）。",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "atlas_open_case",
    description: "按索引打开某个案例并把界面切到能复现它的状态（图层 + 体系 + 参数）。",
    inputSchema: {
      type: "object",
      properties: { index: { type: "number", description: "atlas_cases 返回的序号，从 0 开始" } },
      required: ["index"],
      additionalProperties: false,
    },
  },
  {
    name: "atlas_hud",
    description: "在弹出的悬浮面板里显示当前图层的实时读数（x/y 为屏幕坐标，可省略）。",
    inputSchema: {
      type: "object",
      properties: { x: { type: "number" }, y: { type: "number" } },
      additionalProperties: false,
    },
  },
  {
    name: "atlas_layers",
    description: "列出全部图层（id / 名称 / 空间 / 参数集），便于决定下一步操作。",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
];

async function callTool(name, args) {
  switch (name) {
    case "atlas_state":
      return bridge("/state");
    case "atlas_layers":
      return bridge("/layers");
    case "atlas_cases":
      return bridge("/cases");
    case "atlas_solve":
      return bridge("/solve", { kind: args.kind, spec: args.spec ?? {}, params: args.params ?? {} });
    case "atlas_solver_kinds":
      return bridge("/solver-kinds");
    case "atlas_export": {
      const fmt = args.format === "json" ? "json" : "csv";
      if (fmt === "json") return bridge("/export?format=json");
      const res = await fetch(BRIDGE + "/export?format=csv" + (args.save ? "&save=1" : ""));
      const text = await res.text();
      return { csv: text.slice(0, 200000), bytes: text.length, saved: !!args.save };
    }
    case "atlas_casebook":
      return bridge("/casebook", {});
    case "atlas_open_case":
      return bridge("/case", { index: args.index });
    case "atlas_hud":
      return bridge("/hud", { x: args.x, y: args.y });
    case "atlas_readout":
      return (await bridge("/state")).readout;
    case "atlas_scene":
      return bridge("/scene", { id: args.id });
    case "atlas_material":
      return bridge("/material", { id: args.id });
    case "atlas_time":
      return bridge("/time", { value: args.value });
    case "atlas_set_params":
      return bridge("/params", { params: args.params || {} });
    case "atlas_screenshot":
      return bridge("/shot" + (args.save ? "?save=1" : ""));
    default:
      throw new Error("未知工具：" + name);
  }
}

const send = (msg) => process.stdout.write(JSON.stringify(msg) + "\n");

async function handle(msg) {
  const { id, method, params } = msg;
  if (method === "initialize") {
    return send({
      jsonrpc: "2.0", id,
      result: {
        protocolVersion: PROTOCOL,
        capabilities: { tools: {} },
        serverInfo: { name: "quantum-atlas", version: "0.9.0" },
      },
    });
  }
  if (method === "notifications/initialized" || method === "initialized") return;
  if (method === "ping") return send({ jsonrpc: "2.0", id, result: {} });
  if (method === "tools/list") return send({ jsonrpc: "2.0", id, result: { tools: TOOLS } });
  if (method === "tools/call") {
    const { name, arguments: args } = params || {};
    try {
      const out = await callTool(name, args || {});
      const isImg = out && out.base64;
      return send({
        jsonrpc: "2.0", id,
        result: {
          content: isImg
            ? [{ type: "image", data: out.base64, mimeType: "image/png" }]
            : [{ type: "text", text: JSON.stringify(out, null, 2) }],
          isError: false,
        },
      });
    } catch (e) {
      return send({
        jsonrpc: "2.0", id,
        result: { content: [{ type: "text", text: String(e.message || e) }], isError: true },
      });
    }
  }
  if (id !== undefined) {
    send({ jsonrpc: "2.0", id, error: { code: -32601, message: "未实现的方法：" + method } });
  }
}

const rl = createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const text = line.trim();
  if (!text) return;
  let msg;
  try {
    msg = JSON.parse(text);
  } catch {
    return;
  }
  handle(msg).catch((e) => {
    if (msg && msg.id !== undefined) {
      send({ jsonrpc: "2.0", id: msg.id, error: { code: -32603, message: String(e.message || e) } });
    }
  });
});
