# Quantum Atlas · MCP 服务

把正在运行的三维工作台暴露成 MCP 工具，让 agent 能读状态、切图层、改参数、取截图。

## 两个进程

```
MCP 客户端 ──stdio──> mcp/atlas-mcp.mjs ──HTTP──> 127.0.0.1:8701（Electron 内置桥）──> 渲染进程
```

- **桥在应用里**（`main.cjs` 的 `startBridge()`）：只监听回环地址，端点
  `/health` `/state` `/layers` `/scene` `/params` `/material` `/time` `/shot`。
- **MCP 端是瘦客户端**（`mcp/atlas-mcp.mjs`）：不持有任何状态，界面永远是唯一真源。
  所以 agent 改的东西，用户屏幕上是当场可见的；用户手改的东西，agent 下次读状态也能看到。

端口可用 `ATLAS_MCP_PORT` 改；MCP 端用 `ATLAS_BRIDGE` 指向桥。

## 注册（DSH / Claude Desktop 等）

```json
{
  "mcpServers": {
    "quantum-atlas": {
      "command": "node",
      "args": ["I:/Fermi/quantum-atlas/mcp/atlas-mcp.mjs"],
      "env": { "ATLAS_BRIDGE": "http://127.0.0.1:8701" }
    }
  }
}
```

前提：**应用要先开着**（桥由应用进程启动）。MCP 端在调用时会拿到 `fetch failed`，此时先启动 `pnpm start`。

## 工具一览

| 工具 | 作用 |
|---|---|
| `atlas_state` | 当前图层 / 体系 / 主题 / 全部参数 / 读数行 / 扫描轴位置 |
| `atlas_layers` | 十个图层的 id、名称、空间标签、参数键（用于决定下一步） |
| `atlas_scene` | 切图层（`atlas` … `device`） |
| `atlas_material` | 切体系（Si/Ge/C/Na/graphene），L2/L4 会重算 |
| `atlas_set_params` | 改参数，如 `{devBias: 600, devThick: 2}` |
| `atlas_time` | 把扫描轴设到 0..1（k 路径 / 时间 / 光子能量 / 门序号） |
| `atlas_readout` | 只取读数（label/value），适合做数据抽取与对比 |
| `atlas_screenshot` | 截三维视口，返回 PNG base64（`save:true` 同时写 `docs/mcp-shot.png`） |

## 自检

```bash
pnpm test:mcp     # 9 项：initialize / tools/list / 六个工具 + 截图
```

`scripts/test-mcp.mjs` 会先看 8701 是否已有实例——有就复用，没有就自己拉起一个再关掉。

## 为什么应用里不内置 agent

内嵌 agent 框架要跟着它的版本走、还得复刻一套 UI；而 agent 真正需要的就是这十来个工具。
所以应用只做工具面，规划交给外部客户端——同一个 tlas-mcp 可以被 DSH、opencode、Claude Desktop 共用。

## 设计约束

- 桥**只绑 127.0.0.1**，不对外网暴露；没有任意代码执行端点，只有白名单方法。
- 渲染端 API 在 `src/main.js` 的 `window.__atlas`：`state/layers/scene/params/material/time/shot`。
- `params` 只写 `S` 上的键，不触发重建以外的副作用；`scene` 复用与点导航完全相同的路径，
  因此 agent 的操作会被控件自检与真机探针覆盖到。
