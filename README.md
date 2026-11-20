# Quantum Atlas · 量子材料全尺度三维工作台

把「凝聚态 → 能带 → 自旋 → 光谱/半导体响应 → 液态自旋 → Bloch 动力学 → 量子计算 → 量子电路」
做成**一条重正化流的九个站点**，每站一个可交互的三维视图，参数与读数实时联动。

- 界面：Xel（纯 Web Components，无构建步骤）+ 自研 WebGL2 渲染器（零第三方 3D 依赖）
- 物理：经验赝势能带（真实对角化）、JDOS + Kramers–Kronig 介电函数、线性自旋波、
  Bloch 方程 RK4、复数态矢量子电路模拟
- 文档：[设计说明](docs/设计说明.md)（理论框架 / 界面规范 / 保真度边界）

## 快速开始

```bash
pnpm install          # 只有一个纯前端依赖 xel
pnpm serve            # 浏览器预览 → http://127.0.0.1:8123/

# 三层验证
pnpm test             # 三件一起跑：物理内核 → 浏览器控件 → 真机交互
pnpm test:physics     # 只跑物理内核（8 组断言）
pnpm test:audit       # 只跑浏览器控件自检（?audit=1，84 条）
pnpm test:ui          # 只跑真机交互探针（--probe-ui，11 条，真实鼠标事件）
```

Electron 桌面窗口（需 electron ≥ 28，主进程为 ESM）：

```bash
pnpm add -D electron
pnpm start
```

## 快捷键

| 键 | 作用 |
|---|---|
| `1` … `9` | 切换 L0–L8 |
| `Space` | 播放 / 暂停时间轴 |
| `←` / `→` | 单步 |
| `R` | 重置相机 |
| `T` | 切换明暗 |
| `F1` | 理论框架抽屉 |

## 深链接

- `/?scene=bands&material=Ge` — 直接进入某层某材料
- `/?snap=1` — 快照模式（无头浏览器截图用：把 WebGL 缓冲转成 `<img>`）

## 目录

```
index.html  app.css  main.cjs  preload.cjs
theme/                 atlas-dark.css / atlas-light.css
                       （@import Xel 的 fluent 主题后，只改语义令牌与几何）
src/main.js            外壳：菜单 / 导航 / 检查器 / 传送带 / 拾取 / 主题
src/engine.js          WebGL2 批渲染器（标签去重叠、选中环、网格径向淡出）
src/phys/*.js          materials epm spectra spin liquid bloch quantum
src/scenes/*.js        atlas crystal bands spin spectra liquid bloch qubit circuit device
                       （device = 从 Fermi/PyQt 迁移来的探测器层，见 docs/迁移对照.md）
scripts/               serve.mjs  test-physics.mjs  calibrate.mjs
docs/                  设计说明.md + 各层快照 PNG
```

## 界面要点

- **无边框窗口** + Xel `<x-titlebar>`：标题栏内含菜单栏（文件/视图/图层/帮助）、体系选择、图层徽标。
- **左栏**分组导航（全景与结构 / 电子结构 / 自旋与响应 / 动力学与量子）+ 尺度条。
- **视口恒为暗场**（浅色主题只换 chrome），浮层统一用深色玻璃；空间徽标标明"实/倒/态/参"。
- **右栏**：图层头 + 控制（由场景声明自动生成）+ 读数（等宽数字）+ 可折叠「本层物理」。
- **传送带**：播放/暂停、单步、滑块、数值+单位；每层语义不同（k 路径 / 时间 / 光子能量 / 门序号）。
- **迷你坐标轴**：视口右下，随相机旋转；**读数趋势条**：自动挑"随扫描参数变化最明显"的量画成 26px 折线
  （带 120ms 预算保护，太贵就自动放弃）；**参数预设**：恢复默认 / 保存为默认 / 复制 JSON。
- 快捷键：`1–9` 图层 · `Space` 播放 · `←/→` 步进 · `R` 相机 · `T` 主题 · `F1` 理论框架。

## 物理内核一览

| 模块 | 方法 | 校验 |
|---|---|---|
| `epm.js` | 平面波 EPM + Jacobi 实对称本征分解 | Si 1.135 / Ge 0.653 / C 5.485 eV（实验 1.12 / 0.66 / 5.47） |
| `spectra.js` | JDOS（含动量矩阵元）+ Kramers–Kronig + Drude | ε∞ 归一到实验值；E_g(T) 用 Varshni |
| `spin.js` | 线性自旋波、Brillouin 自洽解、磁滞唯象 | 磁振子带宽随 J、D 单调 |
| `liquid.js` | RSA 构型 → g(r) → S(q) → 扩散型 S(q,ω) | 数密度 0.0164 Å⁻³（液态 ³He） |
| `bloch.js` | Bloch 定理波函数、Bloch 球、RK4 | |M| ≤ 1，T₁/T₂ 极限行为正确 |
| `quantum.js` | 复数态矢、门矩阵、Q-sphere、concurrence | Bell 态 concurrence = 1 |
