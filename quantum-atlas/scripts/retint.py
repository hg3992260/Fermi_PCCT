import io, re, sys

path = r"I:\Fermi\quantum-atlas\app.css"
src = io.open(path, encoding="utf-8").read()

start = src.index(":root {")
end = src.index("* { box-sizing: border-box; }")
head = """/* ── 设计令牌（西门子医疗 UI MarCom 调色板） ────────────────
   品牌色：Healthy Orange #ec6602（主操作/激活态）、Siemens Petrol #009999（次级）
   中性色：white / #f2f2f2 / #e6e6e6 / #cccccc / #999999 / #666666 / #1a1a1a
   附加色：Cyan #3abfed、Blue #2b2483、Berry #7a162d
   功能色：Green #009a38（可访问 #197c3a）、Red #e7001d（可访问 #d9001d）
   ─────────────────────────────────────────────────────────── */
:root {
  --font-ui: "Segoe UI Variable Text", "Segoe UI", "Inter", "PingFang SC", "Microsoft YaHei UI", system-ui, sans-serif;
  --font-mono: "Cascadia Mono", "Consolas", ui-monospace, monospace;

  --titlebar-h: 46px;
  --rail-w: 236px;
  --insp-w: 332px;
  --transport-h: 56px;
  --status-h: 26px;

  /* 浅色（默认） */
  --bg-0: #ffffff;
  --bg-1: #ffffff;
  --bg-2: #ffffff;
  --bg-3: #f2f2f2;
  --line: #e6e6e6;
  --line-2: #cccccc;
  --ink: #1a1a1a;
  --dim: #666666;
  --faint: #999999;
  --accent: #ec6602;
  --accent-soft: rgba(236, 102, 2, 0.10);
  --petrol: #009999;
  --teal: #009999;
  --magenta: #7a162d;
  --amber: #cf4b00;
  --blue: #2b2483;
  --good: #197c3a;
  --bad: #cf4b00;

  --vp-0: #ffffff;
  --vp-1: #eef2f2;
  --vp-grid: rgba(0, 0, 0, 0.06);
  --vp-ink: #1a1a1a;

  --glass-bg: rgba(255, 255, 255, 0.82);
  --glass-line: rgba(0, 0, 0, 0.10);
  --glass-ink: #1a1a1a;

  --ax-x: #ec6602;
  --ax-y: #009a38;
  --ax-z: #2b2483;

  --shadow-1: 0 1px 2px rgba(16, 24, 32, 0.06);
  --shadow-2: 0 12px 28px rgba(16, 24, 32, 0.12);
}

body.dark {
  --bg-0: #0a1113;
  --bg-1: #0c1416;
  --bg-2: #0e1618;
  --bg-3: #142022;
  --line: #1d2b2d;
  --line-2: #2a3a3d;
  --ink: #eef2f6;
  --dim: #9fb0b3;
  --faint: #6f8386;
  --accent: #ff9a3c;
  --accent-soft: rgba(255, 154, 60, 0.14);
  --petrol: #3fc7c7;
  --teal: #3fc7c7;
  --magenta: #ff9d7a;
  --amber: #ffb04d;
  --blue: #9a95d6;
  --good: #57c777;
  --bad: #ff9a3c;

  --vp-0: #0a1113;
  --vp-1: #101a1c;
  --vp-grid: rgba(160, 190, 195, 0.10);
  --vp-ink: #d7e2e4;

  --glass-bg: rgba(8, 16, 18, 0.66);
  --glass-line: rgba(160, 190, 195, 0.18);
  --glass-ink: #d7e2e4;

  --ax-x: #ff9a7a;
  --ax-y: #8fe0a8;
  --ax-z: #9a95d6;

  --shadow-1: 0 1px 2px rgba(0, 0, 0, 0.4);
  --shadow-2: 0 14px 32px rgba(0, 0, 0, 0.5);
}

"""
src = head + src[end:]

# 视口浮层：改用主题令牌（浅色为白玻璃，深色为暗玻璃）
src = src.replace(""".viewport {
  --glass-bg: rgba(8, 12, 18, 0.62);
  --glass-line: rgba(150, 175, 215, 0.16);
  --glass-ink: #cfd9ea;
}""", """.viewport {
  /* 浮层玻璃与坐标轴颜色都来自 :root / body.dark 的令牌 */
}""")

# 3D 相机导轨的 hover 背景保持中性
src = src.replace(".viewport .cam-rail x-button[skin=\"dock\"]:hover { background: rgba(30, 42, 58, 0.8); }",
                  ".viewport .cam-rail x-button[skin=\"dock\"]:hover { background: color-mix(in srgb, var(--glass-bg) 70%, var(--line-2)); }")

# 坐标轴三色改为令牌
src = src.replace(""".gizmo .ax-x line, .gizmo .ax-x text { stroke: #ff8080; fill: #ffb3b3; }
.gizmo .ax-y line, .gizmo .ax-y text { stroke: #9ae66e; fill: #c4f0a8; }
.gizmo .ax-z line, .gizmo .ax-z text { stroke: #6aa9ff; fill: #a9c8ff; }""",
""".gizmo .ax-x line, .gizmo .ax-x text { stroke: var(--ax-x); fill: var(--ax-x); }
.gizmo .ax-y line, .gizmo .ax-y text { stroke: var(--ax-y); fill: var(--ax-y); }
.gizmo .ax-z line, .gizmo .ax-z text { stroke: var(--ax-z); fill: var(--ax-z); }""")

# 视口背景：画布自己画渐变，CSS 只兜底
src = src.replace("""  background: radial-gradient(1200px 640px at 58% 12%, var(--vp-1), var(--vp-0) 72%);
}""", """  background: var(--vp-0);
}""", 1)

io.open(path, "w", encoding="utf-8").write(src)
print("app.css tokens replaced, size", len(src))
