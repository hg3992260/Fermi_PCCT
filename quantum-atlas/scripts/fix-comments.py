import io, re

p = r"I:\Fermi\quantum-atlas\main.cjs"
lines = io.open(p, encoding="utf-8").read().split("\n")

FIX = {
    1: "// Electron 主进程（CommonJS）：ESM 在这里会解析到 npm 包，必须用 require",
    2: "// 走自定义协议 app:// 提供页面资源，否则 Xel 的图标/主题 fetch 在 file:// 下会失败",
    23: "// Windows 任务栏分组 / 图标：必须在窗口创建前设置",
    26: "// 自检时用临时 profile，避免与正在运行的实例争抢 userData 缓存锁",
    84: "// 无头自检：不显示窗口，加载完成后回读运行时状态并抓帧，然后退出",
    182: "// 真实鼠标事件探针（--probe-ui）",
}

for no, text in FIX.items():
    idx = no - 1
    if 0 <= idx < len(lines):
        comment_only = re.match(r"^(\s*)//", lines[idx])
        if comment_only:
            lines[idx] = comment_only.group(1) + text
        else:
            lines.insert(idx, text)

# 133 行是多行注释块，整段替换成两句
s = "\n".join(lines)
s = re.sub(
    r"//[^\n]*no-store[^\n]*\n\s*//[^\n]*\n\s*try \{",
    "// 自己读盘并显式给 no-store：否则 Chromium 会缓存 app:// 响应，\n"
    "    // 改完 CSS/JS 重启仍看到旧界面（这个坑很隐蔽）。\n    try {",
    s,
)
s = re.sub(
    r"//[^\n]*before-input[^\n]*\n",
    "",
    s,
)
io.open(p, "w", encoding="utf-8").write(s)
print("comments rewritten")
