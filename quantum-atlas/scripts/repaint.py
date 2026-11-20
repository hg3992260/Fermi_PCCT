import io, re

base = r"I:\Fermi\quantum-atlas"

def patch(rel, pairs):
    p = base + "\\" + rel
    s = io.open(p, encoding="utf-8").read()
    for a, b in pairs:
        if a not in s:
            print("MISS", rel, a[:60].replace("\n", " "))
            continue
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8").write(s)
    print("ok", rel)

# 1) 默认主题 = 浅色
patch("index.html", [
    ('<meta name="xel-theme" content="theme/atlas-dark.css" />',
     '<meta name="xel-theme" content="theme/atlas-light.css" />'),
])

# 2) util：曲线支持 width（挤出成带），热力色改成品牌色序列
patch("src/scenes/util.js", [
    ("""  thermal: [hex("#2b4b8f"), hex("#37d6c0"), hex("#ffd166"), hex("#ff6b6b")],""",
     """  thermal: [hex("#02787d"), hex("#009999"), hex("#3abfed"), hex("#ec6602"), hex("#7a162d")],"""),
    ("""export function curve3(v, pts, colorA, colorB, alpha = 0.9, width = 1) {
  const n = pts.length;
  for (let i = 0; i < n - 1; i++) {
    const t = i / Math.max(1, n - 2);
    const c = Array.isArray(colorA) && Array.isArray(colorB) && colorA.length === 3 && colorB.length === 3
      ? V3.lerp(colorA, colorB, t)
      : colorA;
    v.line(pts[i], pts[i + 1], c, alpha);
  }
}""",
     """export function curve3(v, pts, colorA, colorB, alpha = 0.9, width = 0) {
  const n = pts.length;
  for (let i = 0; i < n - 1; i++) {
    const t = i / Math.max(1, n - 2);
    const c = Array.isArray(colorA) && Array.isArray(colorB) && colorA.length === 3 && colorB.length === 3
      ? V3.lerp(colorA, colorB, t)
      : colorA;
    if (width > 0) v.thick([pts[i], pts[i + 1]], c, width, alpha);
    else v.line(pts[i], pts[i + 1], c, alpha);
  }
}"""),
])

# 3) 能带：去掉硬编码色 + 主曲线用带状
patch("src/scenes/bands.js", [
    ('const col = b === sel ? (occ ? C.amber : C.cyan) : occ ? hex("#4f74a8") : hex("#7e63c4");',
     'const col = b === sel ? C.amber : occ ? C.ice : C.violet;'),
    ("""      let prev = null;
      for (let i = 0; i < nPts; i++) {
        const p = [X(i), EY(E[i][b]), 0];
        if (prev) v.line(prev, p, col, a);
        prev = p;
      }""",
     """      const pts = [];
      for (let i = 0; i < nPts; i++) pts.push([X(i), EY(E[i][b]), 0]);
      v.thick(pts, col, b === sel ? 0.024 : 0.013, a);"""),
    ("""      const pp = kp.map((p) => [bzo[0] + p[0] * ks, bzo[1], bzo[2] + p[1] * ks]);
      v.poly(pp, C.amber, { a: 0.9 });""",
     """      const pp = kp.map((p) => [bzo[0] + p[0] * ks, bzo[1], bzo[2] + p[1] * ks]);
      v.thick(pp, C.amber, 0.022, 0.95);"""),
])

# 4) 自旋 / 液态：品牌双色
patch("src/scenes/spin.js", [
    ('const col = V3.lerp(hex("#ff7ac8"), hex("#37d6c0"), (sp.m[1] + 1) / 2);',
     'const col = V3.lerp(C.pink, C.ice, (sp.m[1] + 1) / 2);'),
    ("const pts = mg.q.map((q, i) => [x0 + (wSpan * i) / (mg.q.length - 1), (mg.w[i] / wMax) * hSpan, 0]);\n    curve3(v, pts, C.cyan, C.violet, 0.95);",
     "const pts = mg.q.map((q, i) => [x0 + (wSpan * i) / (mg.q.length - 1), (mg.w[i] / wMax) * hSpan, 0]);\n    curve3(v, pts, C.ice, C.violet, 0.95, 0.02);"),
])
patch("src/scenes/liquid.js", [
    ('const col = V3.lerp(hex("#ff7ac8"), hex("#37d6c0"), (c + 1) / 2);',
     'const col = V3.lerp(C.pink, C.ice, (c + 1) / 2);'),
])
patch("src/scenes/qubit.js", [
    ('const col = V3.lerp(hex("#4c9aff"), hex("#ff7ac8"), (nd.phase + Math.PI) / (2 * Math.PI));',
     'const col = V3.lerp(C.ice, C.amber, (nd.phase + Math.PI) / (2 * Math.PI));'),
])
patch("src/scenes/circuit.js", [
    ('const col = V3.lerp(hex("#4c9aff"), hex("#ff7ac8"), (ph + Math.PI) / (2 * Math.PI));',
     'const col = V3.lerp(C.ice, C.amber, (ph + Math.PI) / (2 * Math.PI));'),
    ("v.line(lg[i], lg[i + 1], V3.lerp(hex(\"#4c9aff\"), hex(\"#ff7ac8\"), i / 12), 0.9);",
     "v.line(lg[i], lg[i + 1], V3.lerp(C.ice, C.amber, i / 12), 0.9);"),
])
patch("src/scenes/crystal.js", [
    ("const SPECIES_COLOR = { Si: \"#8fb8ff\", Ge: \"#7c9cff\", C: \"#f2f0ff\", Na: \"#ffb84d\", \"³He\": \"#8fa3bf\" };",
     "// 原子统一用品牌主色（Petrol），离子/元素差异留给标签"),
    ("const col = hex(SPECIES_COLOR[a.species] || mat.color);", "const col = C.ice;"),
])
patch("src/scenes/spectra.js", [
    ("curve3(v, p2, C.violet, C.violet, 0.45);", "curve3(v, p2, C.violet, C.violet, 0.5, 0.012);"),
    ("curve3(v, p1, C.amber, C.amber, 0.5);", "curve3(v, p1, C.amber, C.amber, 0.55, 0.012);"),
])
patch("src/scenes/bloch.js", [
    ("curve3(v, pts, C.cyan, C.violet, 0.95);", "curve3(v, pts, C.ice, C.violet, 0.95, 0.02);"),
])
print("done")
