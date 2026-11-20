import io, json, re, sys, urllib.request, time

ELEMENTS = {
    "C": 6, "Si": 14, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Br": 35,
    "Cd": 48, "Te": 52, "I": 53, "Cs": 55, "Hg": 80, "Tl": 81, "Pb": 82,
}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"}
BASE = "https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z%02d.html"

out = {}
for sym, z in ELEMENTS.items():
    url = BASE % z
    try:
        req = urllib.request.Request(url, headers=UA)
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    except Exception as e:
        print("FAIL", sym, type(e).__name__, e)
        continue
    # 行结构：<TD>能量</TD><TD>μ/ρ</TD><TD>μen/ρ</TD>
    rows = re.findall(
        r"<TD[^>]*>\s*([0-9.]+E[+-][0-9]+)\s*</TD>\s*<TD[^>]*>\s*([0-9.]+E[+-][0-9]+)\s*</TD>",
        html, re.I)
    pts = []
    for e_mev, mu in rows:
        e = float(e_mev) * 1000.0   # keV
        if 4.0 <= e <= 600.0:
            pts.append([round(e, 4), float(mu)])
    pts.sort()
    out[sym] = pts
    print(sym, len(pts), pts[0] if pts else None, pts[-1] if pts else None)
    time.sleep(0.2)

path = r"I:\Fermi\quantum-atlas\src\phys\atten-nist.js"
body = json.dumps(out, separators=(",", ":"))
js = ("/* NIST XCOM 质量衰减系数 μ/ρ (cm²/g)，能量单位 keV。\n"
      "   数据来源：NIST Physical Measurement Laboratory, X-Ray Mass Attenuation\n"
      "   Coefficients, https://physics.nist.gov/PhysRefData/XrayMassCoef/ （公开数据）。\n"
      "   由 scripts/fetch-nist.py 抓取，仅保留 4–600 keV 段。 */\n"
      "export const NIST = " + body + ";\n")
io.open(path, "w", encoding="utf-8").write(js)
print("written", path, len(js), "bytes")
