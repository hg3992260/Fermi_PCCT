import io, json, os, re, tarfile, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"}
ROOT = r"I:\Fermi\quantum-atlas\docs\figs"

PAPERS = {
    "L1_crystal": ["1803.09279"],
    "L2_bands": ["1512.05403"],
    "L3_spin": ["2308.09407"],
    "L4_dielectric": ["0801.1384"],
    "L5_liquid": ["1204.0348"],
    "L6_bloch": ["2606.01405"],
    "L7_qsphere": ["2303.08366"],
    "L8_circuit": ["2307.08969"],
}

def get(url, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read()

def captions_from_tex(tex):
    caps = re.findall(r"\\caption\s*\{((?:[^{}]|\{[^{}]*\})*)\}", tex, re.S)
    out = []
    for c in caps:
        t = re.sub(r"\\[a-zA-Z]+\s*", " ", c)
        t = re.sub(r"[{}$\\]", "", t)
        out.append(re.sub(r"\s+", " ", t).strip()[:260])
    return out

manifest = {}
for layer, ids in PAPERS.items():
    for aid in ids:
        dest = os.path.join(ROOT, f"{layer}_{aid}")
        os.makedirs(dest, exist_ok=True)
        rec = {"id": aid, "dir": dest, "figs": [], "captions": [], "err": None}
        try:
            blob = get(f"https://arxiv.org/e-print/{aid}")
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as t:
                t.extractall(dest)
            files = []
            for base, _, names in os.walk(dest):
                for n in names:
                    files.append(os.path.join(base, n))
            figs = [f for f in files if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".eps"))]
            figs.sort(key=lambda f: os.path.getsize(f), reverse=True)
            rec["figs"] = [{"path": os.path.relpath(f, ROOT), "kb": os.path.getsize(f) // 1024} for f in figs[:10]]
            for f in files:
                if f.endswith(".tex"):
                    try:
                        rec["captions"] = captions_from_tex(io.open(f, encoding="utf-8", errors="replace").read())[:6]
                    except Exception:
                        pass
                    break
            # 渲染最大的三张 PDF 图
            import fitz
            rendered = []
            for f in figs[:6]:
                if not f.lower().endswith(".pdf"):
                    continue
                try:
                    doc = fitz.open(f)
                    page = doc[0]
                    pix = page.get_pixmap(dpi=130)
                    out = f[:-4] + "_render.png"
                    pix.save(out)
                    rendered.append(os.path.relpath(out, ROOT))
                    doc.close()
                    if len(rendered) >= 3:
                        break
                except Exception as e:
                    rec["err"] = f"{type(e).__name__}: {e}"[:120]
            rec["rendered"] = rendered
        except Exception as e:
            rec["err"] = f"{type(e).__name__}: {e}"[:160]
        manifest[f"{layer}_{aid}"] = rec
        print(layer, aid, "figs", len(rec["figs"]), "rendered", len(rec.get("rendered", [])), rec["err"] or "", flush=True)

io.open(os.path.join(ROOT, "manifest.json"), "w", encoding="utf-8").write(json.dumps(manifest, ensure_ascii=False, indent=1))
print("done")
