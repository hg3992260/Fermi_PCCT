import io, json, os, re, time, urllib.parse, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
OUT = r"I:\Fermi\quantum-atlas\docs\figs"
os.makedirs(OUT, exist_ok=True)

def get(url, timeout=45):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read()

def html(url):
    return get(url).decode("utf-8", "replace")

def ids(query, n=3):
    q = urllib.parse.quote(query)
    xml = html(f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results={n}&sortBy=relevance")
    return re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", xml)

TOPICS = {
    "L1": 'all:"crystal structure" AND all:"diamond" AND all:"figure" AND cat:cond-mat.mtrl-sci',
    "L6": 'all:"Bloch sphere" AND all:Rabi AND all:qubit',
    "L7": 'all:"quantum state" AND all:sphere AND all:visualization AND all:qubit',
}

result = {}
for tag, q in TOPICS.items():
    try:
        found = ids(q, 2)
    except Exception as e:
        result[tag] = {"err": type(e).__name__}
        continue
    recs = []
    for aid in found:
        rec = {"id": aid, "caps": [], "imgs": []}
        try:
            doc = html(f"https://ar5iv.org/abs/{aid}")
            caps = re.findall(r"<figcaption[^>]*>(.*?)</figcaption>", doc, re.S)
            rec["caps"] = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip()[:260] for c in caps[:3]]
            rec["imgs"] = re.findall(r'<img[^>]+src="([^"]+\.(?:png|jpg|jpeg))"', doc)[:4]
        except Exception as e:
            rec["err"] = type(e).__name__
        recs.append(rec)
        time.sleep(0.3)
    result[tag] = recs
    print(tag, [r["id"] for r in recs], flush=True)

# 下载几张代表性配图，供直接查看
WANT = [
    ("L2_bands", "1512.05403v2"),
    ("L8_circuit", "2307.08969v1"),
]
for name, aid in WANT:
    try:
        doc = html(f"https://ar5iv.org/abs/{aid}")
        imgs = re.findall(r'<img[^>]+src="([^"]+\.(?:png|jpg|jpeg))"', doc)[:2]
        for i, src in enumerate(imgs):
            url = src if src.startswith("http") else "https://ar5iv.org" + ("" if src.startswith("/") else "/") + src
            try:
                data = get(url, 60)
            except Exception:
                url = "https://ar5iv.labs.arxiv.org" + src if not src.startswith("http") else src
                data = get(url, 60)
            ext = ".png" if ".png" in url else ".jpg"
            p = os.path.join(OUT, f"{name}_{i}{ext}")
            io.open(p, "wb").write(data)
            print("saved", p, len(data))
    except Exception as e:
        print("dl fail", name, type(e).__name__, e)

io.open(r"I:\Fermi\quantum-atlas\docs\fig-survey2.json", "w", encoding="utf-8").write(json.dumps(result, ensure_ascii=False, indent=1))
print("done")
