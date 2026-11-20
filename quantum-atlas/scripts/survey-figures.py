import io, json, re, sys, time, urllib.parse, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

TOPICS = {
    "L1_crystal": 'all:"diamond cubic" AND all:"crystal structure" AND all:silicon',
    "L2_bands": 'all:"band structure" AND all:"pseudopotential" AND all:silicon',
    "L3_spinwave": 'all:magnon AND all:dispersion AND all:"spin wave"',
    "L4_dielectric": 'all:"dielectric function" AND all:"Kramers-Kronig" AND all:silicon',
    "L5_liquid": 'all:"pair correlation function" AND all:liquid AND all:"structure factor"',
    "L6_bloch": 'all:"Bloch equations" AND all:relaxation AND all:visualization',
    "L7_qsphere": 'all:"Q-sphere" AND all:qubit',
    "L8_circuit": 'all:"quantum circuit" AND all:visualization AND all:diagram',
}

def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")

def arxiv_ids(query, n=2):
    q = urllib.parse.quote(query)
    xml = get(f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results={n}&sortBy=relevance")
    return re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", xml)

out = {}
for topic, query in TOPICS.items():
    try:
        ids = arxiv_ids(query, 2)
    except Exception as e:
        out[topic] = {"error": f"api {type(e).__name__}"}
        continue
    papers = []
    for aid in ids[:2]:
        rec = {"id": aid, "captions": [], "figs": []}
        try:
            html = get(f"https://ar5iv.org/abs/{aid}", 45)
            caps = re.findall(r"<figcaption[^>]*>(.*?)</figcaption>", html, re.S)
            for c in caps[:4]:
                txt = re.sub(r"<[^>]+>", "", c)
                txt = re.sub(r"\s+", " ", txt).strip()
                rec["captions"].append(txt[:300])
            figs = re.findall(r'<img[^>]+src="([^"]+\.(?:png|jpg|jpeg|svg))"', html)
            rec["figs"] = figs[:6]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}"
        papers.append(rec)
        time.sleep(0.3)
    out[topic] = papers
    print(topic, [p["id"] for p in papers], flush=True)

io.open(r"I:\Fermi\quantum-atlas\docs\fig-survey.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("done")
