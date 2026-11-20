import os, re, json

base = r'I:\Fermi\paper_2611.11070'
doc = open(os.path.join(base, 'paper.md'), encoding='utf-8').read()

anchors = [(m.start(), m.group(1)) for m in re.finditer(r'<a id="(S\d{3}|F\d{3}|T\d{3})"></a>', doc)]

blocks = []
for k, (pos, aid) in enumerate(anchors):
    end = anchors[k + 1][0] if k + 1 < len(anchors) else len(doc)
    seg = doc[pos:end]

    src = re.search(r'\*\*Source:\*\* (p\.[\d–\-]+) (C?\w+)', seg)
    page = src.group(1) if src else None
    sid = src.group(2) if src else None

    head = re.search(r'^### (.+)$', seg, re.M)
    heading = head.group(1) if head else None

    placed = re.search(r'\*\*Placed near:\*\* ([^\n]+)', seg)
    img = re.search(r'!\[[^\]]*\]\(([^)]+)\)', seg)

    kind = 'body' if aid.startswith('S') else ('figure' if aid.startswith('F') else 'table')

    blocks.append({
        'id': aid,
        'kind': kind,
        'source_page': page,
        'source_id': sid,
        'heading_zh': heading,
        'placed_near': placed.group(1) if placed else None,
        'asset': img.group(1) if img else None,
        'has_original': '**Original:**' in seg,
        'has_translation_zh': ('**中文：**' in seg) or ('**中文图注：**' in seg) or ('**中文表题：**' in seg),
        'chars_original': len(seg),
    })

source_map = {
    'paper': {
        'title': 'Learning Electronic Band Structures Across the s, p, and d Blocks with an Empirical Pseudopotential Network',
        'title_zh': '用经验赝势网络学习跨越 s、p、d 区的电子能带结构',
        'author': 'Qiang Zhu (University of North Carolina at Charlotte)',
        'identifier': 'arXiv:2611.11070v1 [cond-mat.mtrl-sci], 8 Nov 2026',
        'source_file': '2611.11070v1.pdf',
        'pages_total': 30,
        'coverage': {
            'main_text': 'p.1–14 (fully bilingual)',
            'references': 'p.14–20 (listed by number only, not translated)',
            'supplementary_information': 'p.21–30 (fully bilingual, tables transcribed)',
        },
    },
    'counts': {
        'body_blocks': sum(1 for b in blocks if b['kind'] == 'body'),
        'figure_blocks': sum(1 for b in blocks if b['kind'] == 'figure'),
        'table_blocks': sum(1 for b in blocks if b['kind'] == 'table'),
        'blocks_with_translation': sum(1 for b in blocks if b['has_translation_zh']),
    },
    'blocks': blocks,
}

with open(os.path.join(base, 'source_map.json'), 'w', encoding='utf-8') as f:
    json.dump(source_map, f, ensure_ascii=False, indent=2)

print(json.dumps(source_map['counts'], ensure_ascii=False))
missing = [b['id'] for b in blocks if not b['has_translation_zh']]
print('missing zh:', missing)
print('json ok')
