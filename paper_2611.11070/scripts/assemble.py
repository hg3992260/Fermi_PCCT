import io, os, re, json

base = r'I:\Fermi\paper_2611.11070'
parts = ['part1.md', 'part2.md', 'part3.md', 'part4.md', 'part5.md']
texts = [open(os.path.join(base, p), encoding='utf-8').read() for p in parts]
p1, p2, p3, p4, p5 = texts

# --- move Fig S2 block before Table S1 in part4 ---
marker = '\n<a id="F011"></a>'
i = p4.index(marker)
figs2 = p4[i:]
p4 = p4[:i]
j = p4.index('<a id="T001"></a>')
p4 = p4[:j] + figs2.strip('\n') + '\n\n' + p4[j:]

# --- renumber caption source ids ---
mapping = [
    ('p.23 C014', 'p.23 C007'),
    ('p.24 C007', 'p.24 C008'),
    ('p.25 C008', 'p.25 C009'),
    ('p.27 C009', 'p.27 C010'),
    ('p.28 C010', 'p.28 C011'),
    ('p.29 C011', 'p.29 C012'),
    ('p.29 C012', 'p.29 C013'),
    ('p.30 C013', 'p.30 C014'),
]
for k, (old, new) in enumerate(mapping):
    assert p4.count(old) == 1, (old, p4.count(old))
    p4 = p4.replace(old, f'@@TOK{k}@@')
for k, (old, new) in enumerate(mapping):
    p4 = p4.replace(f'@@TOK{k}@@', new)

doc = '\n'.join([p1.rstrip('\n'), p2.strip('\n'), p3.strip('\n'), p4.strip('\n'), p5.strip('\n')]) + '\n'
# collapse the 4-space gap between parts
doc = doc.replace('\n\n\n', '\n\n')

with open(os.path.join(base, 'paper.md'), 'w', encoding='utf-8') as f:
    f.write(doc)

# --- verify ---
n_orig = doc.count('**Original:**')
n_zh = doc.count('**中文:**')
n_zh_cap = doc.count('**中文图注:**')
anchors = re.findall(r'<a id="(S\d{3}|F\d{3}|T\d{3})"></a>', doc)
print('chars', len(doc))
print('Original blocks', n_orig, 'Chinese body', n_zh, 'Chinese captions', n_zh_cap)
print('anchors', len(anchors), 'unique', len(set(anchors)))
missing = [a for a in set(anchors) if doc.count(f'<a id="{a}"></a>') > 1]
print('dup anchors', missing)

# image links exist?
imgs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', doc)
for im in imgs:
    path = os.path.join(base, im.replace('/', os.sep))
    print('img', im, os.path.exists(path), os.path.getsize(path) if os.path.exists(path) else '')

# blocks lacking Chinese
for m in re.finditer(r'<a id="(S\d{3})"></a>\n\*\*Source:\*\* ([^\n]+)\n', doc):
    aid, src = m.group(1), m.group(2)
    start = m.end()
    nxt = doc.find('<a id=', start)
    seg = doc[start:nxt if nxt > 0 else len(doc)]
    if '**中文' not in seg:
        print('NO-ZH', aid, src, repr(seg[:80]))
