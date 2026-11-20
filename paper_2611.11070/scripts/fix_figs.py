import os, re

p = r'I:\Fermi\paper_2611.11070\paper.md'
s = open(p, encoding='utf-8').read()

# 1) renumber figure anchors into reading order
order = [('F011', 'F007'), ('F007', 'F008'), ('F008', 'F009'), ('F009', 'F010'), ('F010', 'F011')]
for k, (old, new) in enumerate(order):
    s = s.replace(f'<a id="{old}"></a>', f'@@{k}@@')
for k, (old, new) in enumerate(order):
    s = s.replace(f'@@{k}@@', f'<a id="{new}"></a>')

# 2) add Placed near to figure/table blocks missing it
placed = {
    'F006': 'p.21 S101',
    'F007': 'p.22 S108',
    'T001': 'p.22 S106',
    'F008': 'p.24 S110',
    'T002': 'p.25 S112',
    'F009': 'p.28 S118',
    'F010': 'p.28 S119',
    'T003': 'p.29 S120',
    'F011': 'p.29 S121',
}
for aid, where in placed.items():
    a = f'<a id="{aid}"></a>'
    i = s.index(a)
    nxt = s.find('\n', s.find('**Source:**', i))
    seg_end = nxt
    # insert after the Source line if no Placed near follows
    after = s[i + len(a):i + len(a) + 160]
    if 'Placed near' in after:
        print('skip', aid)
        continue
    s = s[:seg_end] + f'\n**Placed near:** {where}' + s[seg_end:]

open(p, 'w', encoding='utf-8').write(s)

# 3) report order
for m in re.finditer(r'<a id="([FT]\d{3})"></a>', s):
    print(m.group(1), s[m.end():m.end() + 120].replace('\n', ' | ')[:110])
