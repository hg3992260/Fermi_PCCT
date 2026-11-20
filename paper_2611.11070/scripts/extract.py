import pymupdf, json, os, sys

PDF = r'I:\Fermi\2611.11070v1.pdf'
OUT = r'I:\Fermi\paper_2611.11070'
os.makedirs(OUT, exist_ok=True)

doc = pymupdf.open(PDF)
print('pages:', doc.page_count)

lines = []
for i, pg in enumerate(doc):
    txt = pg.get_text('text')
    lines.append('=' * 20 + f' PAGE {i+1} ' + '=' * 20)
    lines.append(txt)
    d = pg.get_drawings()
    imgs = pg.get_images(full=True)
    print(f'page {i+1}: chars={len(txt)} drawings={len(d)} images={len(imgs)}', flush=True)

with open(os.path.join(OUT, 'raw_text.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# per-page text files
for i, pg in enumerate(doc):
    with open(os.path.join(OUT, f'page_{i+1:02d}.txt'), 'w', encoding='utf-8') as f:
        f.write(pg.get_text('text'))

print('done')
