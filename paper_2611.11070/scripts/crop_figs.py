import pymupdf

doc = pymupdf.open(r'I:\Fermi\2611.11070v1.pdf')
OUT = r'I:\Fermi\paper_2611.11070\assets'

# (page, name, handle, extra_top)
jobs = [
    (5,  'fig1',  'Figure 1.', 6),
    (6,  'fig2',  'Figure 2.', 6),
    (7,  'fig3',  'Figure 3.', 6),
    (8,  'fig4',  'Figure 4.', 6),
    (10, 'fig5',  'Figure 5.', 6),
    (21, 'figS1', 'Fig. S1.', 8),
    (23, 'figS2', 'Fig. S2.', 6),
    (25, 'figS3', 'Fig. S3.', 6),
    (28, 'figS4', 'Fig. S4.', 8),
    (29, 'figS5', 'Fig. S5.', 8),
    (30, 'figS6', 'Fig. S6.', 6),
]

for pno, name, needle, margin in jobs:
    pg = doc[pno-1]
    hits = pg.search_for(needle)
    assert hits, (pno, needle)
    cap = hits[0]
    rects = [d['rect'] for d in pg.get_drawings()]
    rects += [pg.get_image_bbox(i) for i in pg.get_images(full=True)]
    rects = [r for r in rects if r.width > 2 and r.height > 2]
    u = rects[0]
    for r in rects[1:]:
        u |= r
    clip = pymupdf.Rect(u.x0 - margin - 12, u.y0 - margin, u.x1 + margin, min(u.y1 + margin, cap.y0 - 2))
    # limit to text area
    clip = pymupdf.Rect(max(clip.x0, 40), max(clip.y0, 40), min(clip.x1, 560), min(clip.y1, 800))
    pix = pg.get_pixmap(clip=clip, dpi=300)
    path = f'{OUT}/{name}.png'
    pix.save(path)
    print(name, 'clip', [round(v,1) for v in clip], 'size', pix.width, pix.height)
