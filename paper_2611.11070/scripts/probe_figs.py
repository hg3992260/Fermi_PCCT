import pymupdf, json

doc = pymupdf.open(r'I:\Fermi\2611.11070v1.pdf')

def caption_rect(pg, needle):
    r = pg.search_for(needle)
    return r

for pno, needle in [(5,'Figure 1.'),(6,'Figure 2.'),(7,'Figure 3.'),(8,'Figure 4.'),(10,'Figure 5.'),
                    (21,'Fig. S1.'),(23,'Fig. S2.'),(25,'Fig. S3.'),(28,'Fig. S4.'),(29,'Fig. S5.'),(30,'Fig. S6.')]:
    pg = doc[pno-1]
    hits = pg.search_for(needle)
    draws = [d['rect'] for d in pg.get_drawings()]
    imgs = [pg.get_image_bbox(i) for i in pg.get_images(full=True)]
    allr = draws + [r for r in imgs if r.width > 1]
    if allr:
        u = allr[0]
        for r in allr[1:]:
            u |= r
        gfx = u
    else:
        gfx = None
    print(pno, needle, 'hits', hits[:2], 'gfx', gfx, 'n_draw', len(draws), 'n_img', len(imgs))
