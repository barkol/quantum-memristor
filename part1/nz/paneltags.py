"""Wspolne znaczniki paneli (a), (b), ... dla rysunkow zlozonych: jedna wysokosc dla wszystkich paneli (nad najwyzszym elementem
dowolnego panelu, wlacznie z tytulami i legendami), lewa krawedz kazdego znacznika = lewa krawedz pelnego obrysu jego panelu
(z podpisem osi). Uzycie: tag(ax, "(a)") przy rysowaniu, place(fig) tuz przed savefig."""
_REG = {}
def tag(ax, label):
    _REG.setdefault(id(ax.figure), []).append((ax, label))
def place(fig, pad_pt=3.0, fontsize=8):
    items = _REG.pop(id(fig), [])
    if not items: return
    eng = fig.get_layout_engine(); Hin = fig.get_size_inches()[1]
    if eng is not None: eng.set(rect=(0, 0, 1, 1 - (pad_pt + fontsize * 1.35) / 72.0 / Hin))   # pasek na znaczniki nad ukladem
    fig.canvas.draw(); r = fig.canvas.get_renderer(); W, H = fig.get_size_inches() * fig.dpi
    bbs = [ax.get_tightbbox(r) for ax, _ in items]
    ytop = max(b.y1 for b in bbs) / H + pad_pt / 72.0 / fig.get_size_inches()[1]
    for (ax, lab), b in zip(items, bbs):
        fig.text(b.x0 / W, ytop, r"\textbf{" + lab + "}", ha="left", va="bottom", fontsize=fontsize)
