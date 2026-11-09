"""Wspolne znaczniki paneli (a), (b), ... dla rysunkow zlozonych: jedna wysokosc dla wszystkich paneli (nad najwyzszym elementem
dowolnego panelu, wlacznie z tytulami i legendami), lewa krawedz kazdego znacznika = lewa krawedz pelnego obrysu jego panelu
(z podpisem osi). Uzycie: tag(ax, "(a)") przy rysowaniu, place(fig) tuz przed savefig.
Kopia part1/nz/paneltags.py z obsluga kilku wierszy: dla jednego wiersza dziala identycznie jak oryginal; dla kilku wierszy
(constrained layout) rezerwuje pas nad kazdym wierszem i stawia znaczniki jednego wiersza na jednej wysokosci."""
_REG = {}
def tag(ax, label):
    _REG.setdefault(id(ax.figure), []).append((ax, label))
def _row(ax):
    ss = ax.get_subplotspec()
    return ss.rowspan.start if ss is not None else round(ax.get_position().y1, 3)
def place(fig, pad_pt=3.0, fontsize=8):
    items = _REG.pop(id(fig), [])
    if not items: return
    rows = sorted({_row(ax) for ax, _ in items})
    eng = fig.get_layout_engine(); Hin = fig.get_size_inches()[1]
    strip = (pad_pt + fontsize * 1.35) / 72.0          # wysokosc pasa na znacznik (in)
    if len(rows) == 1:                                 # oryginalne zachowanie
        if eng is not None: eng.set(rect=(0, 0, 1, 1 - strip / Hin))   # pasek na znaczniki nad ukladem
        fig.canvas.draw(); r = fig.canvas.get_renderer(); W, H = fig.get_size_inches() * fig.dpi
        bbs = [ax.get_tightbbox(r) for ax, _ in items]
        ytop = max(b.y1 for b in bbs) / H + pad_pt / 72.0 / fig.get_size_inches()[1]
        for (ax, lab), b in zip(items, bbs):
            fig.text(b.x0 / W, ytop, r"\textbf{" + lab + "}", ha="left", va="bottom", fontsize=fontsize)
        return
    # kilka wierszy: pas nad gornym wierszem (rect) i odstep miedzy wierszami (hspace) dobierany iteracyjnie tak,
    # zeby znacznik (wysokosc ~ fontsize + 1 pt) miescil sie miedzy obrysem swojego wiersza a obrysem wiersza wyzej
    tag_h = (fontsize + 1.0) / 72.0 / Hin
    hs = strip / Hin
    for _ in range(8):
        if eng is not None: eng.set(rect=(0, 0, 1, 1 - strip / Hin), hspace=hs)
        fig.canvas.draw(); r = fig.canvas.get_renderer(); W, H = fig.get_size_inches() * fig.dpi
        bbs = [ax.get_tightbbox(r) for ax, _ in items]
        pos, deficit, prev_bottom = [], 0.0, None
        for row in rows:
            idx = [k for k, (ax, _) in enumerate(items) if _row(ax) == row]
            ytop = max(bbs[k].y1 for k in idx) / H + pad_pt / 72.0 / Hin
            pos.append((idx, ytop))
            if prev_bottom is not None: deficit = max(deficit, ytop + tag_h - prev_bottom)
            prev_bottom = min(bbs[k].y0 for k in idx) / H
        if deficit <= 1e-4 or eng is None: break
        hs += 1.2 * deficit
    if deficit > 1e-4: print(f"paneltags: znaczniki wierszy moga dotykac wiersza wyzej (brak {deficit * Hin * 72:.1f} pt)")
    for idx, ytop in pos:
        for k in idx:
            fig.text(bbs[k].x0 / W, ytop, r"\textbf{" + items[k][1] + "}", ha="left", va="bottom", fontsize=fontsize)
