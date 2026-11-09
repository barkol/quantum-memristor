"""Panel tags (a), (b), ... with row support. Copy of part1/nz/paneltags.py extended for multi-row figures.

Rules as in the original: within one row all tags sit at one height (above the tallest element of any panel in that
row, including titles and legends); the left edge of each tag is the left edge of its panel's full bounding box
(with axis labels). Extensions: tag(ax, label, row=k, title=None, x0=None) groups panels by row; an optional title
is set after the bold tag in the same line ("(a) criterion"); x0 (figure fraction) overrides the left edge, e.g.
for schematic panels drawn on an invisible axis. place(fig) must be called after all drawing and before savefig;
the figure must use a fixed layout (no layout engine), and the caller leaves room above each row.
"""
_REG = {}


def tag(ax, label, row=0, title=None, x0=None):
    _REG.setdefault(id(ax.figure), []).append((ax, label, row, title, x0))


def place(fig, pad_pt=2.0, fontsize=8, title_fontsize=7, ytop=None):
    """ytop: optional dict row -> figure-fraction baseline, overriding the automatic height."""
    items = _REG.pop(id(fig), [])
    if not items:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * fig.dpi
    Hin = fig.get_size_inches()[1]
    bbs = [ax.get_tightbbox(r) for ax, *_ in items]
    rows = {}
    for (ax, lab, row, title, x0), b in zip(items, bbs):
        rows.setdefault(row, []).append(b)
    yrow = {row: max(b.y1 for b in bl) / H + pad_pt / 72.0 / Hin for row, bl in rows.items()}
    if ytop:
        yrow.update(ytop)
    out = []
    for (ax, lab, row, title, x0), b in zip(items, bbs):
        x = b.x0 / W if x0 is None else x0
        s = r"\textbf{" + lab + "}"
        t = fig.text(x, yrow[row], s, ha="left", va="baseline", fontsize=fontsize)
        if title:
            fig.canvas.draw()
            bt = t.get_window_extent(r)
            fig.text(bt.x1 / W + 2.5 / 72.0 / fig.get_size_inches()[0], yrow[row], title,
                     ha="left", va="baseline", fontsize=title_fontsize)
        out.append(t)
    return out
