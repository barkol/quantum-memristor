#!/usr/bin/env python3
r"""Figure template / methodology -- single source of truth for figure SHAPE,
in-figure FONT SIZE, and PANEL-LABEL size/position.

Derived from the manuscript's APPROVED ("best") figures so that every figure
renders at a uniform ~9 pt apparent (on-page) font and fills its column. Use
this for ALL manuscript figures (and reuse the module across papers) instead of
hand-picking figsizes/fonts per figure.

Two placement classes (PRX Quantum / revtex4-2 two-column):
  * COLUMN     figure   -> displayed at \columnwidth  ~ 3.375 in
  * FULLWIDTH  figure*  -> displayed at \textwidth     ~ 6.9 in

Apparent-font rule:  apparent_pt = base_font * (display_width / figsize_width)
  COLUMN    : figsize width 7.0 in, base 18  ->  18*3.375/7.0 = 8.7 pt
  FULLWIDTH : figsize width 10.0 in, base 13 ->  13*6.9/10.0  = 9.0 pt
Both land at ~9 pt on the page -> uniform apparent font across the whole paper.

Panel labels: bold "(a)" placed OUTSIDE the top-left corner of the axes (the
APS/PRX convention), via figstyle.panel (x=-0.02, y=1.02, ha='right',
va='bottom', fontsize = axes.titlesize).

Usage:
    import figtemplate as ft
    fig, ax  = ft.column_figure()             # single-panel column figure
    fig, axs = ft.fullwidth_figure(1, 2)      # 1x2 full-width (figure*)
    ft.panel(axs[0], "(a)")                    # uniform panel marker
    leg = ft.legend(ax, loc="upper left")      # opaque, occludes data cleanly
    ft.save(fig, ".../figures/foo.pdf")        # tight PDF + 150-dpi PNG preview

In the .tex, COLUMN figures go in `figure` with width=\columnwidth; FULLWIDTH
figures go in `figure*` with width=\textwidth.
"""
import figstyle
import matplotlib.pyplot as plt

# ---- canonical on-page widths (revtex4-2 two-column) -----------------------
COLUMN_DISPLAY_IN    = 3.375    # \columnwidth
FULLWIDTH_DISPLAY_IN = 6.9      # \textwidth (spans both columns)

# ---- canonical generation parameters (reproduce the approved figures) ------
COLUMN_W,    COLUMN_BASE    = 7.0, 18    # column figs   -> ~8.7 pt apparent
FULLWIDTH_W, FULLWIDTH_BASE = 10.0, 13   # figure* figs  -> ~9.0 pt apparent

# default per-row heights that fill the width with a sensible aspect
COLUMN_ROW_H    = 4.3    # single column panel ~ 7.0 x 4.3
FULLWIDTH_ROW_H = 4.2    # full-width panel row ~ 10.0 x 4.2 (2x2 -> 10 x 8.4)

# re-export the house palette so generators import one module
BLUE, RED, GREY = figstyle.BLUE, figstyle.RED, figstyle.GREY
THIRD = getattr(figstyle, "THIRD", "0.35")


def column_figure(nrows=1, ncols=1, height=None, **kw):
    """COLUMN figure (display ~3.375 in), apparent font ~8.7 pt. Fills the
    column; pass `height` for a custom aspect."""
    figstyle.apply(base_fontsize=COLUMN_BASE)
    if height is None:
        height = COLUMN_ROW_H * nrows
    return plt.subplots(nrows, ncols, figsize=(COLUMN_W, height), **kw)


def fullwidth_figure(nrows=1, ncols=2, height=None, **kw):
    """FULL-WIDTH figure* (display ~6.9 in), apparent font ~9.0 pt. Use for
    multi-panel figures; pass `height` for a custom aspect."""
    figstyle.apply(base_fontsize=FULLWIDTH_BASE)
    if height is None:
        height = FULLWIDTH_ROW_H * nrows
    return plt.subplots(nrows, ncols, figsize=(FULLWIDTH_W, height), **kw)


def figure(placement="column", **kw):
    """Dispatch: placement='column' or 'fullwidth'."""
    return (fullwidth_figure if placement == "fullwidth" else column_figure)(**kw)


def panel(ax, label, **kw):
    """Uniform panel marker '(a)' bold, OUTSIDE top-left of the axes."""
    return figstyle.panel(ax, label, **kw)


def legend(ax, loc="best", opaque=True, **kw):
    """Standard legend: opaque white box (so it cleanly occludes any data it
    overlaps) at the house legend font size. Set opaque=False for frameless."""
    if opaque:
        kw.setdefault("framealpha", 1.0)
        kw.setdefault("facecolor", "white")
        kw.setdefault("edgecolor", "0.7")
    else:
        kw.setdefault("frameon", False)
    kw.setdefault("fontsize", 0.78 * plt.rcParams["font.size"])
    return ax.legend(loc=loc, **kw)


def save(fig, path, pad=0.3, **kw):
    """tight_layout + save PDF and a 150-dpi PNG preview next to it."""
    try:
        fig.tight_layout(pad=pad)
    except Exception:
        pass
    fig.savefig(path, **kw)
    fig.savefig(path[:-4] + ".png" if path.endswith(".pdf") else path + ".png",
                dpi=150, **kw)
    return path
