#!/usr/bin/env python3
r"""Common publication figure style for the photonic-quantum-memristor papers.

Matches the house style of art-ent-meter-HOM (last figs/data_to_plot/figs.ipynb):
LaTeX-rendered serif text, large fonts, a two-colour blue / tab:red scheme,
thin solid curves for theory with small markers for discrete data, gray shaded
regions with in-axes text labels, panel labels a)/b), no grid, dpi 300.

Usage:
    import figstyle
    figstyle.apply()
    ...
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)
    ax.plot(x, y, color=figstyle.BLUE, lw=2, label="theory")
    ax.plot(xd, yd, color=figstyle.RED, ls="", marker="x", ms=4, label="data")
    figstyle.panel(ax, "a)")
    figstyle.shade_below(ax, 0.0, "Separable")
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- HOM colour scheme -------------------------------------------------
BLUE = "b"          # primary series (theory / curve 1)
RED = "tab:red"     # secondary series (curve 2)
GREY = "0.5"        # auxiliary lines (zero axes, baselines)
THIRD = "0.35"      # neutral third series when two colours are not enough
SHADE = "0.85"      # light-grey fill for highlighted regions
SHADE_ALPHA = 1.0

# ---- canonical marker conventions (HOM) --------------------------------
# theory: solid line, no marker.  data/experiment: 'x','+'.  simulation: '1','2'.
M_DATA1, M_DATA2 = "x", "+"
M_SIM1, M_SIM2 = "1", "2"

# ---- figure sizes (inches); fonts sized for ~column-width inclusion ----
SINGLE = (7.0, 5.0)        # one panel
WIDE = (10.0, 4.2)         # two panels side by side
GRID2x2 = (10.0, 8.0)      # 2x2 panel grid
TALL = (7.0, 8.0)          # stacked main + residual


def apply(base_fontsize=18):
    """Install the common rcParams.  Call once at import time of each script.

    base_fontsize sets the body font: 18 for print figures (paper default),
    ~13 for presentation/slide figures (`figstyle.apply(base_fontsize=13)`) so
    that slide figures inherit the same colours/markers/conventions at a size
    appropriate to a smaller figure box.  All other type sizes scale with it.

    text.usetex is on (the HOM style and the rest of these papers use it); if a
    given machine lacks a LaTeX install, set FIGSTYLE_NO_TEX=1 to fall back to
    mathtext without touching the scripts.
    """
    import os
    usetex = os.environ.get("FIGSTYLE_NO_TEX", "0") != "1"
    f = float(base_fontsize)
    plt.rcParams.update({
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{physics}\usepackage{braket}",
        "font.family": "serif",
        "font.style": "normal",
        "font.size": f,
        "axes.labelsize": f,
        "axes.titlesize": f,
        "xtick.labelsize": 0.83 * f,
        "ytick.labelsize": 0.83 * f,
        "legend.fontsize": 0.78 * f,
        # Opaque white legend box by DEFAULT so a legend cleanly occludes any
        # data behind it instead of letting curves bleed through the text.
        # Schematics that want a floating frameless legend pass frameon=False.
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.facecolor": "white",
        "legend.edgecolor": "0.7",
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })
    if not usetex:
        plt.rcParams["mathtext.fontset"] = "cm"


def panel(ax, label, x=-0.02, y=1.02, **kw):
    """PRX-style panel marker, e.g. '(a)', placed OUTSIDE the panel area --
    just above the top-left corner of the axes -- in bold (the APS/PRX Quantum
    convention). Pass the label WITH parentheses, e.g. panel(ax, '(a)').

    The default (x=-0.02, y=1.02, ha='right', va='bottom') sits the marker above
    and just left of the axes box so it never overlaps the data; override x/y for
    axis-free panels (e.g. Bloch spheres: panel(ax, '(a)', x=0.0, y=1.05))."""
    kw.setdefault("fontweight", "bold")
    kw.setdefault("ha", "right")
    kw.setdefault("va", "bottom")
    kw.setdefault("fontsize", mpl.rcParams["axes.titlesize"])
    # usetex ignores fontweight -- bold must be requested in the TeX string
    if mpl.rcParams.get("text.usetex", False) and "\\textbf" not in label:
        label = r"\textbf{" + label + "}"
    ax.text(x, y, label, transform=ax.transAxes, clip_on=False, **kw)


def shade_below(ax, y0, label=None, above_label=None, color=SHADE):
    """Shade the region below y0 (e.g. a 'forbidden'/'separable' zone) and
    optionally annotate the lower (and upper) regions, HOM-style."""
    lo, hi = ax.get_ylim()
    ax.axhspan(lo, y0, color=color, lw=0, zorder=0)
    ax.set_ylim(lo, hi)
    x0, x1 = ax.get_xlim()
    xt = x0 + 0.6 * (x1 - x0)
    if above_label:
        ax.annotate(above_label, xy=(xt, y0 + 0.04 * (hi - lo)), fontsize=12)
    if label:
        ax.annotate(label, xy=(xt, y0 - 0.13 * (hi - lo)), fontsize=12)


def zero_axes(ax, x=True, y=True):
    """Thin grey reference lines through the origin (no dashes-heavy clutter)."""
    if y:
        ax.axhline(0.0, color=GREY, lw=0.8, zorder=0)
    if x:
        ax.axvline(0.0, color=GREY, lw=0.8, zorder=0)


# ── Tick conventions (house rules) ─────────────────────────────────────────
# * 4-5 labels per axis (never a dense ladder of ticks).
# * Strip trailing zeros: "1" not "1.00", "0" not "0.00", "0.5" not "0.50".
# * Axes whose data spans the unit interval use the symmetric set
#   {-1, -0.5, 0, 0.5, 1} so the +/- limits read symmetrically.
import math

from matplotlib.ticker import FuncFormatter, FixedLocator, MaxNLocator


def _strip_zeros(x, _=None):
    s = f"{x:.10g}"            # general format: 1.0->"1", 0.50->"0.5", 0.0->"0"
    return "0" if s == "-0" else s


STRIP_FMT = FuncFormatter(_strip_zeros)


def _round_ticks_in_range(lo, hi, want=5):
    """Round-step ticks (1/2/2.5/4/5 x10^k) lying INSIDE [lo,hi] with NO range
    change. Returns 4-5 ticks or None. Larger steps tried first (fewer ticks)."""
    span = hi - lo
    mag = 10 ** math.floor(math.log10(span / (want - 1)))
    # smallest step first -> the densest layout still within the 4-5 cap, which
    # favours conventional steps (20 for a 0..84 axis, not 25).
    for base in (0.5, 1, 2, 2.5, 4, 5, 10):
        step = base * mag
        first = math.ceil(lo / step - 1e-9) * step
        ticks, t = [], first
        while t <= hi + 1e-9 * step:
            ticks.append(t)
            t += step
        if 4 <= len(ticks) <= 5:
            dec = max(0, -math.floor(math.log10(step)) + 1)
            return [round(v, dec) for v in ticks]
    return None


def nice_axis(lo, hi, want=5, max_ext=0.35):
    """Round-number ticks (steps 1/2/2.5/4/5 x10^k) giving 4-5 labels.

    PREFER ticks that fit INSIDE the current range (limits unchanged) -- so a
    time axis ending at 84 keeps [0,84] with ticks 0..80 rather than padding out
    to 100.  Only when no round step fits inside do we EXTEND the range outward
    to round multiples (up to `max_ext` of the span).

    Returns (new_lo, new_hi, ticks); new_lo/new_hi equal lo/hi when not extended.
    Returns None if even extension cannot land a round step in [4,5]."""
    if not (hi > lo):
        return None
    inside = _round_ticks_in_range(lo, hi, want)
    if inside:
        return (lo, hi, inside)
    span = hi - lo
    mag = 10 ** math.floor(math.log10(span / (want - 1)))
    cands = []
    for base in (0.5, 1, 2, 2.5, 4, 5, 10):  # round-number steps (x mag)
        step = base * mag
        slo = math.floor(lo / step + 1e-9) * step
        shi = math.ceil(hi / step - 1e-9) * step
        n = int(round((shi - slo) / step)) + 1
        if 4 <= n <= 5:
            cands.append(((shi - slo) - span, slo, shi, step, n))
    cands = [c for c in cands if c[0] <= max_ext * span]
    if not cands:
        return None
    ext, slo, shi, step, n = min(cands, key=lambda c: (c[0], abs(c[4] - want)))
    dec = max(0, -math.floor(math.log10(step)) + 1)
    return (round(slo, dec), round(shi, dec),
            [round(slo + i * step, dec) for i in range(n)])


def _odd_fallback(lo, hi, want=5):
    """Last resort when round steps need too much range extension: allow
    1.5x/3x steps within the EXISTING range (no extension)."""
    span = hi - lo
    mag = 10 ** math.floor(math.log10(span / (want - 1)))
    best = None
    for mult in (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        step = mult * mag
        first = math.ceil(lo / step - 1e-9) * step
        ticks, t = [], first
        while t <= hi + 1e-9 * step:
            ticks.append(t)
            t += step
        n = len(ticks)
        dec = max(0, -math.floor(math.log10(step)) + 1)
        rticks = [round(v, dec) for v in ticks]
        if 4 <= n <= 5:
            return rticks
        if best is None or abs(n - want) < best[0]:
            best = (abs(n - want), rticks)
    return best[1] if best else None


def _apply_nice(ax, name):
    lo0, hi0 = getattr(ax, f"get_{name}lim")()
    inverted = lo0 > hi0
    lo, hi = min(lo0, hi0), max(lo0, hi0)
    axis = getattr(ax, f"{name}axis")
    res = nice_axis(lo, hi)
    if res:
        slo, shi, ticks = res
    else:
        slo, shi, ticks = lo, hi, _odd_fallback(lo, hi)
    if not ticks:
        return
    # y-axis: keep the highest tick clear of the panel marker '(a)' sitting just
    # above the top-left corner -- if the top tick crowds the axis top, lift the
    # upper limit a touch so the topmost label has breathing room.
    if name == "y":
        rng = shi - slo
        if rng > 0 and (max(ticks) - slo) / rng > 0.90:
            shi = slo + (max(ticks) - slo) / 0.86
    getattr(ax, f"set_{name}lim")((shi, slo) if inverted else (slo, shi))
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(STRIP_FMT)


def tidy_ticks(ax, axis="both"):
    """4-5 round-number labels per axis with trailing zeros stripped (extending
    the range outward if needed). Use for non-unit axes (time, angle, ...)."""
    for name in {"x": ["x"], "y": ["y"]}.get(axis, ["x", "y"]):
        _apply_nice(ax, name)


def cap_ticks(fig):
    """Enforce the house rule of 4-5 labels per axis on EVERY axis of `fig`.

    Applied at save time so all figures comply by construction.  Skips:
      * log / symlog axes (their LogLocator decades are left alone), and
      * axes already given an explicit FixedLocator (unit_axis / set_ticks),
    so deliberately-chosen tick sets are preserved.  On the remaining linear
    axes it installs 4-5 ROUND ticks (extending the range outward if needed)
    with trailing zeros stripped."""
    for ax in fig.axes:
        for name in ("x", "y"):
            if getattr(ax, f"get_{name}scale")() != "linear":
                continue
            if isinstance(getattr(ax, f"{name}axis").get_major_locator(), FixedLocator):
                continue
            _apply_nice(ax, name)


def unit_axis(ax, axis="y", lim=1.15):
    """Symmetric unit axis: limits +/-lim, the 5-label set {-1,-0.5,0,0.5,1},
    trailing zeros stripped.  Use for axes whose data spans the unit interval."""
    ticks = [-1, -0.5, 0, 0.5, 1]
    if axis in ("x", "both"):
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(STRIP_FMT)
    if axis in ("y", "both"):
        ax.set_ylim(-lim, lim)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(STRIP_FMT)
