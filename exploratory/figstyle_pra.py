"""PRA two-column style for Part I figures, as in part1/nz/make_fig_nz.py: usetex serif, ticks 6 pt, labels 7 pt,
panel tags 8 pt (paneltags_rows.py), Turbo palette without its very light ends. Figures are saved at their final
size (COLW = \\columnwidth = 246 pt), so these sizes are the sizes on the page."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLW = 246.0 / 72.27          # \columnwidth in inches (3.404)

RC = {"text.usetex": True, "font.family": "serif", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
      "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5,
      "lines.linewidth": 0.9, "lines.markersize": 2.8, "legend.frameon": False,
      "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
      "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
      "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.0,
      "text.latex.preamble": r"\usepackage{amsmath}", "savefig.bbox": None, "savefig.dpi": 300,
      "pdf.fonttype": 42}


def apply():
    plt.rcParams.update(RC)


def turbo(x):
    return plt.cm.turbo(x)


# plot colours: two-series pair and a four-series set (light ends of Turbo avoided)
C_BLUE = turbo(0.12)          # dark blue
C_RED = turbo(0.88)           # dark red-orange
C_GREEN = turbo(0.62)         # yellow-green
C4 = turbo(np.linspace(0.1, 0.9, 4))


def tint(c, f):
    """Blend colour c with white: f=0 -> c, f=1 -> white (muted schematic fills)."""
    from matplotlib.colors import to_rgb
    r = np.array(to_rgb(c))
    return tuple(r + (1 - r) * f)


def shade(c, f):
    """Blend colour c with black (muted schematic edges)."""
    from matplotlib.colors import to_rgb
    return tuple(np.array(to_rgb(c)) * (1 - f))
