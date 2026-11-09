r"""
Experimental setup diagram (Fig. 3 of Part I) for the quantum memristor paper.

Matplotlib schematic with clean geometric shapes. Output: figures/memristor_setup.pdf

Layout/style notes
------------------
* Native size = printed size: the figure is included at 0.86\columnwidth (0.86 x 246 pt = 2.93 in), so the canvas
  is exactly 2.93 in wide and all fonts are their final sizes (component labels 7 pt, secondary labels 6 pt).
* The axes fill the canvas with equal aspect; the drawing coordinates (arbitrary units) are mapped to inches by the
  x-limits, and the canvas height follows from the y-limits, so nothing is cropped or rescaled on save.
* Muted fills derived from the Turbo palette used in the plots (blue end for optics, red end for the memory
  element and the feedback).
No data: pure drawing.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle_pra as fs

fs.apply()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

F_MAIN = 7.0                     # box / component labels
F_SUB = 6.0                      # secondary captions (Type-II ..., herald, update)
F_MATH = 7.0                     # in-line math (theta, |H>, ...)
F_TITLE = 7.0                    # region title

# ── Canvas: final width, height from the drawing limits ────────────────
XLIM = (-0.35, 13.05)
YLIM = (-2.60, 2.92)
FIG_W = 0.86 * fs.COLW
UNIT = FIG_W / (XLIM[1] - XLIM[0])                     # inches per drawing unit
FIG_H = UNIT * (YLIM[1] - YLIM[0])
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_aspect("equal")
ax.axis("off")

# ── Colours: muted tints of the Turbo ends used in the plots ───────────
BLUE, RED = fs.C_BLUE, fs.C_RED
c_source = fs.tint(BLUE, 0.84)
c_pbs = fs.tint(BLUE, 0.76)
c_hwp = fs.tint(RED, 0.80)
c_det = "0.88"
c_fpga = "0.92"
c_beam = "0.10"
c_fb = RED
c_ctrl = "0.35"
c_region = fs.tint(BLUE, 0.93)
c_region_ec = fs.shade(BLUE, 0.15)
LW_BOX = 0.6
LW_BEAM = 0.8


# ── Helper functions ───────────────────────────────────────────────────
def box(x, y, w, h, label, color, fontsize=F_MAIN):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08", linewidth=LW_BOX,
                                edgecolor="black", facecolor=color))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize)
    return x, y


def circle_node(x, y, r, label, color, fontsize=F_MAIN):
    ax.add_patch(plt.Circle((x, y), r, linewidth=LW_BOX, edgecolor="black", facecolor=color))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize)
    return x, y


def diamond(x, y, size, label, color, fontsize=F_MAIN):
    s = size / 2
    verts = [(x, y + s), (x + s, y), (x, y - s), (x - s, y), (x, y + s)]
    ax.add_patch(plt.Polygon(verts, closed=True, linewidth=LW_BOX, edgecolor="black", facecolor=color))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize)
    return x, y


def arrow(x1, y1, x2, y2, color=c_beam, lw=LW_BEAM, style="-"):
    """Straight arrow (open head)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.40", color=color, lw=lw,
                                linestyle=style, shrinkA=0, shrinkB=0))


# ── Background region: memristor core (title inside, bottom right) ─────
region = FancyBboxPatch((6.0, -1.05), 6.45, 3.63, boxstyle="round,pad=0.15", linewidth=0.7,
                        edgecolor=c_region_ec, facecolor=c_region, linestyle="--", zorder=0)
ax.add_patch(region)
ax.text(12.42, -1.17, "Memristor core", ha="right", va="bottom", fontsize=F_TITLE, fontstyle="italic")

# ── Source stage ───────────────────────────────────────────────────────
box(0.55, 0.5, 1.5, 0.9, "Pump", c_source)
box(2.85, 0.5, 1.4, 0.9, "BBO", c_source)
ax.text(2.85, -0.10, "Type-II\nSPDC", ha="center", va="top", fontsize=F_SUB, linespacing=1.0)

diamond(5.0, 0.5, 1.45, "PBS", c_pbs, fontsize=F_SUB)
YDET = 2.20                                            # detectors above the PBSs
box(5.0, YDET, 1.0, 0.7, r"D$_\mathrm{H}$", c_det)

# ── Memristor core ─────────────────────────────────────────────────────
R_HWP = 0.74
circle_node(7.5, 0.5, R_HWP, "HWP", c_hwp, fontsize=F_SUB)
ax.text(7.5, 0.5 + R_HWP + 0.08, r"$\theta(t)$", ha="center", va="bottom", fontsize=F_MATH)

diamond(9.8, 0.5, 1.45, "PBS", c_pbs, fontsize=F_SUB)
box(9.8, YDET, 1.0, 0.7, r"D$_1$", c_det)
box(11.9, 0.5, 1.0, 0.7, r"D$_2$", c_det)

# ── FPGA ───────────────────────────────────────────────────────────────
YF = -2.00
box(8.6, YF, 4.3, 0.92, "FPGA + Memory", c_fpga)

# ── Beam arrows (source) ───────────────────────────────────────────────
y_det_bot = YDET - 0.35 - 0.10
arrow(1.38, 0.5, 2.07, 0.5)                            # Pump -> BBO
arrow(3.63, 0.5, 4.22, 0.5)                            # BBO -> PBS1
arrow(5.0, 1.27, 5.0, y_det_bot)                       # PBS1 -> D_H (up)
ax.text(4.74, 1.50, "herald", ha="right", va="center", fontsize=F_SUB)

# ── Beam arrows (memristor core) ───────────────────────────────────────
arrow(5.78, 0.5, 7.5 - R_HWP - 0.04, 0.5)              # PBS1 -> HWP
arrow(7.5 + R_HWP + 0.02, 0.5, 9.02, 0.5)              # HWP -> PBS2
arrow(9.8, 1.27, 9.8, y_det_bot)                       # PBS2 -> D1 (up)
ax.text(10.0, 1.47, r"$|H\rangle$", ha="left", va="center", fontsize=F_MATH)   # upper output -> D1
arrow(10.58, 0.5, 11.32, 0.5)                          # PBS2 -> D2 (right)
ax.text(10.95, 0.22, r"$|V\rangle$", ha="center", va="top", fontsize=F_MATH)   # lower output -> D2

# ── Feedback lines: detectors -> rail right of D2 -> FPGA right side ───
rail_x = 12.85
ax.plot([10.38, rail_x, rail_x, 11.15], [YDET, YDET, YF, YF], color=c_ctrl, lw=0.7, linestyle="--",
        dashes=(3, 2), solid_capstyle="round")
ax.plot([rail_x, 12.0], [0.5, 0.5], color=c_ctrl, lw=0.7, linestyle="--", dashes=(3, 2), zorder=0.8)
arrow(11.15, YF, 10.84, YF, color=c_ctrl, lw=0.7, style="-")

# FPGA -> HWP (feedback, red)
arrow(7.5, YF + 0.46 + 0.08, 7.5, 0.5 - R_HWP - 0.03, color=c_fb, lw=0.9)
ax.text(7.80, -0.44, r"update $\theta$", ha="left", va="center", fontsize=F_SUB)

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "memristor_setup.pdf")
fig.savefig(out_path)
print(f"Saved: {out_path}  ({FIG_W:.3f} x {FIG_H:.3f} in)")
plt.close(fig)
