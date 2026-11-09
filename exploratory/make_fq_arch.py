#!/usr/bin/env python3
r"""Publication SCHEMATIC of the coherent delay-loop extension (Fig. 5 of Part I). Static concept diagram:

  synchronized photon train -> 3-qubit programmable controlled-phase gate with TWO coherent delay loops
  (target -> program -> control, NO measurement inside the loops) -> output photons that, entangled at the first
  joint application, are fed to a temporal-steering test in random MUBs.

  Quantum path: S_N > sqrt(N)  (one-sided DI certificate, up to N).
  Loop decoherence (strong loss) -> classical-feedback limit  S_N <= sqrt(N).

Single-column figure drawn at its final size (\columnwidth = 3.40 in) in inch coordinates, so fonts are final:
7 pt for the main labels, 6 pt for secondary ones. Port labels sit inside the gate box next to their ports.
Muted colours derived from the Turbo palette of the plots. No data: pure drawing.
Output: figures/paper_fq_architecture.pdf (and papers/figures/ copy, where the old script wrote it).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle_pra as fs

fs.apply()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

F7, F6 = 7.0, 6.0
BLUE, RED = fs.C_BLUE, fs.C_RED
GREEN = fs.shade(fs.turbo(0.58), 0.40)          # quantum-certificate accent
GOLD = fs.shade(fs.turbo(0.74), 0.15)           # decoherence / classical limit
ENT = fs.shade(fs.turbo(0.30), 0.25)            # entanglement link
GREYL = "0.4"

# ---------------------------------------------------------------- vertical layout (inches, bottom up)
FW = fs.COLW
VB0, VB1 = 0.05, 0.49                           # verdict boxes
YLAB = 0.585                                    # labels above the verdict boxes
CYB, CYT = 0.70, 1.40                           # steering box
yout = 1.89                                     # output photons k, k+1
y_kh = yout + 0.30                              # horizontal run of the photon-k flow line
GY0 = y_kh + 0.27                               # gate box
yC = GY0 + 0.15
yP = yC + 0.25
yT = yP + 0.25
GY1 = yT + 0.40
APX_T, APX_B = GY1 + 0.17, GY0 - 0.17           # loop apexes
Y_SW = APX_T + 0.10                             # pair-run switch line
FH = Y_SW + 0.06

GX0, GX1 = 1.50, 2.95                           # gate box
XL, XR = 1.20, 3.10                             # loop routing lanes
XS = 0.82                                       # input switch
XK, XK1 = 2.45, 3.05                            # output photons
XFK, XFK1 = 3.20, 3.30                          # output flow lanes (k, k+1)

fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.set_aspect("equal")
ax.axis("off")

AH = "-|>,head_width=0.16,head_length=0.30"     # filled arrow head (mutation scale = 10)


def arrow(xy0, xy1, color=GREYL, lw=0.7, ls="-"):
    ax.annotate("", xy=xy1, xytext=xy0, arrowprops=dict(arrowstyle=AH, color=color, lw=lw, linestyle=ls,
                                                        shrinkA=0, shrinkB=0, mutation_scale=10))


def photon(x, y, ms=5.0, color=BLUE, z=6):
    ax.plot([x], [y], "o", color=color, ms=ms, zorder=z, mec="none")
    ax.plot([x], [y], "o", color="white", ms=ms * 0.34, zorder=z + 1)


# ---------------------------------------------------------------- delay loops
# Each loop leaves the gate on the RIGHT, routes around the outside of the box and re-enters on the LEFT one row
# down. Top loop: target-out -> program-in. Bottom loop: program-out -> control-in. One-step delay each.
for y_out, y_in, y_apex, ytxt, vat in ((yT, yP, APX_T, GY1 + 0.035, "bottom"), (yP, yC, APX_B, GY0 - 0.035, "top")):
    P = np.array([(GX1 + 0.05, y_out), (XR, y_out), (XR, y_apex), (XL, y_apex), (XL, y_in), (GX0 - 0.05, y_in)])
    ax.plot(P[:, 0], P[:, 1], ls=(0, (3.5, 2)), color=RED, lw=0.9, zorder=5)
    ya = P[3, 1] + 0.40 * (P[4, 1] - P[3, 1])               # head on the left lane, along the re-entry direction
    dy = 0.01 if P[4, 1] < P[3, 1] else -0.01
    arrow((XL, ya + dy), (XL, ya - dy), color=RED, lw=0.9)
    arrow((XL + 0.32, y_apex), (XL + 0.30, y_apex), color=RED, lw=0.9)   # circulation XR -> XL on the apex
    ax.text((GX0 + GX1) / 2, ytxt, "delay 1 step", ha="center", va=vat, fontsize=F6)

# ---------------------------------------------------------------- gate box
ax.add_patch(FancyBboxPatch((GX0, GY0), GX1 - GX0, GY1 - GY0, boxstyle="round,pad=0.02", ec=BLUE, lw=0.8,
                            fc=fs.tint(BLUE, 0.90), zorder=2))
gcx = (GX0 + GX1) / 2
ax.text(gcx, GY1 - 0.105, "3-qubit programmable", ha="center", va="center", fontsize=F7)
ax.text(gcx, GY1 - 0.235, "controlled-phase gate", ha="center", va="center", fontsize=F7)
for y, lab in ((yT, "target"), (yP, "program"), (yC, "control")):
    ax.plot([GX0], [y], ">", color=BLUE, ms=3.2, zorder=3, mec="none")
    ax.text(GX0 + 0.08, y, lab, ha="left", va="center", fontsize=F6)
    ax.plot([GX1], [y], "o", color=BLUE, ms=2.0, zorder=3, mec="none")

# pair-run switches: input into the program-port delay, program output to the analyzers
ax.plot([XS, XS, XR, XR], [yT, Y_SW, Y_SW, APX_T], color=GREYL, lw=0.6, ls=(0, (3, 2)), zorder=4)
for xs, ys in ((XS, yT), (XR, yP)):
    ax.text(xs, ys, "S", ha="center", va="center", fontsize=F6, zorder=8,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="0.25", lw=0.5))

# ---------------------------------------------------------------- input train
arrow((0.03, yT), (GX0 - 0.04, yT), lw=0.7)
for xq in (0.14, 0.38, 0.62):
    photon(xq, yT)
ax.text(0.03, yT + 0.09, "synchronized\nphoton train", ha="left", va="bottom", fontsize=F7, linespacing=1.0)

# ---------------------------------------------------------------- band 1 -> output pair
# photon k+1 leaves through the switched program exit, photon k through the control exit
ax.plot([XR + 0.07, XFK1, XFK1], [yP, yP, yout], color=GREYL, lw=0.7, zorder=4)
ax.plot([GX1 + 0.05, XFK, XFK, 2.30, 2.30], [yC, yC, y_kh, y_kh, yout], color=GREYL, lw=0.7, zorder=4)
arrow((2.30, yout), (XK - 0.09, yout))
arrow((XFK1, yout), (XK1 + 0.09, yout))

# ---------------------------------------------------------------- "no measurement" badge
ax.add_patch(FancyBboxPatch((0.08, yout - 0.15), 1.66, 0.30, boxstyle="round,pad=0.025", ec=GREEN, lw=0.7,
                            fc=fs.tint(GREEN, 0.90), zorder=2))
ax.text(0.91, yout, "no measurement inside loops\nS: pair-run switches", ha="center", va="center", fontsize=F7,
        linespacing=1.1)

# ---------------------------------------------------------------- output pair
photon(XK, yout, ms=5.5)
photon(XK1, yout, ms=5.5)
ax.text(XK, yout - 0.075, r"$k$", ha="center", va="top", fontsize=F7)
ax.text(XK1, yout - 0.075, r"$k{+}1$", ha="center", va="top", fontsize=F7)
sx = np.linspace(XK, XK1, 60)
sy = yout + 0.17 + 0.035 * np.sin((sx - XK) / (XK1 - XK) * 3 * np.pi)
ax.plot(sx, sy, color=ENT, lw=0.9, zorder=4)
ax.plot([XK, XK], [yout + 0.05, sy[0]], color=ENT, lw=0.6, ls=":", zorder=4)
ax.plot([XK1, XK1], [yout + 0.05, sy[-1]], color=ENT, lw=0.6, ls=":", zorder=4)
ax.text(3.34, yout - 0.285, "entangled at the first joint application", ha="right", va="center", fontsize=F7)
arrow((2.75, yout - 0.36), (2.75, CYT + 0.03))

# ---------------------------------------------------------------- steering box
CXL, CXR = 0.08, 3.32
ax.add_patch(FancyBboxPatch((CXL, CYB), CXR - CXL, CYT - CYB, boxstyle="round,pad=0.02", ec="0.25", lw=0.8,
                            fc="#fbfbfb", zorder=2))
tx = 0.98
ax.text(tx, CYT - 0.11, "steering test", ha="center", va="center", fontsize=F7)
ax.text(tx, CYT - 0.25, r"random MUB $n$", ha="center", va="center", fontsize=F7)
ax.text(tx, CYB + 0.24, r"steering parameter $\mathcal{S}_N$", ha="center", va="center", fontsize=F7)
ax.text(tx, CYB + 0.10, r"($N$ mutually unbiased settings)", ha="center", va="center", fontsize=F7)
yd = CYT - 0.24
for cx, lab, deg in ((2.20, r"$\hat\sigma_n^{(k)}$", 35), (2.85, r"$\hat\sigma_{\pi(n)}^{(k{+}1)}$", 125)):
    ax.add_patch(Circle((cx, yd), 0.13, fc="white", ec="0.3", lw=0.7, zorder=3))
    a = np.deg2rad(deg)
    ax.plot([cx, cx + 0.10 * np.cos(a)], [yd, yd + 0.10 * np.sin(a)], color=RED, lw=0.9, zorder=4)
    ax.text(cx, yd - 0.17, lab, ha="center", va="top", fontsize=F7)

# ---------------------------------------------------------------- verdicts
LX0, LX1, RX0, RX1 = 0.08, 1.54, 1.86, 3.32
ax.text((LX0 + LX1) / 2, YLAB, r"quantum bound $\mathcal{S}_N\le N$", ha="center", va="center", fontsize=F7)
ax.text((RX0 + RX1) / 2, YLAB, "loop decoherence (strong loss)", ha="center", va="center", fontsize=F7)
ax.add_patch(FancyBboxPatch((LX0, VB0), LX1 - LX0, VB1 - VB0, boxstyle="round,pad=0.025", ec=GREEN, lw=0.8,
                            fc=fs.tint(GREEN, 0.90), zorder=2))
ax.text((LX0 + LX1) / 2, (VB0 + VB1) / 2, "$\\mathcal{S}_N>\\sqrt{N}$:\nsteerable; no EB\nproduct mixture",
        ha="center", va="center", fontsize=F7, linespacing=1.15)
arrow((LX1 + 0.05, (VB0 + VB1) / 2), (RX0 - 0.05, (VB0 + VB1) / 2), color=GOLD, lw=0.9, ls=(0, (3, 1.5)))
ax.add_patch(FancyBboxPatch((RX0, VB0), RX1 - RX0, VB1 - VB0, boxstyle="round,pad=0.025", ec=GOLD, lw=0.8,
                            fc=fs.tint(GOLD, 0.90), zorder=2))
ax.text((RX0 + RX1) / 2, (VB0 + VB1) / 2, "$\\mathcal{S}_N\\le\\sqrt{N}$:\nnothing\ncertified",
        ha="center", va="center", fontsize=F7, linespacing=1.15)

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("figures", os.path.join("papers", "figures")):
    out = os.path.join(repo, d, "paper_fq_architecture")
    fig.savefig(out + ".pdf")
    if d != "figures":
        fig.savefig(out + ".png", dpi=150)
    print("saved", out + ".pdf", f"({FW:.3f} x {FH:.3f} in)")
