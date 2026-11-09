#!/usr/bin/env python3
r"""Perspective schematic: photon-number taxonomy of 'quantum' photonic neural nets.

A single horizontal log axis of photons per detection window separating the
genuine-quantum few-quanta regime (<n> <~ 1) from the classical-wave regime
(10^6-10^7 photons), with the 1 mW / 1550 nm / ~8e6-photons-per-ns operating
point of typical coherent optical neural nets marked deep in the classical zone.

All numbers taken verbatim from papers/prxquantum-hopfield-networks.tex.

STYLE: every piece of TEXT is rendered in BLACK; only background SHADING and
the operating-point MARKERS carry colour (blue = genuine-quantum regime,
orange = classical-wave regime).  Sized as a single-column (SINGLE-like)
figure so the on-page font at \columnwidth matches the other column figures.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import figstyle

# Single-column figure: SINGLE-like width with paper base font so that the
# on-page apparent text (at \columnwidth ~3.375in) matches memristor_main_hysteresis.
figstyle.apply(base_fontsize=18)
# figstyle preamble lacks amssymb (needed for \lesssim); add it.
plt.rcParams["text.latex.preamble"] = (
    plt.rcParams["text.latex.preamble"] + r"\usepackage{amssymb}"
)

# Colour is used ONLY for background shading and markers -- never for text.
C_Q = "tab:blue"     # genuine-quantum few-quanta regime
C_C = "tab:orange"   # classical-wave regime
BLACK = "k"

fig, ax = plt.subplots(figsize=(7.0, 2.85))

xmin, xmax = 1e-1, 1e8
ax.set_xscale("log")
ax.set_xlim(xmin, xmax)
ax.set_ylim(0.0, 1.0)

# ---- shaded regimes (colour allowed) -----------------------------------
# genuine quantum few-quanta zone: <n> <~ 1 photon per mode
ax.axvspan(xmin, 1.0, color=C_Q, alpha=0.16, lw=0, zorder=0)
# classical-wave zone: 10^6-10^7 photons per detection window
ax.axvspan(1e6, 1e7, color=C_C, alpha=0.22, lw=0, zorder=0)

# threshold line at <n> = 1
ax.axvline(1.0, color=figstyle.GREY, lw=1.0, ls="--", zorder=1)

# ---- regime labels (BLACK) ---------------------------------------------
ax.text(0.38, 0.78, r"\textbf{[Q]} genuine",
        ha="center", va="center", color=BLACK)
ax.text(0.38, 0.64, r"quantum",
        ha="center", va="center", color=BLACK)
ax.text(0.38, 0.50, r"$\langle n\rangle\!\lesssim\!1$",
        ha="center", va="center", color=BLACK)

ax.text(3.2e6, 0.78, r"\textbf{[C]} classical",
        ha="center", va="center", color=BLACK)
ax.text(3.2e6, 0.64, r"wave optics",
        ha="center", va="center", color=BLACK)
ax.text(3.2e6, 0.50, r"$10^{6}\!-\!10^{7}$",
        ha="center", va="center", color=BLACK)

ax.text(1.25, 0.40, r"$\langle n\rangle=1$",
        ha="left", va="center", color=BLACK,
        fontsize=0.78 * plt.rcParams["font.size"])

# ---- operating points (coloured markers, BLACK text) -------------------
# single-photon memristor (companion paper) in the quantum zone
x_mem = 0.5
ax.plot([x_mem], [0.30], marker="o", ms=11, color=C_Q, zorder=5,
        markeredgecolor="k", markeredgewidth=0.8)
ax.annotate("single-photon\nmemristor",
            xy=(x_mem, 0.30), xytext=(0.30, 0.075),
            ha="center", va="center", color=BLACK,
            fontsize=0.78 * plt.rcParams["font.size"],
            arrowprops=dict(arrowstyle="-", lw=1.0, color=BLACK))

# typical 'quantum' optical neural net: 1 mW, 1550 nm -> ~8e6 photons / ns
x_op = 8e6
ax.plot([x_op], [0.30], marker="D", ms=11, color=C_C, zorder=5,
        markeredgecolor="k", markeredgewidth=0.8)
ax.annotate(r"typical `quantum'" + "\n" + r"optical neural net" + "\n"
            + r"$P=1$\,mW, $\lambda=1550$\,nm" + "\n"
            + r"$\sim\!8\times10^{6}$ photons/ns",
            xy=(x_op, 0.30), xytext=(2.0e3, 0.30),
            ha="center", va="center", color=BLACK,
            fontsize=0.78 * plt.rcParams["font.size"],
            arrowprops=dict(arrowstyle="->", lw=1.1, color=BLACK))

# ---- "deep in classical regime" sweep arrow ----------------------------
arr = FancyArrowPatch((1.5, 0.93), (1.2e7, 0.93),
                      arrowstyle="-|>", mutation_scale=12,
                      lw=1.1, color=BLACK, zorder=4)
ax.add_patch(arr)
ax.text(3.0e3, 0.965, r"many photonic `quantum' nets live here",
        ha="center", va="bottom", color=BLACK,
        fontsize=0.78 * plt.rcParams["font.size"])

# ---- axis cosmetics ----------------------------------------------------
ax.set_yticks([])
ax.set_xlabel(r"photons per $\sim$1\,ns detection window")
ax.set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
ax.tick_params(axis="x", which="both", top=False, labelcolor=BLACK)
ax.tick_params(axis="y", which="both", left=False, right=False)
for s in ("left", "right"):
    ax.spines[s].set_visible(False)
ax.spines["top"].set_visible(False)

fig.tight_layout(pad=0.3)
out = "figures/persp_photon_regimes"
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=150)
print("saved", out + ".pdf")
