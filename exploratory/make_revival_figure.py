#!/usr/bin/env python3
r"""Publication-quality STATIC figure: non-Markovian coherence revival of the
photonic quantum memristor's output qubit.

Reuses the VERIFIED computation from code/make_animations.py
(`_memristor_bloch_traj`, modulated=True) VERBATIM -- same H(t), c_ops,
gamma_t, psi(), seeds, qm.solve_theta(n_cycles=2). ONLY the plotting is
changed: instead of a frame-stack animation we capture three Bloch-sphere
snapshots (|r_avg| max -> min -> max over one drive cycle) plus a companion
panel of |r_avg|(t) over the two cycles with the snapshot phases marked.

Run: python3 make_revival_figure.py
"""
import os
import sys

# import the verified code module so we reuse qm, constants, RED, etc.
# (resolve relative to THIS file so the script runs identically in any
# working copy)
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

import numpy as np
import matplotlib

matplotlib.use("Agg")

# House PRINT style. SINGLE-COLUMN figure: native width ~3.4 in included at
# \columnwidth (~3.4 in), i.e. fonts render ~1:1 on the page. base=9.4 ->
# ~9 pt effective after the small tight-bbox inflation from the 3D sphere
# axes (media box a touch wider than figsize), matching the other converted
# column figures (8-9.5 pt on-page target).
import figstyle
figstyle.apply(base_fontsize=9.4)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import qutip as qt

import quantum_memristor_phase_presentation as qm  # the 'qm' of make_animations

# qm calls figstyle.apply() (base 18) at import time -- RE-apply the print
# style so THIS figure's fonts are the intended single-column 9.4 pt base.
figstyle.apply(base_fontsize=9.4)

RED = figstyle.RED
BLUE = figstyle.BLUE
GREY = figstyle.GREY
USETEX = plt.rcParams["text.usetex"]

cols = ["#1f77b4", "#2ca02c", "#e31a8c"]   # blue / green / magenta -- all distinct, and from RED
ORANGE = "#ff7f0e"
per = 2 * np.pi / qm.OMEGA

# ======================================================================
#  DATA CACHE: the expensive QuTiP trajectories run only when the cache
#  is missing; the plotting below uses only the cached arrays.
# ======================================================================
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / "figures" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "make_revival_figure.npz"

if CACHE.exists():
    d = np.load(CACHE, allow_pickle=False)
    t = d["t"]
    trajs = d["trajs"]
    ax_, ay_, az_ = d["ax_"], d["ay_"], d["az_"]
    rlen = d["rlen"]
    arc_x, arc_z = d["arc_x"], d["arc_z"]
    st_x, st_z = d["st_x"], d["st_z"]
else:
    # ==================================================================
    #  VERIFIED COMPUTATION -- copied verbatim from _memristor_bloch_traj
    #  (make_animations.py, modulated=True). No physics/params/seeds changed.
    # ==================================================================
    sx, sy, sz, sm = qt.sigmax(), qt.sigmay(), qt.sigmaz(), qt.sigmam()
    t, theta = qm.solve_theta(n_cycles=2)
    thetadot = qm.ETA * qm.drive(t) - qm.GAMMA * (theta - qm.THETA0)
    H = [[sy, 2.0 * thetadot]]                    # SAME modulation as the pinched loop
    # tap rate modulated by the drive: strong on the down-swing (decohere), ~0 on the
    # up-swing where the drive re-coheres the state (recover) -> visible revival
    gamma_t = 0.55 * (1 + np.cos(qm.OMEGA * t)) / 2 + 0.02
    c_ops = [[sm, np.sqrt(gamma_t)]]
    eops = [sx, sy, sz]

    def psi(th):                                  # R_y(4 th)|0> -> Bloch (sin4th,0,cos4th)
        return (np.cos(2 * th) * qt.basis(2, 0) + np.sin(2 * th) * qt.basis(2, 1)).unit()

    sett = t >= (t.max() - per)
    th_amp = 0.5 * (theta[sett].max() - theta[sett].min())
    modulated = True
    # modulated start points (original sin() spacing -- preserves the ensemble
    # |r| range [0.27, 0.97]).  Visual distinctness of the three trajectories on
    # the spheres is achieved by line width / colour contrast, NOT by widening the
    # spread (which would change the ensemble decoherence and the |r| range).
    th_in = ([qm.THETA0 + th_amp * np.sin(2 * np.pi * i / len(cols)) for i in range(len(cols))]
             if modulated else [qm.THETA0] * len(cols))
    trajs = []
    for i, th0 in enumerate(th_in):
        mc = qt.mcsolve(H, psi(th0), t, c_ops, e_ops=eops, ntraj=1, seeds=i + 1)
        trajs.append([np.array(e) for e in mc.expect])
    trajs = np.array(trajs)                       # (3 trajectories, 3 components, n)
    rho0 = sum((qt.ket2dm(psi(th0)) for th0 in th_in),
               0 * qt.ket2dm(psi(th_in[0]))) / len(th_in)
    me = qt.mesolve(H, rho0, t, c_ops, e_ops=eops)
    ax_, ay_, az_ = (np.array(e) for e in me.expect)
    rlen = np.sqrt(ax_ ** 2 + ay_ ** 2 + az_ ** 2)

    # input-modulation arc + start markers (Bloch angle = 4*theta); qutip maps (x,y,z)->(y,-x,z)
    tarc = np.linspace(qm.THETA0 - th_amp, qm.THETA0 + th_amp, 40)
    arc_x, arc_z = np.sin(4 * tarc), np.cos(4 * tarc)
    st_x = np.array([np.sin(4 * th0) for th0 in th_in])
    st_z = np.array([np.cos(4 * th0) for th0 in th_in])

    np.savez(CACHE, t=t, trajs=trajs, ax_=ax_, ay_=ay_, az_=az_, rlen=rlen,
             arc_x=arc_x, arc_z=arc_z, st_x=st_x, st_z=st_z)

print(f"  bloch_revival: |r_avg| decoheres/recovers in [{rlen.min():.2f}, {rlen.max():.2f}]")
n = len(t)

# ======================================================================
#  STATIC PLOTTING  (new) -- pick the snapshot phases: |r_avg| max -> min
#  -> max across ONE drive cycle (the SECOND, settled cycle).
# ======================================================================
RMIN, RMAX = rlen.min(), rlen.max()

# work inside the second cycle [per, 2*per] so transients have settled
cyc = (t >= per - 1e-9) & (t <= 2 * per + 1e-9)
ti = np.where(cyc)[0]
# index of the |r_avg| MINIMUM inside this cycle
i_min = ti[np.argmin(rlen[ti])]
# the two surrounding MAXIMA (a max before and a max after the min)
left = ti[ti <= i_min]
right = ti[ti >= i_min]
i_maxL = left[np.argmax(rlen[left])]
i_maxR = right[np.argmax(rlen[right])]
snap_idx = [i_maxL, i_min, i_maxR]
snap_lab = ["max", "min", "max"]
# single NEUTRAL grey for all three snapshot stems/markers: matches the panel-(d)
# legend handle exactly, avoids reusing the ensemble-vector RED for an unrelated
# meaning, and the direct 0.97/0.63/0.91 annotations already disambiguate phases
PHASE_C = ["0.45", "0.45", "0.45"]

print(f"  snapshot |r_avg|: {rlen[i_maxL]:.2f} -> {rlen[i_min]:.2f} -> {rlen[i_maxR]:.2f}")


def draw_bloch(fig, gridspec_cell, j, title):
    """Render ONE Bloch sphere (state up to time index j) into a 3D sub-axes."""
    b = qt.Bloch(fig=fig, axes=fig.add_subplot(gridspec_cell, projection="3d"))
    b.frame_alpha = 0.10
    b.sphere_alpha = 0.04
    b.font_size = int(round(0.95 * plt.rcParams["font.size"]))
    b.xlabel = [r"$x$", ""]
    b.ylabel = [r"$y$", ""]
    b.zlabel = [r"$|0\rangle$", r"$|1\rangle$"]
    # DECLUTTER: subsample each trajectory (every SUB-th point) and draw thin,
    # semi-transparent lines so the three paths read as distinct strands instead
    # of a tangle. Colours (blue/green/magenta) stay fully distinct.
    SUB = 3
    for (tx, ty, tz), c in zip(trajs, cols):
        b.add_points([tx[:j:SUB], ty[:j:SUB], tz[:j:SUB]], meth="l", colors=[c],
                     alpha=0.65)
    b.point_color = list(cols)
    b.point_marker = ["o"]
    # thin trajectory lines for a clean, readable sphere (de-tangled)
    b.line_width = 1.3
    b.vector_color = [RED]
    b.vector_width = 3
    b.add_vectors([ax_[j - 1], ay_[j - 1], az_[j - 1]])
    b.render()
    a3 = b.axes
    # ZOOM IN: tighter cube limits make the unit sphere fill its cell (the
    # default 1.2 padding wasted ~40% of each panel as empty space). 0.78 keeps
    # the |0>/|1> pole labels and the x/y axis labels just inside the frame.
    a3.set_xlim3d(-0.82, 0.82)
    a3.set_ylim3d(-0.82, 0.82)
    a3.set_zlim3d(-0.82, 0.82)
    # input-modulation arc + start markers (orange)
    a3.plot(np.zeros_like(arc_x), -arc_x, arc_z, color=ORANGE, lw=1.4, ls="--", alpha=0.9)
    a3.scatter(np.zeros(len(st_x)), -st_x, st_z, color=ORANGE, s=22, marker="D",
               depthshade=False, zorder=6)
    # snapshot |r| value as a compact subtitle sitting close to the sphere top
    a3.set_title(title, fontsize=0.92 * plt.rcParams["font.size"], pad=-14)
    return b


# -------- SINGLE-COLUMN layout (3.4 in): 2+1 Bloch-sphere arrangement above
# the revival curve.  Row 0: snapshots (a),(b); row 1: snapshot (c) + the
# shared trajectory legend in the otherwise-empty right cell; row 2: the
# full-width revival curve (d).  Each sphere cell is ~1.65 in wide, so the
# rendered spheres stay > 1 in in diameter.  Fonts render ~1:1 at
# width=\columnwidth.
fig = plt.figure(figsize=(3.4, 5.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.5],
                      hspace=0.16, wspace=0.0,
                      left=0.135, right=0.975, top=0.975, bottom=0.083)

# --- rows 0-1: the three snapshots in a 2+1 arrangement ---
# manuscript notation: the ensemble length is |<r>| (matches the caption's
# $|\langle\mathbf{r}\rangle|$ and panel (d)'s y-label), not bare |r|
rr = lambda v: (f"$|\\langle\\mathbf{{r}}\\rangle|={v:.2f}$" if USETEX
                else f"|<r>|={v:.2f}")
plab = ["(a)", "(b)", "(c)"]
sphere_cells = [gs[0, 0], gs[0, 1], gs[1, 0]]
for col, (idx, lab, pc) in enumerate(zip(snap_idx, snap_lab, PHASE_C)):
    j = idx + 1
    # |r| value kept as a small subtitle only (NOT the panel letter)
    b = draw_bloch(fig, sphere_cells[col], j, rr(rlen[idx]))
    # clean OUTSIDE panel marker, parenthesized + bold (axis-free Bloch panel).
    # Bloch axes are 3D, so use text2D (axes fractions) to match figstyle.panel.
    # usetex ignores fontweight -- bold must be requested in the TeX string
    # (same convention as the patched figstyle.panel).
    lab = (r"\textbf{" + plab[col] + "}") if USETEX else plab[col]
    b.axes.text2D(0.06, 1.02, lab, transform=b.axes.transAxes,
                  fontweight="bold", ha="right", va="bottom",
                  fontsize=matplotlib.rcParams["axes.titlesize"])

# --- bottom: |r_avg|(t) revival curve over BOTH cycles ---
axc = fig.add_subplot(gs[2, :])
tt = t / per
axc.plot(tt, rlen, color=BLUE, lw=2.0, zorder=3)
axc.axvspan(1.0, 2.0, color="0.92", lw=0, zorder=0)   # the settled cycle shown above
# mark the three snapshot phases (dotted vertical lines + dots locate the
# Bloch snapshots; NOT double-labelled with panel letters). Annotate each with
# its |r| value so the dip-to-0.63 and recovery-to-0.91 are obvious.
val_off = [(0.0, 0.07, "center", "bottom"),     # (a) max above
           (0.0, -0.095, "center", "top"),      # (b) dip, below
           (-0.045, 0.075, "right", "bottom")]  # (c) recovery, left of edge
for col, (idx, pc, off) in enumerate(zip(snap_idx, PHASE_C, val_off)):
    axc.axvline(tt[idx], color=pc, ls=":", lw=1.3, zorder=2)
    # clip_on=False: the (c) snapshot sits exactly at t/T = 2 (the x-limit),
    # so a clipped marker would render as a half-dot under the right spine
    axc.plot(tt[idx], rlen[idx], "o", color=pc, ms=6.0, zorder=4,
             mec="white", mew=0.8, clip_on=False)
    dx, dy, ha, va = off
    axc.annotate(f"{rlen[idx]:.2f}", xy=(tt[idx], rlen[idx]),
                 xytext=(tt[idx] + dx, rlen[idx] + dy), ha=ha, va=va,
                 color="k", fontsize=0.9 * plt.rcParams["font.size"],
                 fontweight="bold", zorder=5,
                 bbox=dict(facecolor="0.92", edgecolor="none", pad=0.6))
axc.set_xlim(0, tt.max())
axc.set_ylim(max(0.0, RMIN - 0.10), 1.26)   # extra headroom for the in-panel legend
axc.set_xlabel(r"drive phase $\;t/T_{\Omega}$" if USETEX else r"drive phase  t/T")
axc.set_ylabel(r"$|\langle\mathbf{r}\rangle|$" if USETEX else r"|<r>|")
axc.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
axc.set_yticks([0.25, 0.5, 0.75, 1.0])
axc.xaxis.set_major_formatter(figstyle.STRIP_FMT)   # 1 not 1.00, 0.5 not 0.50
axc.yaxis.set_major_formatter(figstyle.STRIP_FMT)
axc.text(1.5, RMIN - 0.05, "settled cycle (a-c)", ha="center", va="bottom",
         color="k", fontsize=0.9 * plt.rcParams["font.size"], zorder=5,
         bbox=dict(facecolor="0.92", edgecolor="none", pad=0.6))

# OUTSIDE panel marker for the curve panel (parenthesized + bold, top-left)
figstyle.panel(axc, "(d)")

# --- panel (d)'s OWN legend: describe ONLY what panel (d) plots ---
# the blue ensemble |r|(t) curve, and the dotted snapshot-phase markers that
# locate the three Bloch snapshots (a)-(c).  No trajectory entries here.
d_handles = [
    Line2D([0], [0], color=BLUE, lw=2.0,
           label=r"ensemble $|\langle\mathbf{r}\rangle|(t)$" if USETEX
           else "ensemble |<r>|(t)"),
    Line2D([0], [0], color="0.45", ls=":", lw=1.3, marker="o", mfc="0.45",
           mec="white", mew=0.8, ms=6,
           label="snapshot phases (a-c)"),
]
# upper-LEFT (the unshaded first cycle leaves the top-left corner empty, and
# this keeps the legend clear of the 0.97 snapshot annotation further right)
axc.legend(handles=d_handles, loc="upper left", ncol=1,
           fontsize=0.78 * plt.rcParams["font.size"], frameon=False,
           handlelength=1.2, labelspacing=0.25, borderpad=0.15,
           handletextpad=0.5, bbox_to_anchor=(0.01, 1.02))

# --- SHARED legend for the Bloch-sphere ROW (a)-(c): the coloured trajectories
# and the modulated input-start key live on the spheres, so the colour key
# belongs above that row, not on panel (d). ---
row_handles = [
    Line2D([0], [0], color=ORANGE, ls="--", lw=1.6, marker="D", ms=5,
           label="input start (modulated)"),
    Line2D([0], [0], color=cols[0], lw=2.4, label="trajectory 1"),
    Line2D([0], [0], color=cols[1], lw=2.4, label="trajectory 2"),
    Line2D([0], [0], color=cols[2], lw=2.4, label="trajectory 3"),
    Line2D([0], [0], color=RED, lw=2.6, label="ensemble avg.\\ vector" if USETEX
           else "ensemble avg. vector"),
]
# place the shared legend in the EMPTY grid cell next to snapshot (c)
# (row 1, right column): a single vertical column of the five entries.
pc_ = gs[1, 1].get_position(fig)   # the empty legend cell
fig.legend(handles=row_handles, loc="center", ncol=1,
           fontsize=0.78 * plt.rcParams["font.size"],   # figstyle legend size
           frameon=False,
           handlelength=1.0, columnspacing=1.0, labelspacing=0.5,
           borderpad=0.1, handletextpad=0.4,
           bbox_to_anchor=(0.5 * (pc_.x0 + pc_.x1), 0.5 * (pc_.y0 + pc_.y1)))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "papers", "figures", "paper_bloch_revival")
# Save at the EXACT figure size (no tight bbox): the qutip 3D Bloch axes
# overflow their gridspec cells, and a tight bbox would inflate the PDF media
# box to ~10 in wide, shrinking the on-page apparent font. Fixing the media box
# to the 7.1 in figsize keeps apparent font ~= base * 7.0/7.1 ~= base.
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("wrote", OUT + ".pdf", "and .png")
print(f"FINAL |r_avg| range: [{RMIN:.2f}, {RMAX:.2f}]")
