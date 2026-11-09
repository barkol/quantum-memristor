#!/usr/bin/env python3
r"""Generate all presentation illustrations as matplotlib PDF figures with
LaTeX-rendered labels.  All outputs go to ../figures/pres_*.pdf.

Every figure is the result of an actual numerical simulation or geometric
construction (no hand-drawn TikZ); labels use the same braket/physics LaTeX
macros as the slide deck so notation matches exactly.
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

import figstyle

# ----------------------------------------------------------------------
# global style: shared HOM/blue-red house style (figstyle), sized for slides
# ----------------------------------------------------------------------
figstyle.apply(base_fontsize=13)

# HOM colour scheme (inherited from figstyle): blue primary, tab:red secondary,
# grey/neutral auxiliary lines.  THIRD is used where a 3rd colour is needed.
QBLUE = figstyle.BLUE    # 'b'  -- primary series / schematics
QRED = figstyle.RED      # 'tab:red' -- secondary series / phase / feedback
QGREEN = figstyle.THIRD  # neutral third colour (was a separate green)
GREY = figstyle.GREY

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUTDIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {path}")


# ======================================================================
# 1.  Classical Hopfield energy landscape (3 stored patterns)
# ======================================================================
def fig_hopfield_landscape():
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.linspace(-2.6, 2.6, 600)
    centres = [-1.6, 0.0, 1.6]
    E = 0.05 * x ** 2 + 1.4
    for c in centres:
        E -= 0.55 * np.exp(-2.5 * (x - c) ** 2)
    ax.plot(x, E, color=QBLUE, lw=1.6)
    for c, lbl in zip(centres, [r"$\xi^{1}$", r"$\xi^{2}$", r"$\xi^{3}$"]):
        well_y = 0.05 * c ** 2 + 1.4 - 0.55
        ax.plot(c, well_y, "o", color=QRED, ms=4.5)
        ax.text(c, well_y - 0.18, lbl, color=QRED, ha="center", fontsize=10)
    ax.set_xlabel(r"configuration $\sigma$")
    ax.set_ylabel(r"energy $E(\sigma)$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(bottom=0.45)
    save(fig, "pres_hopfield_landscape.pdf")


# ======================================================================
# 2.  Pinched I-V loop from HP-style memristor simulation
# ======================================================================
def _strukov_pinched(I0=1.0, omega=2*np.pi, w0=0.5, R_on=0.5, R_off=5.0,
                     D=1.0, mu_eff=4.0, cycles=2, N=4000):
    """Strukov memristor under sinusoidal current drive.
    Returns (i, v) traces for the *second* cycle (transient discarded).
    """
    t = np.linspace(0, cycles * 2 * np.pi / omega, N)
    dt = t[1] - t[0]
    i = I0 * np.sin(omega * t)
    w = np.zeros_like(t)
    w[0] = w0
    v = np.zeros_like(t)
    for k in range(1, len(t)):
        # window function to keep w in [0,1]
        f = w[k-1] * (1 - w[k-1])
        w[k] = np.clip(w[k-1] + mu_eff * (R_on / D) * i[k-1] * f * dt,
                       1e-4, 1 - 1e-4)
        R = R_on * w[k] / D + R_off * (1 - w[k] / D)
        v[k] = R * i[k]
    # take last full cycle
    cyc = slice(N // 2, N)
    return i[cyc], v[cyc]


def fig_memristor_iv_classical():
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    i, v = _strukov_pinched()
    ax.plot(i, v, color=QBLUE, lw=1.6)
    ax.plot(0, 0, "o", color=QBLUE, ms=4)
    ax.set_xlabel(r"current $i$")
    ax.set_ylabel(r"voltage $v$")
    figstyle.zero_axes(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.02, 0.02, r"pinched at $(0,0)$",
            transform=ax.transAxes, fontsize=9, color=QBLUE)
    save(fig, "pres_memristor_iv.pdf")


# ======================================================================
# 3.  Hopfield network topology (K_5 with Hebb weight annotation)
# ======================================================================
def fig_hopfield_network_topology():
    import networkx as nx
    fig, ax = plt.subplots(figsize=(2.8, 2.3))
    G = nx.complete_graph(5)
    pos = nx.circular_layout(G)
    nx.draw_networkx_edges(G, pos, edge_color=GREY, width=0.7, alpha=0.7, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=QBLUE, node_size=420,
                           alpha=0.85, ax=ax)
    labels = {i: rf"$x_{{{i+1}}}$" for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_color="white",
                            font_size=9, ax=ax)
    ax.text(0.5, 1.18,
            r"$J_{ij}=\dfrac{1}{n}\sum_{\mu}\xi_i^{\mu}\xi_j^{\mu}$",
            transform=ax.transAxes, ha="center", color=QRED, fontsize=10)
    ax.axis("off")
    save(fig, "pres_hopfield_network.pdf")


# ======================================================================
# 4.  Bloch sphere (X-Z projection) with HWP rotation arc
# ======================================================================
def _bloch_disk(ax, with_hwp_arc=True, basis_arrows=False):
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color=QBLUE, lw=1.0))
    ax.annotate("", xy=(1.25, 0), xytext=(-1.25, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=0.6))
    ax.annotate("", xy=(0, 1.25), xytext=(0, -1.25),
                arrowprops=dict(arrowstyle="->", color="k", lw=0.6))
    ax.text(1.32, 0.02, r"$\hat\sigma_x$", fontsize=9)
    ax.text(0.04, 1.30, r"$\hat\sigma_z$", fontsize=9)
    pts = [(0, 1, r"$\ket{H}$", QBLUE, "left", "bottom"),
           (0, -1, r"$\ket{V}$", QBLUE, "left", "top"),
           (1, 0, r"$\ket{+}$", QGREEN, "left", "center"),
           (-1, 0, r"$\ket{-}$", QGREEN, "right", "center")]
    for x, y, lbl, c, ha, va in pts:
        # in the basis-arrow figure the |+> point is the feedback source and is
        # surrounded by arrows/arc; place its ket label below-right so it does
        # not collide with them.
        if basis_arrows and (x, y) == (1, 0):
            ax.plot(x, y, "o", color=QGREEN, ms=5)
            ax.text(1.06, -0.16, lbl, color=c, ha="left", va="center",
                    fontsize=10)
            continue
        ax.plot(x, y, "o", color=c, ms=4)
        dx = 0.07 if ha == "left" else -0.07
        ax.text(x + dx, y + 0.05 if va == "bottom" else (y - 0.10 if va == "top" else y),
                lbl, color=c, ha=ha, va="center", fontsize=10)
    if with_hwp_arc:
        arc = Arc((0, 0), 2.0, 2.0, theta1=-30, theta2=90,
                  color=QRED, lw=1.2, linestyle="--")
        ax.add_patch(arc)
        ax.annotate("", xy=(0.866, -0.5), xytext=(0.95, -0.4),
                    arrowprops=dict(arrowstyle="->", color=QRED, lw=1.0))
        ax.text(1.05, 0.75, r"HWP rotation",
                color=QRED, fontsize=9)
    if basis_arrows:
        # input |+> at (1,0); show how feedback in different bases would rotate
        # it.  Arrows emanate from |+> in three clearly separated directions;
        # each label sits OUTSIDE the circle with a short leader line so nothing
        # overlaps the circle, axes, or the state kets.
        src = (1.0, 0.0)
        ax.plot(*src, "o", color=QGREEN, ms=5)

        def _fb(angle_deg, length, color, label, lab_xy, lab_ha,
                arc=False):
            ang = np.radians(angle_deg)
            tip = (src[0] + length * np.cos(ang),
                   src[1] + length * np.sin(ang))
            if arc:
                a = Arc((src[0] - 0.18, 0.0), 0.34, 0.34,
                        theta1=-150, theta2=150, color=color, lw=1.4)
                ax.add_patch(a)
                anchor = (src[0] - 0.18, -0.17)
            else:
                ax.annotate("", xy=tip, xytext=src,
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=1.6, mutation_scale=13,
                                            shrinkA=0, shrinkB=0))
                anchor = tip
            # leader line from arrow tip/arc to the external label
            ax.annotate("", xy=anchor, xytext=lab_xy,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.6))
            ax.text(lab_xy[0], lab_xy[1], label, color=color, fontsize=9,
                    ha=lab_ha, va="center")

        # Z-feedback: rotation about Z runs up the equator -> tilt up-right
        _fb(58, 0.62, QBLUE, r"$Z$-fb", (1.20, 0.82), "left")
        # Y-feedback: rotation about Y tilts toward |H> -> steeper, up-left
        _fb(118, 0.62, QRED, r"$Y$-fb", (-0.52, 1.10), "center")
        # X-feedback: |+> is an X eigenstate -> trivial small loop, label below
        _fb(0, 0, QGREEN, r"$X$-fb (no rot.)", (0.50, -1.22), "left",
            arc=True)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_bloch_primer():
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    _bloch_disk(ax, with_hwp_arc=True, basis_arrows=False)
    save(fig, "pres_bloch_primer.pdf")


def fig_bloch_basis_intuition():
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    _bloch_disk(ax, with_hwp_arc=False, basis_arrows=True)
    save(fig, "pres_bloch_basis.pdf")


# ======================================================================
# 5.  Pinched loop (small, for "Why memristors?" slide — same sim)
# ======================================================================
def fig_pinched_loop_small():
    fig, ax = plt.subplots(figsize=(2.8, 2.0))
    i, v = _strukov_pinched()
    ax.plot(i, v, color=QBLUE, lw=1.6)
    ax.plot(0, 0, "o", color=QBLUE, ms=4)
    imax = max(abs(i).max(), 1.0)
    vmax = max(abs(v).max(), 1.0)
    ax.annotate("", xy=(1.15 * imax, 0), xytext=(-1.15 * imax, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=0.5))
    ax.annotate("", xy=(0, 1.20 * vmax), xytext=(0, -1.20 * vmax),
                arrowprops=dict(arrowstyle="->", color="k", lw=0.5))
    ax.text(1.20 * imax, 0.02, r"$I$", fontsize=10)
    ax.text(0.05, 1.25 * vmax, r"$V$", fontsize=10)
    ax.axis("off")
    ax.text(0, -1.40 * vmax, r"pinched at $(0,0)$", color=QBLUE,
            ha="center", fontsize=9)
    save(fig, "pres_pinched_small.pdf")


# ======================================================================
# 6.  Synapse–memristor analogy
# ======================================================================
def fig_synapse_memristor():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    # synapse
    ax.add_patch(FancyBboxPatch((0.05, 1.4), 0.35, 0.4,
                                boxstyle="round,pad=0.02",
                                ec=QBLUE, fc="white", lw=1.0))
    ax.text(0.225, 1.6, "pre-syn", ha="center", va="center",
            color=QBLUE, fontsize=9)
    ax.add_patch(FancyBboxPatch((0.60, 1.4), 0.35, 0.4,
                                boxstyle="round,pad=0.02",
                                ec=QBLUE, fc="white", lw=1.0))
    ax.text(0.775, 1.6, "post-syn", ha="center", va="center",
            color=QBLUE, fontsize=9)
    ax.annotate("", xy=(0.60, 1.6), xytext=(0.40, 1.6),
                arrowprops=dict(arrowstyle="->", color=QGREEN, lw=1.4))
    ax.text(0.5, 1.85, r"synapse $w$", ha="center", color=QGREEN, fontsize=9)
    # memristor circuit
    ax.annotate("", xy=(0.30, 0.6), xytext=(0.05, 0.6),
                arrowprops=dict(arrowstyle="->", color=QRED, lw=1.0))
    ax.add_patch(FancyBboxPatch((0.30, 0.5), 0.20, 0.2,
                                boxstyle="square,pad=0.02",
                                ec="k", fc=QGREEN, alpha=0.4, lw=1.0))
    ax.text(0.40, 0.6, r"$M$", ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(0.75, 0.6), xytext=(0.50, 0.6),
                arrowprops=dict(arrowstyle="->", color=QRED, lw=1.0))
    ax.text(0.4, 0.30, "memristor", ha="center", fontsize=9)
    # dashed analogy line
    ax.plot([0.5, 0.5], [1.35, 0.75], "k--", lw=0.8)
    ax.text(0.55, 1.05, "analogy", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 2.0)
    ax.axis("off")
    save(fig, "pres_synapse_memristor.pdf")


# ======================================================================
# 7.  Two-level architecture (quantum + classical feedback loop)
# ======================================================================
def fig_feedback_diagram():
    fig, ax = plt.subplots(figsize=(3.6, 2.4))

    def _arrow(p0, p1, color, lw=1.8, ls="-", style="-|>", rad=0.0):
        cs = f"arc3,rad={rad}" if rad else "arc3"
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    ls=ls, shrinkA=0, shrinkB=0,
                                    connectionstyle=cs,
                                    mutation_scale=14))

    # ----- top (quantum) lane: |psi_in> -> U(theta) -> |psi_out> -----------
    y_top = 1.60
    # U(theta) box, centred on the lane
    ux0, uy0, uw, uh = 1.00, y_top - 0.18, 0.52, 0.36
    ucx = ux0 + uw / 2          # box centre x
    # input ket -> left edge of U box
    ax.text(0.10, y_top, r"$\ket{\psi_\text{in}}$", color="k", ha="left",
            va="center", fontsize=11)
    _arrow((0.46, y_top), (ux0, y_top), QBLUE)
    ubox = FancyBboxPatch((ux0, uy0), uw, uh,
                          boxstyle="round,pad=0.015",
                          ec=QBLUE, fc="#cce5f1", lw=1.4)
    ax.add_patch(ubox)
    ax.text(ucx, y_top, r"$\hat U(\theta)$", ha="center", va="center",
            color="k", fontsize=12)
    ax.text(ucx, uy0 + uh + 0.12, r"quantum", color=QBLUE, ha="center",
            fontsize=9)
    # right edge of U box -> output ket
    _arrow((ux0 + uw, y_top), (2.06, y_top), QBLUE)
    ax.text(2.12, y_top, r"$\ket{\psi_\text{out}}$", color="k", ha="left",
            va="center", fontsize=11)

    # ----- bottom (classical) lane: memory box -----------
    y_bot = 0.55
    mx0, my0, mw, mh = 0.85, y_bot - 0.18, 0.84, 0.36
    mcx = mx0 + mw / 2
    mb = FancyBboxPatch((mx0, my0), mw, mh,
                        boxstyle="round,pad=0.015",
                        ec=QRED, fc="#f4d1d1", lw=1.4)
    ax.add_patch(mb)
    ax.text(mcx, y_bot, r"memory $\theta(t)$", ha="center", va="center",
            color="k", fontsize=10)
    ax.text(mcx, my0 - 0.16, r"measurement record",
            color=GREY, ha="center", fontsize=8.5)

    # ----- measurement N(t): U-box bottom -> memory-box top (right side) ----
    x_meas = ux0 + uw - 0.08
    _arrow((x_meas, uy0), (x_meas, my0 + mh), "k", lw=1.6, ls="--")
    ax.text(x_meas + 0.06, (uy0 + my0 + mh) / 2, r"$N(t)$", color="k",
            ha="left", va="center", fontsize=9.5)

    # ----- red feedback loop: memory-box left -> up -> into U-box bottom ----
    # route out to the left of the memory box (clear of the input ket), up the
    # left margin, then in along the lane height and UP into the bottom-left of
    # the U box.  The arrowhead enters the box bottom, well left of N(t).
    x_fb = 0.30
    x_into = ux0 + 0.14        # where the loop turns up into the U box bottom
    y_turn = uy0 - 0.18        # horizontal run sits below the U box, clear of
    #                            the input lane (y_top) above it
    _arrow((mx0, y_bot), (x_fb, y_bot), QRED, ls="--", style="-")
    _arrow((x_fb, y_bot), (x_fb, y_turn), QRED, ls="--", style="-")
    _arrow((x_fb, y_turn), (x_into, y_turn), QRED, ls="--", style="-")
    _arrow((x_into, y_turn), (x_into, uy0), QRED, ls="--")
    ax.text(x_fb - 0.06, (y_bot + y_turn) / 2, r"feedback", color=QRED,
            ha="right", va="center", rotation=90, fontsize=9.5)
    # control label beside the up-into-box segment (memory theta -> U)
    ax.text(x_into + 0.06, (y_turn + uy0) / 2, r"control $\theta$",
            color=QRED, ha="left", va="center", fontsize=9.5)
    ax.set_xlim(-0.05, 2.70)
    ax.set_ylim(0.10, 2.05)
    ax.axis("off")
    save(fig, "pres_feedback_diagram.pdf")


# ======================================================================
# 8.  Lorentzian A(omega)
# ======================================================================
def fig_lorentzian_Aw():
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = np.logspace(-1.3, 1.3, 400)
    A = 2 * w / (1 + w ** 2)
    ax.plot(w, A, color=QBLUE, lw=1.6,
            label=r"$A(\omega)\propto \dfrac{\eta\Gamma_\text{ph}\,\omega}{\gamma^2+\omega^2}$")
    ax.axvline(1, color=QGREEN, lw=1.0, ls="--")
    ax.text(1.05, 0.55, r"$\omega=\gamma$", color=QGREEN, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"drive frequency $\omega/\gamma$")
    ax.set_ylabel(r"loop area $A(\omega)/A_\text{max}$")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", frameon=False, fontsize=8)
    save(fig, "pres_lorentzian_Aw.pdf")


# ======================================================================
# 9.  Retrieval energy landscape with tunnelling arrow
# ======================================================================
def fig_retrieval_landscape():
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    x = np.linspace(-2.6, 2.6, 600)
    centres = [-1.6, 0.0, 1.6]
    E = 0.05 * x ** 2 + 1.4
    for c in centres:
        E -= 0.55 * np.exp(-2.5 * (x - c) ** 2)
    ax.plot(x, E, color=QBLUE, lw=1.5)
    for c, lbl in zip(centres, [r"$\xi^{1}$", r"$\xi^{2}$", r"$\xi^{3}$"]):
        well_y = 0.05 * c ** 2 + 1.4 - 0.55
        ax.plot(c, well_y, "o", color=QGREEN, ms=4)
        ax.text(c, well_y - 0.18, lbl, color=QGREEN, ha="center", fontsize=10)
    # tunnelling arrow from xi^1 to xi^2 over barrier
    well_y1 = 0.05 * (-1.6) ** 2 + 1.4 - 0.55
    ax.annotate("", xy=(0, well_y1), xytext=(-1.6, well_y1),
                arrowprops=dict(arrowstyle="->", color=QRED,
                                connectionstyle="arc3,rad=-0.5",
                                ls="--", lw=1.2))
    ax.text(-0.8, well_y1 + 0.55, r"$\Gamma_\text{tun}\sim\Omega$",
            color=QRED, ha="center", fontsize=10)
    ax.set_xlabel(r"$\sigma$ (configuration)")
    ax.set_ylabel(r"$E$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(bottom=0.45)
    save(fig, "pres_retrieval_landscape.pdf")


# ======================================================================
# 10.  QHN topology (qubits + attached memristors)
# ======================================================================
def fig_qhn_network():
    import networkx as nx
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    G = nx.complete_graph(5)
    pos = nx.circular_layout(G)
    # shrink the qubit ring slightly to leave room for memristor boxes outside
    pos = {k: (0.75 * v[0], 0.75 * v[1]) for k, v in pos.items()}
    nx.draw_networkx_edges(G, pos, edge_color=GREY, width=0.6, alpha=0.7, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=QBLUE, node_size=380,
                           alpha=0.85, ax=ax)
    qlabels = {i: rf"q$_{i+1}$" for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=qlabels, font_color="white",
                            font_size=9, ax=ax)
    bw, bh = 0.30, 0.18         # memristor box size (consistent)
    off = 0.42                  # consistent radial offset from node centre
    for i, (x, y) in pos.items():
        # push each memristor box straight outward along the node's radial
        # direction by the SAME offset, so every mu_i sits in the same
        # relative position just outside its qubit node.
        r = np.hypot(x, y)
        ux, uy = (x / r, y / r) if r > 1e-3 else (1, 0)
        mx, my = x + off * ux, y + off * uy
        # short clean link from node edge to box edge (radial)
        ax.plot([x + 0.16 * ux, mx - 0.5 * bw * ux],
                [y + 0.16 * uy, my - 0.5 * bh * uy],
                color=QRED, lw=0.8)
        ax.add_patch(Rectangle((mx - bw / 2, my - bh / 2), bw, bh,
                               ec=QRED, fc="#f4d1d1", lw=1.0))
        ax.text(mx, my, rf"$\mu_{i+1}$", ha="center", va="center",
                fontsize=8, color="k")
    # anchor the coupling label on one representative edge (q1 <-> q2)
    (xa, ya), (xb, yb) = pos[0], pos[1]
    ax.annotate(r"$J_{ij}(\mu_i,\mu_j)$",
                xy=((xa + xb) / 2, (ya + yb) / 2),
                xytext=(0.0, 1.78), color=QRED, ha="center", fontsize=10,
                arrowprops=dict(arrowstyle="-", color=QRED, lw=0.7))
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.6, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "pres_qhn_network.pdf")


# ======================================================================
# 11.  Palimpsest retrievability
# ======================================================================
def fig_palimpsest():
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    x = np.linspace(0, 6, 200)
    ax.plot(x, np.exp(-x), color=QBLUE, lw=1.6)
    ax.set_xlabel(r"pattern age $\gamma(T-t_\mu)$")
    ax.set_ylabel(r"retrievability $r_\mu$")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    save(fig, "pres_palimpsest.pdf")


# ======================================================================
# 12.  Energy landscape morphing during storage (3 time slices)
# ======================================================================
def fig_landscape_morphing():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.linspace(-2.5, 2.5, 400)
    centres = [-1.4, 0.0, 1.4]
    for offset, depth, col, lbl in [(0.0, 0.15, QBLUE, r"$t_0$"),
                                    (1.2, 0.35, QGREEN, r"$t_1$"),
                                    (2.4, 0.55, QRED, r"$t_2$")]:
        E = offset + 1.0
        for c in centres:
            E = E - depth * np.exp(-2.5 * (x - c) ** 2)
        ax.plot(x, E, color=col, lw=1.4)
        ax.text(2.6, offset + 0.85, lbl, color=col, fontsize=10)
    ax.set_xlabel(r"configuration $\sigma$")
    ax.set_ylabel(r"$E$ (time-shifted)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"time $\to$ wells deepen at stored patterns",
                 fontsize=9, pad=4)
    save(fig, "pres_landscape_morphing.pdf")


# ======================================================================
# 13.  Composition flowchart (4 layers)
# ======================================================================
def fig_composition_flowchart():
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    layers = [
        (r"single photon $\ket{\psi}$",  "#cce5f1", "quantum object"),
        (r"HWP + detector + FPGA",       "#aacfe6", "measurement + control loop"),
        (r"photonic memristor (qubit + $\theta$)", "#cce8c2", "history-dependent unit"),
        (r"memristive QHN ($n$ qubits, $\binom{n}{2}$ couplings)",
                                          "#f4d1d1", "associative memory"),
    ]
    width = 3.6
    x0 = 0.0
    annox = x0 + width + 0.4
    for i, (txt, col, anno) in enumerate(layers):
        y = (3 - i) * 0.65
        ax.add_patch(FancyBboxPatch((x0, y), width, 0.45,
                                    boxstyle="round,pad=0.02",
                                    fc=col, ec="k", lw=0.6))
        ax.text(x0 + width / 2, y + 0.22, txt, ha="center", va="center",
                fontsize=9)
        ax.text(annox, y + 0.22, anno, ha="left", va="center",
                fontsize=9, color=GREY)
        if i < 3:
            ax.annotate("", xy=(x0 + width / 2, y),
                        xytext=(x0 + width / 2, y - 0.2),
                        arrowprops=dict(arrowstyle="<-", color="k", lw=0.8))
    ax.set_xlim(-0.2, 8.0)
    ax.set_ylim(-0.1, 2.7)
    ax.axis("off")
    save(fig, "pres_composition.pdf")


# ======================================================================
# 14.  NARMA architecture
# ======================================================================
def fig_narma_architecture():
    import networkx as nx
    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    G = nx.gnp_random_graph(8, 0.35, seed=2)
    pos = nx.spring_layout(G, seed=2)
    pos = {k: (v[0] * 0.7, v[1] * 0.55) for k, v in pos.items()}
    ax.add_patch(FancyBboxPatch((-1.05, -0.85), 2.1, 1.7,
                                boxstyle="round,pad=0.04",
                                ec=QBLUE, fc="#e6f1f7", lw=1.0))
    nx.draw_networkx_edges(G, pos, edge_color=GREY, width=0.6, alpha=0.7, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=QBLUE, node_size=80,
                           alpha=0.85, ax=ax)
    ax.annotate("", xy=(-1.05, 0), xytext=(-1.6, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
    ax.text(-1.65, 0, r"$u_t$", ha="right", va="center", fontsize=10)
    ax.annotate("", xy=(1.55, 0), xytext=(1.05, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
    ax.text(1.60, 0, r"$\hat y_{t+1}$", ha="left", va="center", fontsize=10)
    ax.text(0, 0.95, r"photonic reservoir", ha="center",
            color=QBLUE, fontsize=9)
    ax.text(0, -1.05, r"only linear readout weights trained",
            ha="center", color=QGREEN, fontsize=9)
    ax.set_xlim(-2.0, 2.2)
    ax.set_ylim(-1.3, 1.2)
    ax.axis("off")
    save(fig, "pres_narma_arch.pdf")


# ======================================================================
# 15.  NARMA-10 simulation + reservoir prediction
# ======================================================================
def fig_narma_prediction():
    """Run NARMA-10 task; reservoir = random linear-nonlinear features."""
    rng = np.random.default_rng(0)
    T = 800
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for t in range(10, T - 1):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * y[t - 9:t + 1].sum()
                    + 1.5 * u[t - 9] * u[t]
                    + 0.1)
    # simple "reservoir" features: 30 random tanh of past inputs/outputs
    n_feat = 30
    Win = rng.standard_normal((n_feat, 21)) * 0.5
    feats = np.zeros((T, n_feat))
    for t in range(20, T):
        window = np.concatenate([u[t - 9:t + 1], y[t - 9:t + 1], [1.0]])
        feats[t] = np.tanh(Win @ window)
    train = slice(100, 600)
    test = slice(600, 780)
    # ridge regression
    X, t_target = feats[train], y[train.start + 1:train.stop + 1]
    alpha = 1e-3
    W = np.linalg.solve(X.T @ X + alpha * np.eye(n_feat), X.T @ t_target)
    y_pred = feats[test] @ W
    y_true = y[test.start + 1:test.stop + 1]
    nmse = np.mean((y_true - y_pred) ** 2) / np.var(y_true)

    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    tt = np.arange(len(y_true))
    ax.plot(tt, y_true, color=QBLUE, lw=1.0, label=r"target $y_{t+1}$")
    ax.plot(tt, y_pred, color=QRED, lw=1.0, ls="--", label=r"reservoir $\hat y_{t+1}$")
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel(r"NARMA-10")
    ax.set_title(rf"NMSE $\approx {nmse:.3f}$", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    save(fig, "pres_narma_pred.pdf")


# ======================================================================
# 16.  Vision: superposition of hysteresis loops
# ======================================================================
def fig_vision_loops():
    fig, ax = plt.subplots(figsize=(3.2, 2.6))

    def loop(theta_max, scale=1.0):
        t = np.linspace(0, 2 * np.pi, 400)
        I = np.sin(t)
        # output via a HWP-like memory: simplified parametric pinched loop
        O = scale * np.sin(2 * theta_max) * np.sin(t) * (1 + 0.4 * np.cos(t))
        return I, O

    I1, O1 = loop(0.3, 1.0)
    I2, O2 = loop(0.6, 1.0)
    Os = 0.5 * (O1 + O2)
    ax.plot(I1, O1, color=QBLUE, lw=1.2, alpha=0.8,
            label=r"$\ket{\theta_1}$")
    ax.plot(I2, O2, color=QRED, lw=1.2, alpha=0.8,
            label=r"$\ket{\theta_2}$")
    ax.plot(I1, Os, color=QGREEN, lw=1.4, ls="--",
            label=r"$\ket{\Theta}=\alpha\ket{\theta_1}+\beta\ket{\theta_2}$")
    figstyle.zero_axes(ax)
    ax.set_xlabel(r"input $\sin(2\alpha)$")
    ax.set_ylabel(r"output")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    save(fig, "pres_vision_loops.pdf")


# ======================================================================
# 17.  theta(t) trace for HHVHV worked example
# ======================================================================
def fig_theta_HHVHV():
    eta, gamma, dt = 0.10, 0.05, 1.0
    seq = [1, 1, 0, 1, 0]  # H,H,V,H,V
    times, thetas, ev_t, ev_th = [], [], [], []
    theta = 0.0
    t = 0.0
    times.append(t); thetas.append(theta)
    # finer continuous decay between detections plus instantaneous jumps
    for k, N in enumerate(seq):
        # decay segment
        sub_t = np.linspace(t, t + 1.0, 50)
        for tt in sub_t[1:]:
            theta = theta * np.exp(-gamma * (tt - t))
            t = tt
            times.append(t); thetas.append(theta)
        # detection jump
        theta = theta + eta * N
        times.append(t); thetas.append(theta)
        ev_t.append(t); ev_th.append(theta)
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.plot(times, thetas, color=QBLUE, lw=1.4)
    for k, (et, eth) in enumerate(zip(ev_t, ev_th)):
        col = QBLUE if seq[k] == 1 else QRED
        ax.plot(et, eth, "o", color=col, ms=5)
        ax.text(et, eth + 0.018, "H" if seq[k] == 1 else "V",
                color=col, ha="center", fontsize=9)
    ax.set_xlabel(r"time $t/\Delta t$")
    ax.set_ylabel(r"memory $\theta(t)$")
    save(fig, "pres_theta_HHVHV.pdf")


# ======================================================================
# main
# ======================================================================
def fig_quantum_network():
    """Fully-quantum optical Hopfield-network architecture.  Each memristor node
    is a 3-qubit programmable gate with THREE input ports and THREE output ports
    (target / program / control) plus two coherent delay loops: the target
    output recirculates as the program (1-step delay) and the program output as
    the control (2-step delay); the control output exits to measurement.  Two
    nodes are shown, entangled by an inter-node coupling, with a Bell/CHSH
    readout; an ellipsis denotes the n-node network."""
    from matplotlib.patches import Wedge
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    gw, gh = 1.55, 1.05
    cys = [2.7, 0.0]            # two node centres
    dy = 0.32                   # port spacing inside a box
    x0 = 2.8                    # gate left edge
    stub = 0.22
    det_x = 6.9

    def arrow(p0, p1, color, lw=1.6, style="-|>"):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    shrinkA=0, shrinkB=0))

    def node(cy, first=False):
        xl, xr = x0, x0 + gw
        yt, ym, yb = cy + dy, cy, cy - dy           # target / program / control
        ax.add_patch(FancyBboxPatch((xl, cy - gh / 2), gw, gh,
                                    boxstyle="round,pad=0.02",
                                    ec=QBLUE, fc="#e9f2f8", lw=1.6))
        ax.text(xl + gw / 2, cy, "3-qubit\ngate", ha="center", va="center",
                fontsize=9)
        for y in (yt, ym, yb):                       # port stubs
            ax.plot([xl - stub, xl], [y, y], color=QBLUE, lw=1.4)
            ax.plot([xr, xr + stub], [y, y], color=QBLUE, lw=1.4)
        # delay loop A: target-out -> program-in  (1-step delay, red, over top)
        ax.add_patch(FancyArrowPatch((xr + stub, yt), (xl - stub, ym),
                                     connectionstyle="arc3,rad=0.55",
                                     arrowstyle="-|>", color=QRED, lw=1.5,
                                     mutation_scale=11))
        # delay loop B: program-out -> control-in (2-step delay, larger arc)
        ax.add_patch(FancyArrowPatch((xr + stub, ym), (xl - stub, yb),
                                     connectionstyle="arc3,rad=1.0",
                                     arrowstyle="-|>", color=QRED, lw=1.5,
                                     mutation_scale=11))
        arrow((xl - 0.95, yt), (xl - stub, yt), QBLUE)     # fresh photon -> target
        arrow((xr + stub, yb), (det_x - 0.2, yb), QBLUE)   # control -> detector
        ax.add_patch(Wedge((det_x, yb), 0.18, 90, 270, fc=GREY, ec=GREY))
        if first:
            for y, lab in ((yt, "target"), (ym, "program"), (yb, "control")):
                ax.text(xl - 1.55, y, lab, ha="left", va="center",
                        fontsize=8, color=GREY)
        return yb

    yb0 = node(cys[0], first=True)
    yb1 = node(cys[1])

    # inter-node entangling coupling
    cx = x0 + gw / 2
    ax.plot([cx, cx], [cys[0] - gh / 2, cys[1] + gh / 2], color=QRED, lw=1.5)
    ax.text(cx + 0.12, (cys[0] + cys[1]) / 2, r"entangling $J_{ij}$",
            color=QRED, ha="left", va="center", fontsize=9)

    # CHSH readout fed by the two control-exit detectors
    bx, by, bw, bh = 7.75, 1.0, 1.3, 0.7
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.03",
                                ec=GREY, fc="white", lw=1.3))
    ax.text(bx + bw / 2, by + bh / 2, "CHSH", ha="center", va="center")
    for yb in (yb0, yb1):
        arrow((det_x + 0.18, yb), (bx, by + bh / 2), GREY, lw=1.0, style="-")
    ax.text(bx + bw / 2, by + bh + 0.22, r"$\mathcal{B}>2\Rightarrow$ quantum",
            color="k", ha="center", fontsize=9)

    # labels
    ax.text(x0 - 0.95, cys[0] + dy + 0.18, "fresh photons", color=QBLUE,
            ha="left", va="bottom", fontsize=9)
    ax.text(x0 + gw / 2, cys[0] + gh / 2 + 1.15,
            "coherent delay loops (1- \\& 2-step)",
            color=QRED, ha="center", va="bottom", fontsize=9)
    ax.annotate("", xy=(x0 + gw * 0.5, cys[0] + gh / 2 + 0.85),
                xytext=(x0 + gw * 0.5, cys[0] + gh / 2 + 1.12),
                arrowprops=dict(arrowstyle="-", color=QRED, lw=0.7))
    ax.text(det_x, yb1 - 0.45, "measure", color=GREY, ha="center", fontsize=9)
    ax.text(cx, cys[1] - gh / 2 - 0.5, r"$\vdots\quad n$ coupled memristors",
            color="k", ha="center", va="top", fontsize=9)

    ax.set_xlim(0.6, 9.4)
    ax.set_ylim(-1.6, 5.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    save(fig, "pres_quantum_network.pdf")


if __name__ == "__main__":
    fig_quantum_network()
    fig_hopfield_landscape()
    fig_memristor_iv_classical()
    fig_pinched_loop_small()
    fig_hopfield_network_topology()
    fig_bloch_primer()
    fig_bloch_basis_intuition()
    fig_synapse_memristor()
    fig_feedback_diagram()
    fig_lorentzian_Aw()
    fig_retrieval_landscape()
    fig_qhn_network()
    # fig_palimpsest() -- SUPERSEDED: pres_palimpsest.pdf is now produced from
    #   real simulation data by sim_qhnm.py; do not regenerate here (would
    #   overwrite the real-data figure).
    fig_landscape_morphing()
    fig_composition_flowchart()
    fig_narma_architecture()
    # fig_narma_prediction() -- SUPERSEDED: pres_narma_pred.pdf is now produced
    #   from real simulation data by sim_qrc_narma.py; do not regenerate here
    #   (would overwrite the real-data figure).
    fig_vision_loops()
    fig_theta_HHVHV()
    print("all figures generated.")
