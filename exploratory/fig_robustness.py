"""Robustness of a perturbed photon: single device (no correction) vs trained
network (pattern completion).

(a) The single-device map U(theta) is a HWP reflection on the Bloch sphere:
    reflections preserve angles, so an ideal/perturbed pair of inputs separated by
    delta comes out separated by the SAME delta -> fidelity preserved, no denoising.
(b) The trained QHN-M is an attractor: a corrupted input flows to the stored
    pattern. REAL data from sim_qhnm (retrieve_quantum_anneal vs retrieve_classical):
    recall overlap vs input corruption, Omega-on (quantum tunnelling) vs Omega-off.

Emits TWO standalone panel PDFs (pres_robustness_a / _b) sized for a two-column
slide layout (caption beneath each panel), plus the combined pres_robustness.pdf
for backup/compatibility.
"""
import os
import numpy as np

import figstyle
figstyle.apply(base_fontsize=12)          # slide-appropriate type, not print 18
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sim_qhnm as q

BLUE, RED, GREY = figstyle.BLUE, figstyle.RED, figstyle.GREY
THIRD = getattr(figstyle, "THIRD", "0.35")


# ----------------------------------------------------------------------
def panel_bloch(ax):
    """U(theta) = reflection across axis at 2theta: angle a -> 4theta - a.
    Decluttered: solid arrows = inputs, dashed = U-outputs; a compact legend
    carries the blue=ideal / red=perturbed key, so no crowded inline kets."""
    th = np.deg2rad(np.linspace(0, 360, 200))
    ax.plot(np.cos(th), np.sin(th), color="0.7", lw=1.0)
    ax.axhline(0, color="0.9", lw=0.6); ax.axvline(0, color="0.9", lw=0.6)

    theta = np.pi / 8                      # HWP angle -> reflection axis at 2theta=45deg
    a0 = np.deg2rad(36)                    # ideal input Bloch angle (from +z)
    delta = np.deg2rad(26)                 # perturbation (infidelity) angle
    ain = [a0, a0 + delta]
    aout = [4 * theta - a for a in ain]    # reflected

    # reflection axis (dashed, both directions)
    axang = 2 * theta
    ax.plot([-np.sin(axang), np.sin(axang)], [-np.cos(axang), np.cos(axang)],
            ls="--", color="0.55", lw=1.3)
    ax.text(np.sin(axang) * 1.10, np.cos(axang) * 1.10, r"HWP axis",
            color="0.45", fontsize=8.5, ha="left", va="bottom")
    # mark the reflection-axis angle 2*theta measured from |0> (vertical)
    arc2 = np.linspace(0.0, axang, 24)
    ax.plot(0.24 * np.sin(arc2), 0.24 * np.cos(arc2), color="0.45", lw=1.1)
    ax.text(0.34 * np.sin(0.45 * axang), 0.34 * np.cos(0.45 * axang),
            r"$2\theta$", color="0.45", fontsize=9, ha="center", va="center")

    def arrow(a, color, dashed=False):
        # a linear polarization is an undirected axis -> draw it as a DOUBLE-
        # headed arrow through the origin (from -(x,y) to +(x,y)).
        x, y = np.sin(a), np.cos(a)        # amplitude: x=sin a (|1>), y=cos a (|0>)
        ax.annotate("", xy=(x, y), xytext=(-x, -y),
                    arrowprops=dict(arrowstyle="<|-|>", color=color, lw=2.4,
                                    alpha=0.5 if dashed else 1.0,
                                    ls="--" if dashed else "-"))

    arrow(ain[0], BLUE);  arrow(ain[1], RED)          # inputs  (solid)
    arrow(aout[0], BLUE, True); arrow(aout[1], RED, True)   # outputs (dashed)

    # delta arcs (input + output) -- SAME opening angle, the whole point.
    # nested at different radii so the two delta labels don't collide.
    for (a, b), r in ((ain, 0.36), (aout, 0.60)):
        arc = np.linspace(a, b, 24)
        ax.plot(r * np.sin(arc), r * np.cos(arc), color="0.2", lw=1.6)
    am = 0.5 * (ain[0] + ain[1]); ax.text(0.49 * np.sin(am), 0.49 * np.cos(am),
            r"$\delta$", fontsize=10, ha="center", va="center")
    amo = 0.5 * (aout[0] + aout[1]); ax.text(0.73 * np.sin(amo), 0.73 * np.cos(amo),
            r"$\delta$", fontsize=10, ha="center", va="center")

    # Real-amplitude (Jones) plane: a HWP is a REFLECTION, so the basis states
    # are orthogonal -- |0> on the vertical axis, |1> on the HORIZONTAL axis
    # (the state at angle a from |0> has amplitude sin(a) along |1>, so a=90 deg
    # is |1>), not antipodal as on the Bloch sphere.
    ax.text(0, 1.16, r"$|0\rangle$", ha="center", va="bottom", fontsize=10)
    ax.text(1.16, 0, r"$|1\rangle$", ha="left", va="center", fontsize=10)
    # Tight limits -> the Bloch circle FILLS the column (less surrounding margin).
    ax.set_xlim(-1.30, 1.30); ax.set_ylim(-1.30, 1.34)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(r"HWP $\hat U(\theta)$: reflect polarization across $2\theta$",
                 fontsize=10, pad=4)

    handles = [Line2D([0], [0], color=BLUE, lw=2.4, label="ideal"),
               Line2D([0], [0], color=RED, lw=2.4, label="perturbed"),
               Line2D([0], [0], color="0.4", lw=2.0, ls="-", label="input"),
               Line2D([0], [0], color="0.4", lw=2.0, ls="--", label=r"$\hat U$ output")]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, ncol=1,
              framealpha=1.0, edgecolor="0.8", handlelength=1.6,
              labelspacing=0.3, borderpad=0.4,
              bbox_to_anchor=(-0.06, 1.0))


# ----------------------------------------------------------------------
def panel_recovery(ax):
    # MATCHES the paper's Fig. fig_retrieval (sim_qhnm.fig_retrieval): n=8, L=3
    # at the working-capacity edge, streamed MEMRISTIVE couplings (not textbook
    # Hebb), Omega0=1.0, paper noise grid (includes f=0.125), 40 trials -> so the
    # slide reproduces the paper's headline "91% vs 77% at f=0.125" by construction.
    n, L, trials = 8, 3, 40
    eta, gamma = 1.0 / n, 0.01
    patterns = q.RNG.choice([-1, 1], size=(L, n))
    J, _ = q.stream_memristor_couplings(patterns, eta=eta, gamma=gamma,
                                        dt=1.0, steps_per_pattern=1)
    spins = q.spin_table(n)
    noise = np.array([0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5])
    mq = np.zeros((len(noise), trials)); mc = np.zeros_like(mq); minp = np.zeros_like(mq)
    for ip, p in enumerate(noise):
        nflip = int(round(p * n))
        for tr in range(trials):
            mu = q.RNG.integers(L)
            xi = patterns[mu].copy()
            noisy = xi.copy()
            if nflip:
                idx = q.RNG.choice(n, size=nflip, replace=False)
                noisy[idx] *= -1
            minp[ip, tr] = q.overlap(noisy, xi)
            mc[ip, tr] = q.overlap(q.retrieve_classical(J, noisy), xi)
            mq[ip, tr] = q.overlap(q.retrieve_quantum_anneal(J, noisy, Omega0=1.0,
                                    spins=spins), xi)
    x = noise * 100
    ax.plot(x, minp.mean(1), color=GREY, ls=":", lw=1.6, label="input (no retrieval)")
    ax.plot(x, mc.mean(1), color=THIRD, lw=2.0, marker="s", ms=4.5,
            label=r"$\Omega=0$ classical")
    ax.plot(x, mq.mean(1), color=BLUE, lw=2.4, marker="o", ms=4.5,
            label=r"$\Omega>0$ quantum")
    ax.axhline(1.0, color="0.8", lw=0.8)
    ax.set_xlabel(r"input corruption (\% bits flipped)")
    ax.set_ylabel(r"recall overlap")
    ax.set_ylim(-0.05, 1.12); ax.set_xlim(0, 50)
    ax.legend(loc="lower left", fontsize=10, framealpha=1.0, facecolor="white",
              edgecolor="0.8", handlelength=1.7, labelspacing=0.3, borderpad=0.4)
    ax.set_title("Trained network: pattern completion", fontsize=13)
    print(f"  recovery: Omega>0 mean overlap @12.5% = {mq[noise==0.125].mean():.2f}"
          if (noise == 0.125).any() else "  (12.5% not on grid)")


def main():
    # --- standalone panels.  Panel A (used as the paper's column Fig. 15) is
    # widened so its on-page apparent font lands at ~9 pt at \columnwidth and the
    # Bloch circle fills the column; the tight limits in panel_bloch do the rest.
    figA, axA = plt.subplots(figsize=(4.6, 4.6))
    panel_bloch(axA); figA.tight_layout(pad=0.2)
    figA.savefig("figures/pres_robustness_a.pdf"); figA.savefig("figures/pres_robustness_a.png", dpi=150)

    figB, axB = plt.subplots(figsize=(4.0, 3.5))
    panel_recovery(axB); figB.tight_layout(pad=0.3)
    figB.savefig("figures/pres_robustness_b.pdf"); figB.savefig("figures/pres_robustness_b.png", dpi=150)

    # --- combined (backup/compat)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.5),
                               gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    panel_bloch(a); panel_recovery(b); fig.tight_layout()
    fig.savefig("figures/pres_robustness.pdf"); fig.savefig("figures/pres_robustness.png", dpi=150)
    print("saved figures/pres_robustness_a.pdf figures/pres_robustness_b.pdf figures/pres_robustness.pdf")


if __name__ == "__main__":
    main()
