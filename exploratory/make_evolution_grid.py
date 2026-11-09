r"""Two-row time-evolution figure (Fig. S2):
  (a) input drive v(t) and output response i(t)=G(theta)v
  (b) memory/HWP angle theta(t) with the (scaled) drive overlaid
Both panels mark the transient -> settled (periodic limit-cycle) transition:
the memory relaxes on ~1/gamma and the response locks into a repeating
waveform after about one drive period T=2*pi/omega.

Model from quantum_memristor_phase_presentation.py (solve_theta, drive,
transmission); same parameters as the fingerprint figure.
Data in figures/data/make_evolution_grid.npz (computed if missing); the
drawing layer only reads it.  Drawn at its final size (\columnwidth = 3.40 in).
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = Path(os.environ.get("FIG_DATA_DIR", REPO / "figures" / "data")) / "make_evolution_grid.npz"
OUT = REPO / "papers" / "figures" / "paper_evolution.pdf"


def compute():
    sys.path.insert(0, str(HERE))
    import quantum_memristor_phase_presentation as qm
    t, theta = qm.solve_theta(n_cycles=4)
    v = qm.drive(t)
    i = qm.transmission(theta) * v
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    # model constants stored with the data so that drawing needs no model import
    np.savez(CACHE, t=t, theta=theta, v=v, i=i, OMEGA=qm.OMEGA, THETA0=qm.THETA0)


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    sys.path.insert(0, str(HERE))
    from paneltags import tag, place
    plt.rcdefaults()   # model modules may have changed rcParams (e.g. savefig.bbox)
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 7, "axes.labelsize": 7,
                         "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5,
                         "lines.linewidth": 0.9, "lines.markersize": 2.8, "legend.frameon": False,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    d = np.load(CACHE, allow_pickle=False)
    t, theta, v, i = d["t"], d["theta"], d["v"], d["i"]
    if "OMEGA" in d.files:
        OMEGA, THETA0 = float(d["OMEGA"]), float(d["THETA0"])
    else:   # caches written by the previous version of this script
        OMEGA, THETA0 = 0.30, np.pi / 8
    T = 2.0 * np.pi / OMEGA
    tmax = float(t[-1])
    blue, grey = plt.cm.turbo(0.12), "0.45"
    fmt = FuncFormatter(lambda x, _: rf"${x:g}$")

    fig, (aA, aB) = plt.subplots(2, 1, figsize=(3.40, 1.85), sharex=True, layout="constrained")
    for ax in (aA, aB):
        ax.axvspan(0, T, color="0.92", lw=0, zorder=0)
        ax.axvline(T, color="0.5", ls=":", lw=0.6, zorder=1)
        ax.set_xlim(0, tmax)

    # (a) drive and output; regime labels in a clear band below the data, legend in a band above
    aA.axhline(0, color="0.75", lw=0.4, zorder=0)
    aA.plot(t, v, "--", color=grey, lw=0.9, label=r"input drive $v(t)$")
    aA.plot(t, i, color=blue, lw=1.0, label=r"output $i(t)$")
    aA.set_ylim(-1.75, 1.6)
    aA.yaxis.set_major_locator(FixedLocator([-1, 0, 1]))
    aA.yaxis.set_major_formatter(fmt)
    aA.set_ylabel(r"signal")
    xt = aA.get_xaxis_transform()
    aA.text(0.5 * T, 0.05, "transient", transform=xt, ha="center", va="bottom", fontsize=6)
    aA.text(0.5 * (T + tmax), 0.05, "settled (periodic)", transform=xt, ha="center", va="bottom", fontsize=6)
    aA.legend(loc="upper right", ncol=2, handlelength=1.6, handletextpad=0.4, columnspacing=1.0,
              borderaxespad=0.15, borderpad=0.1)
    tag(aA, "(a)")

    # (b) memory angle theta(t) with the scaled drive and the rest angle theta0
    settled = t >= 0.5 * T
    amp = 0.5 * (theta[settled].max() - theta[settled].min())
    aB.plot(t, THETA0 + amp * v, color=grey, lw=0.9, ls="--", label=r"drive $\propto\sin\omega t$")
    aB.plot(t, theta, color=blue, lw=1.0, label=r"$\theta(t)$")
    aB.axhline(THETA0, color="0.5", lw=0.5, ls=":")
    aB.set_ylim(-0.25, 1.55)
    aB.yaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
    aB.yaxis.set_major_formatter(fmt)
    aB.set_ylabel(r"$\theta(t)$ (rad)")
    aB.set_xlabel(r"time $t$")
    aB.xaxis.set_major_formatter(fmt)
    aB.legend(loc="upper right", ncol=2, handlelength=1.6, handletextpad=0.4, columnspacing=1.0,
              borderaxespad=0.15, borderpad=0.1)
    tag(aB, "(b)")
    fig.align_ylabels((aA, aB))
    place(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    if not CACHE.exists():
        compute()
    draw()
