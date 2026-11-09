r"""2x2 "memristive fingerprint" figure (Fig. S1): the pinched hysteresis loop
plus the three single-parameter sweeps (drive frequency, feedback strength,
relaxation rate).  Model and parameters from
quantum_memristor_phase_presentation.py (loop_last_cycles, memoryless_line);
no physics changed.

  (a) pinched hysteresis loop vs the memoryless line
  (b) frequency sweep          omega in {0.10,0.30,1.0,3.0}
  (c) feedback-strength sweep  eta   in {0.05,..,0.30}
  (d) relaxation-rate sweep    gamma in {0.03,..,1.0}

Data in figures/data/make_fingerprint_grid.npz (computed if missing); the
drawing layer only reads it.  Drawn at its final size (\columnwidth = 3.40 in).
"""
import os
import sys
from pathlib import Path

import numpy as np

OMEGAS = [0.10, 0.30, 1.0, 3.0]
ETAS = [0.05, 0.10, 0.15, 0.20, 0.30]
GAMMAS = [0.03, 0.10, 0.30, 1.0]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = Path(os.environ.get("FIG_DATA_DIR", REPO / "figures" / "data")) / "make_fingerprint_grid.npz"
OUT = REPO / "papers" / "figures" / "paper_fingerprint.pdf"


def compute():
    sys.path.insert(0, str(HERE))
    import quantum_memristor_phase_presentation as qm
    v, i = qm.loop_last_cycles()
    vv = np.linspace(-qm.V0, qm.V0, 200)
    ml = qm.memoryless_line(vv)
    freq_loops = [qm.loop_last_cycles(omega=w, n_cycles=10, skip=7) for w in OMEGAS]
    eta_loops = [qm.loop_last_cycles(eta=e) for e in ETAS]
    gamma_loops = [qm.loop_last_cycles(gamma=g, n_cycles=10, skip=7) for g in GAMMAS]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, v=v, i=i, vv=vv, ml=ml,
             **{f"fv_{k}": a for k, (a, _) in enumerate(freq_loops)},
             **{f"fi_{k}": b for k, (_, b) in enumerate(freq_loops)},
             **{f"ev_{k}": a for k, (a, _) in enumerate(eta_loops)},
             **{f"ei_{k}": b for k, (_, b) in enumerate(eta_loops)},
             **{f"gv_{k}": a for k, (a, _) in enumerate(gamma_loops)},
             **{f"gi_{k}": b for k, (_, b) in enumerate(gamma_loops)})


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
    fam = lambda n: plt.cm.turbo(np.linspace(0.12, 0.88, n))
    fig, axs = plt.subplots(2, 2, figsize=(3.40, 2.55), sharex=True, sharey=True, layout="constrained")
    aA, aB, aC, aD = axs.flat
    leg = dict(loc="upper left", handlelength=1.2, handletextpad=0.4, labelspacing=0.15,
               borderaxespad=0.2, borderpad=0.1)

    # (a) pinched loop and memoryless line
    aA.plot(d["vv"], d["ml"], color=plt.cm.turbo(0.88), lw=0.8, ls="--", label=r"no memory")
    v, i = d["v"], d["i"]
    aA.plot(v, i, color=plt.cm.turbo(0.12), lw=1.0, label=r"memristive")
    for frac in (0.20, 0.70):
        k = int(frac * (len(v) - 2))
        aA.annotate("", xy=(v[k + 1], i[k + 1]), xytext=(v[k], i[k]),
                    arrowprops=dict(arrowstyle="-|>", color=plt.cm.turbo(0.12), lw=0.8, mutation_scale=6))
    aA.plot(0.0, 0.0, "o", color="k", ms=2.5, zorder=5)
    aA.annotate("pinched", xy=(0.0, 0.0), xytext=(0.12, -0.45), fontsize=6)
    h, l = aA.get_legend_handles_labels()
    aA.legend(h[::-1], l[::-1], **leg)
    tag(aA, "(a)")

    for ax, key, vals, sym, lab in ((aB, "f", OMEGAS, r"\omega", "(b)"),
                                    (aC, "e", ETAS, r"\eta", "(c)"),
                                    (aD, "g", GAMMAS, r"\gamma", "(d)")):
        for k, (val, c) in enumerate(zip(vals, fam(len(vals)))):
            ax.plot(d[f"{key}v_{k}"], d[f"{key}i_{k}"], color=c, lw=0.9, label=rf"${sym}={val:g}$")
        # the quadrant v > 0, i < 0 is empty for every loop (i = G v with G >= 0): legend goes there
        ax.legend(**{**leg, "loc": "lower right"})
        tag(ax, lab)

    fmt = FuncFormatter(lambda x, _: rf"${x:g}$")
    for ax in axs.flat:
        ax.axhline(0.0, color="0.75", lw=0.4, zorder=0)
        ax.axvline(0.0, color="0.75", lw=0.4, zorder=0)
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_locator(FixedLocator([-1, 0, 1]))
            a.set_major_formatter(fmt)
    for ax in (aA, aC):
        ax.set_ylabel(r"Output $i$")
    for ax in (aC, aD):
        ax.set_xlabel(r"Input drive $v$")
    place(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    if not CACHE.exists():
        compute()
    draw()
