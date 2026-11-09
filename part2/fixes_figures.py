#!/usr/bin/env python3
r"""Plotting layer for the audit fixes: reads figures/data/*.npz only, never
recomputes.  Style follows make_paper_figures.py / figstyle.py."""
import os
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import figstyle

figstyle.apply()
BLUE, RED, THIRD, GREY = figstyle.BLUE, figstyle.RED, figstyle.THIRD, figstyle.GREY
GREEN = "#009E73"                     # Wong colour-blind-safe 4th series
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "figures" / "data"
OUT = HERE.parent / "papers" / "figures"; OUT.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    figstyle.cap_ticks(fig)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    assert p.stat().st_size > 0
    print(f"wrote {p} ({p.stat().st_size} bytes)")

def ci(per, axis=0):
    return per.mean(axis), 1.96*per.std(axis, ddof=1)/np.sqrt(per.shape[axis])

# ================================================================ Fig. 2
def fig_qhnm():
    figstyle.apply(base_fontsize=13)
    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=figstyle.GRID2x2)

    # -- (a) Hebbian correlation vs coincidence budget (shot noise restored)
    d = np.load(DATA/"fix_panel_a_shotnoise.npz")
    G, mr, sr = d["budgets"].astype(float), d["mean_r"], d["std_r"]
    axA.axhline(float(d["r_det"]), ls="--", color=GREY, lw=1.4,
                label=r"mean-field drive")
    axA.fill_between(G, mr-sr, mr+sr, color=BLUE, alpha=0.18, lw=0)
    axA.semilogx(G, mr, "-o", color=BLUE, ms=5, lw=2, label=r"sampled coincidences")
    axA.set_xlabel(r"coincidence budget $\Gamma_{\rm c}\Delta t$")
    axA.set_ylabel(r"correlation $r$ with $J^{\rm Hebb}$")
    axA.set_ylim(0.55, 1.04)
    axA.legend(loc="lower right", handlelength=1.6,
               fontsize=0.72*plt.rcParams["font.size"])
    figstyle.panel(axA, "(a)")

    # -- (b) recall overlap, four rules, CI across disorder realisations
    f = DATA/"fix_panel_b_big.npz"
    b = np.load(f if f.exists() else DATA/"fix_panel_b_disorder.npz")
    nz = b["noise"]
    series = ((b["per_greedy_bias"], GREEN, "-D", r"$\Omega=0$ greedy, biased"),
              (b["per_sa"],          THIRD, "-^", r"classical SA (biased)"),
              (b["per_q"],           BLUE,  "-o", r"$\Omega>0$ quantum anneal"),
              (b["per_greedy"],      RED,   "-s", r"$\Omega=0$ greedy, unbiased"))
    for per, col, mk, lab in series:
        mu, cc = ci(per)
        axB.fill_between(nz, mu-cc, mu+cc, color=col, alpha=0.15, lw=0)
    for per, col, mk, lab in series:
        mu, _ = ci(per)
        axB.plot(nz, mu, mk, color=col, ms=4.5, lw=2, label=lab)
    axB.set_xlabel(r"input noise fraction $f$")
    axB.set_ylabel(r"recall overlap")
    axB.set_ylim(0.15, 1.04)
    axB.legend(loc="lower left", handlelength=1.5,
               fontsize=0.63*plt.rcParams["font.size"])
    figstyle.panel(axB, "(b)")

    # -- (c) capacity vs L (unchanged protocol)
    c = np.load(DATA/"fix_panel_c_capacity.npz")
    cols = {6: BLUE, 8: RED, 10: THIRD}
    for n in (6, 8, 10):
        Ls, s, e = c[f"Ls_{n}"], c[f"succ_{n}"], c[f"sci_{n}"]
        axC.fill_between(Ls, s-e, s+e, color=cols[n], alpha=0.15, lw=0)
        axC.plot(Ls, s, "-o", color=cols[n], ms=5, lw=2, label=rf"$n={n}$")
        axC.axvline(0.14*n, color=cols[n], ls=":", lw=1.2, alpha=0.9)
    axC.text(0.97, 0.55, r"$0.14\,n$ bound", transform=axC.transAxes, ha="right",
             va="center", color=GREY, fontsize=0.72*plt.rcParams["font.size"])
    axC.set_xlabel(r"stored patterns $L$"); axC.set_ylabel(r"recall success")
    axC.set_ylim(-0.03, 1.05)
    axC.legend(loc="lower left", ncol=3, columnspacing=0.8, handletextpad=0.3,
               handlelength=1.2)
    figstyle.panel(axC, "(c)")

    # -- (d) palimpsest, with the gamma->0 control
    p = np.load(DATA/"fix_panel_d_palimpsest.npz")
    mf, cf = ci(p["per_fade"]); mn, cn = ci(p["per_nofade"])
    ages = np.arange(len(mf)); g = float(p["gamma_fade"])
    axD.bar(ages-0.20, mf, width=0.40, color=BLUE, alpha=0.60,
            label=rf"$\gamma={g:g}$")
    axD.bar(ages+0.20, mn, width=0.40, facecolor="none", edgecolor=RED, lw=1.1,
            label=r"$\gamma\to0$ (no forgetting)")
    axD.errorbar(ages-0.20, mf, yerr=cf, fmt="none", ecolor="0.25",
                 elinewidth=0.8, capsize=1.5, zorder=5)
    axD.errorbar(ages+0.20, mn, yerr=cn, fmt="none", ecolor=RED,
                 elinewidth=0.8, capsize=1.5, zorder=5)
    axD.plot(ages, np.exp(-g*ages), color=GREY, lw=1.6, ls="--",
             label=r"$e^{-\gamma\,\mathrm{age}}$")
    axD.axhline(0.5, color=GREY, ls=":", lw=1.0)
    axD.set_xlabel(r"pattern age $(T-t_\mu)/\Delta t$")
    axD.set_ylabel(r"retrievability"); axD.set_ylim(0, 1.05)
    axD.set_xlim(-0.7, len(ages)-0.3)
    axD.legend(loc="upper right", handlelength=1.5,
               fontsize=0.68*plt.rcParams["font.size"])
    figstyle.panel(axD, "(d)")

    fig.tight_layout(); save(fig, "paper_qhnm.pdf")

# ================================================================ Fig. 3
def fig_dense():
    figstyle.apply()
    d = np.load(DATA/"fix_dense_bootstrap.npz")
    n2, n3, R2, R3 = d["n2"], d["n3"], d["R2"], d["R3"]
    c2, c3 = R2.mean(1), R3.mean(1)
    e2 = R2.std(1, ddof=1)/np.sqrt(R2.shape[1]); e3 = R3.std(1, ddof=1)/np.sqrt(R3.shape[1])
    s2 = np.polyfit(np.log(n2), np.log(c2), 1)[0]
    s3 = np.polyfit(np.log(n3), np.log(c3), 1)[0]
    rb = np.random.default_rng(999); b2, b3 = [], []
    for _ in range(2000):
        i = rb.integers(0, R2.shape[1], R2.shape[1]); b2.append(np.polyfit(np.log(n2), np.log(R2[:, i].mean(1)), 1)[0])
        i = rb.integers(0, R3.shape[1], R3.shape[1]); b3.append(np.polyfit(np.log(n3), np.log(R3[:, i].mean(1)), 1)[0])
    l2, h2 = np.percentile(b2, [2.5, 97.5]); l3, h3 = np.percentile(b3, [2.5, 97.5])
    print(f"  slopes p2={s2:.3f} [{l2:.3f},{h2:.3f}]  p3={s3:.3f} [{l3:.3f},{h3:.3f}]")
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)
    ax.errorbar(n2, c2, yerr=e2, fmt="o", color=BLUE, ms=8, capsize=3,
                label=rf"2-body (CZ): ${s2:.2f}\,[{l2:.2f},{h2:.2f}]$")
    ax.errorbar(n3, c3, yerr=e3, fmt="s", color=RED, ms=8, capsize=3,
                label=rf"3-body (CCZ): ${s3:.2f}\,[{l3:.2f},{h3:.2f}]$")
    ax.loglog(n2, 0.14*n2, ":", color=GREY, lw=1.6, label=r"$0.14\,n$ (classical bound)")
    for nn, cc, col in ((n2, c2, BLUE), (n3, c3, RED)):
        fp = np.polyfit(np.log(nn), np.log(cc), 1)
        xx = np.linspace(nn.min(), nn.max(), 40)
        ax.loglog(xx, np.exp(np.polyval(fp, np.log(xx))), "--", color=col, lw=1.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"network size $n$"); ax.set_ylabel(r"storage capacity $L_{\max}$")
    ax.legend(loc="upper left", fontsize=0.72*plt.rcParams["font.size"])
    ax.set_xticks([8, 12, 16, 20, 24, 32]); ax.set_xticklabels(["8","12","16","20","24","32"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout(); save(fig, "paper_dense_capacity.pdf")

if __name__ == "__main__":
    fig_dense(); fig_qhnm(); print("figures done")
