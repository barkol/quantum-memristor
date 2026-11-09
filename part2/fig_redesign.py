#!/usr/bin/env python3
r"""Plotting layer of the manuscript figures: reads figures/data/ only, never
recomputes.  Fonts 6-8 pt at the printed size, Turbo palette, wide and low panels."""
import json, re
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import NullFormatter, FixedLocator
import paneltags

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "figures" / "data"
OUT = HERE.parent / "papers" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
WIDE, COL = 7.06, 3.375
T = colormaps["turbo"]
C1, C2, C3, C4 = T(0.12), T(0.35), T(0.70), T(0.88)
GREY = "0.45"
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{braket}\DeclareMathSizes{8}{8}{6}{6}\DeclareMathSizes{7}{7}{6}{6}\DeclareMathSizes{6.5}{6.5}{6}{6}\DeclareMathSizes{6}{6}{6}{6}",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "lines.linewidth": 1.0, "lines.markersize": 2.8, "legend.frameon": False,
    "legend.handlelength": 1.5, "legend.labelspacing": 0.2, "legend.borderpad": 0.1,
    "legend.handletextpad": 0.4, "savefig.dpi": 600, "pdf.fonttype": 42,
})


def save(fig, name):
    paneltags.place(fig)
    fig.savefig(OUT/name)
    plt.close(fig)
    print("wrote", OUT/name)


def ci(per, axis=0):
    return per.mean(axis), 1.96*per.std(axis, ddof=1)/np.sqrt(per.shape[axis])


# ------------------------------------------------------------------ concept
def fig_concept():
    A = json.loads((DATA/"fig_concept_anchors.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(WIDE, 1.85), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.005, wspace=0.0)
    box = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82)

    def lab(ax, W, H, xy, txt, dx=0, dy=0, **kw):
        ax.text(xy[0]*W+dx, (1-xy[1])*H+dy, txt, fontsize=7, ha=kw.pop("ha", "center"),
                va=kw.pop("va", "center"), bbox=box, **kw)

    heads = {"a": r"one configuration returned, $F\le 1/2$", "b": r"same bit flip on both branches, $F=1$",
             "c": r"different bit flips, $F=1/2$"}
    for ax, n, tagname in zip(axs, "abc", ("(a)", "(b)", "(c)")):
        im = plt.imread(DATA/f"fig_concept_{n}.png")
        al = np.clip((im[..., 3:]-0.03)/0.97, 0, 1)  # denoiser leaves a faint alpha haze on the empty background
        rgb = im[..., :3]*al + (1-al)
        ax.imshow(rgb, interpolation="none")
        H, W = rgb.shape[:2]
        ax.axis("off")
        ax.set_xlim(0, W); ax.set_ylim(0.91*H, 0.03*H)
        ax.set_title(heads[n], pad=1.5)
        paneltags.tag(ax, tagname)
        a = A[n]
        lab(ax, W, H, a["w1"], r"$c_1$" if n == "a" else r"$\ket{c_1}$", dx=-55, dy=-130)
        lab(ax, W, H, a["w2"], r"$c_2$" if n == "a" else r"$\ket{c_2}$", dx=55 if n == "a" else -55, dy=-130)
        if n == "a":
            lab(ax, W, H, a["in1"], r"$c_1\oplus e$", dy=125)
        elif n == "b":
            lab(ax, W, H, a["in1"], r"$\ket{c_1\oplus e}$", dy=125)
            lab(ax, W, H, a["in2"], r"$\ket{c_2\oplus e}$", dy=125, dx=-60)
            lab(ax, W, H, a["plate"], r"ancilla", dx=-40, dy=-62, ha="right")
            lab(ax, W, H, a["s1"], r"$\ket{e}$", dy=-100)
        else:
            lab(ax, W, H, a["in1"], r"$\ket{c_1\oplus e_1}$", dy=125, dx=60)
            lab(ax, W, H, a["in2"], r"$\ket{c_2\oplus e_2}$", dy=125, dx=-60)
            lab(ax, W, H, a["plate"], r"ancilla", dx=-40, dy=-62, ha="right")
            lab(ax, W, H, a["s1"], r"$\ket{e_1}$", dy=-100)
            lab(ax, W, H, a["s2"], r"$\ket{e_2}$", dy=-100)
    save(fig, "fig_concept.pdf")


# ------------------------------------------------------------------ network
def fig_network():
    fig, ax = plt.subplots(figsize=(COL, 1.28), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005)
    ax.set_xlim(-0.2, 11.3); ax.set_ylim(-0.1, 3.65); ax.set_aspect("equal"); ax.axis("off")
    # stream of polarization-encoded pattern photons
    for k, (x, s) in enumerate(zip((0.1, 0.7, 1.3, 1.9), "HVHH")):
        ax.add_patch(Circle((x, 1.9), 0.25, fc=T(0.70), ec="k", lw=0.5))
        ax.text(x, 1.9, rf"\textit{{{s}}}", ha="center", va="center", fontsize=6)
    ax.text(1.0, 2.35, r"pattern $\xi^\mu$", ha="center", va="bottom", fontsize=7)
    ax.add_patch(FancyArrowPatch((2.3, 1.9), (3.2, 1.9), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=C1))
    # register: all pairs carry a memristor
    P = [(3.75, 3.05), (5.75, 3.05), (5.75, 0.75), (3.75, 0.75)]
    names = "ijkl"
    for a in range(4):
        for b in range(a+1, 4):
            ax.plot(*zip(P[a], P[b]), color=C1, lw=0.8, zorder=1)
            m = np.add(P[a], P[b])/2 + (np.array((0.28, -0.28)) if (a, b) == (0, 2) else np.array((-0.28, -0.28)) if (a, b) == (1, 3) else 0)
            ax.add_patch(FancyBboxPatch(m-0.17, 0.34, 0.34, boxstyle="round,pad=0.02", fc=C2, ec="k", lw=0.4, zorder=2))
    for p, s in zip(P, names):
        ax.add_patch(Circle(p, 0.3, fc="white", ec="k", lw=0.6, zorder=3))
        ax.text(*p, rf"${s}$", ha="center", va="center", fontsize=7, zorder=4)
    ax.add_patch(FancyBboxPatch((3.27, 0.33), 2.96, 3.14, boxstyle="round,pad=0.02", fc="none", ec=GREY, lw=0.6, ls=(0, (2, 2)), zorder=1))
    # detectors and classical feedback
    ax.add_patch(FancyArrowPatch((6.3, 1.9), (7.05, 1.9), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=C1))
    ax.add_patch(FancyBboxPatch((7.15, 1.3), 3.95, 1.2, boxstyle="round,pad=0.05", fc="white", ec="k", lw=0.6))
    ax.text(9.12, 2.13, r"coincidences", ha="center", va="center", fontsize=7)
    ax.text(9.12, 1.65, r"$N_{ij}=N^{\rm same}_{ij}-N^{\rm opp}_{ij}$", ha="center", va="center", fontsize=6.5)
    ax.add_patch(FancyArrowPatch((9.12, 1.22), (4.75, 0.3), connectionstyle="arc3,rad=-0.30", arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, ls=(0, (3, 2)), color=C4, zorder=5))
    ax.text(9.6, 0.62, r"$\theta_{ij}\to\phi_{ij}\to J_{ij}$", ha="center", va="center", fontsize=7, color=C4)
    save(fig, "memristor_network.pdf")


# ------------------------------------------------------------------ network results
def fig_qhnm():
    fig, axs = plt.subplots(1, 4, figsize=(WIDE, 1.72), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.005, wspace=0.015)
    axA, axB, axC, axD = axs
    # (a)
    d = np.load(DATA/"fix_panel_a_shotnoise.npz")
    G, mr, sr = d["budgets"].astype(float), d["mean_r"], d["std_r"]
    axA.axhline(float(d["r_det"]), ls="--", color=GREY, lw=0.8, label=r"mean field")
    axA.fill_between(G, mr-sr, mr+sr, color=C1, alpha=0.2, lw=0)
    axA.semilogx(G, mr, "-o", color=C1, label=r"sampled")
    axA.set_xlabel(r"budget $\Gamma_{\rm c}\Delta t$"); axA.set_ylabel(r"correlation $r$")
    axA.set_ylim(0.55, 1.04); axA.set_yticks([0.6, 0.8, 1]); axA.set_yticklabels(["0.6", "0.8", "1"])
    axA.set_xticks([1, 10, 100, 1000]); axA.set_xticklabels(["1", "10", "100", "1000"])
    axA.xaxis.set_minor_formatter(NullFormatter())
    axA.legend(loc="lower right"); paneltags.tag(axA, "(a)")
    # (b)
    b = np.load(DATA/"fix_panel_b_big.npz"); nz = b["noise"]
    series = ((b["per_greedy_bias"], C3, "-D", r"greedy, biased"),
              (b["per_sa"], C2, "-^", r"SA"),
              (b["per_q"], C1, "-o", r"quantum anneal"),
              (b["per_greedy"], C4, "-s", r"greedy, unbiased"))
    for per, col, mk, lab in series:
        mu, cc = ci(per)
        axB.fill_between(nz, mu-cc, mu+cc, color=col, alpha=0.18, lw=0)
    for per, col, mk, lab in series:
        axB.plot(nz, per.mean(0), mk, color=col, ms=2.0, lw=0.9, label=lab)
    axB.set_xlabel(r"input noise $f$"); axB.set_ylabel(r"recall overlap $m$")
    axB.set_ylim(0.15, 1.04); axB.set_yticks([0.2, 0.4, 0.6, 0.8, 1]); axB.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1"])
    axB.legend(loc="lower left", fontsize=6); paneltags.tag(axB, "(b)")
    # (c)
    c = np.load(DATA/"fix_panel_c_capacity.npz")
    cols = {6: C1, 8: C3, 10: C4}
    for n in (6, 8, 10):
        Ls, s, e = c[f"Ls_{n}"], c[f"succ_{n}"], c[f"sci_{n}"]
        axC.fill_between(Ls, s-e, s+e, color=cols[n], alpha=0.18, lw=0)
        axC.plot(Ls, s, "-o", color=cols[n], label=rf"$n={n}$")
        axC.axvline(0.14*n, color=cols[n], ls=":", lw=0.8)
    axC.set_xlabel(r"stored patterns $L$"); axC.set_ylabel(r"recall success")
    axC.set_ylim(-0.03, 1.05); axC.set_yticks([0, 0.5, 1]); axC.set_yticklabels(["0", "0.5", "1"])
    axC.legend(loc="upper right"); paneltags.tag(axC, "(c)")
    # (d) lollipops instead of bars
    p = np.load(DATA/"fix_panel_d_palimpsest.npz")
    mf, cf = ci(p["per_fade"]); mn, cn = ci(p["per_nofade"])
    ages = np.arange(len(mf)); g = float(p["gamma_fade"])
    for off, m, e, col, mk, lab in ((-0.17, mf, cf, C1, "o", rf"$\gamma={g:g}$"),
                                    (0.17, mn, cn, C4, "s", r"$\gamma\to0$")):
        axD.vlines(ages+off, 0, m, color=col, lw=0.7)
        axD.errorbar(ages+off, m, yerr=e, fmt=mk, color=col, ms=2.0, elinewidth=0.5, capsize=0, label=lab)
    axD.plot(ages, np.exp(-g*ages), color=GREY, lw=0.8, ls="--", label=r"$e^{-\gamma\,\mathrm{age}}$")
    axD.set_xlabel(r"pattern age"); axD.set_ylabel(r"retrievability $R$")
    axD.set_xlim(-0.7, 10.5); axD.set_ylim(0, 1.05)
    axD.set_yticks([0, 0.5, 1]); axD.set_yticklabels(["0", "0.5", "1"])
    axD.legend(loc="upper right"); paneltags.tag(axD, "(d)")
    save(fig, "paper_qhnm.pdf")


# ------------------------------------------------------------------ capacity
def fig_dense():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL, 1.62), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.005, wspace=0.01)
    d = np.load(DATA/"fix_dense_bootstrap.npz")
    n2, n3, R2, R3 = d["n2"], d["n3"], d["R2"], d["R3"]
    for nn, R, col, mk, lab in ((n3, R3, C4, "s", r"3-body"), (n2, R2, C1, "o", r"2-body")):
        c = R.mean(1); e = R.std(1, ddof=1)/np.sqrt(R.shape[1])
        fp = np.polyfit(np.log(nn), np.log(c), 1)
        xx = np.array([nn.min(), nn.max()])
        ax.errorbar(nn, c, yerr=e, fmt=mk, color=col, ms=2.4, elinewidth=0.6, label=lab)
        ax.plot(xx, np.exp(np.polyval(fp, np.log(xx))), "--", color=col, lw=0.7)
    ax.plot(n2, 0.14*n2, ":", color=GREY, lw=0.9, label=r"$0.14\,n$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"network size $n$"); ax.set_ylabel(r"capacity $L_{\max}$")
    ax.xaxis.set_major_locator(FixedLocator([8, 16, 32])); ax.set_xticklabels(["8", "16", "32"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(1, 60)
    ax.legend(loc="upper left"); paneltags.tag(ax, "(a)")
    t = np.load(DATA/"fig_capacity_code.npz")
    n = t["n"]
    bx.errorbar(n, t["mean_rand3"], yerr=t["sd_rand3"], fmt="s-", color=C4, ms=2.4, lw=0.8, elinewidth=0.6, label=r"random, 3-body")
    bx.plot(n, n-1, ":", color=GREY, lw=0.9, label=r"$n-1$")
    bx.errorbar(n, t["mean_code2"], yerr=t["sd_code2"], fmt="D-", color=C3, ms=2.4, lw=0.8, elinewidth=0.6, label=r"Hadamard, 2-body")
    bx.errorbar(n, t["mean_rand2"], yerr=t["sd_rand2"], fmt="o-", color=C1, ms=2.4, lw=0.8, elinewidth=0.6, label=r"random, 2-body")
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xlabel(r"network size $n$"); bx.set_ylabel(r"exact-recovery capacity")
    bx.xaxis.set_major_locator(FixedLocator([64, 128, 256])); bx.set_xticklabels(["64", "128", "256"])
    bx.xaxis.set_minor_formatter(NullFormatter())
    bx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    bx.set_ylim(4, 1.5e4)
    bx.legend(loc="upper left", fontsize=6); paneltags.tag(bx, "(b)")
    save(fig, "paper_dense_capacity.pdf")


# ------------------------------------------------------------------ reservoir
def fig_qrc():
    d = np.load(DATA/"fig_qrc_gamma.npz")
    g, S = d["gammas"], d["S"]
    m, s = S.mean(0), S.std(0, ddof=1)
    fig, ax = plt.subplots(figsize=(COL, 1.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.005)
    ax.fill_between(g, m-s, m+s, color=C1, alpha=0.2, lw=0)
    ax.semilogx(g, m, "-o", color=C1, label=r"reservoir")
    ax.plot([g[m.argmin()]], [m.min()], "*", color=C4, ms=6, zorder=5)
    ax.axhline(float(d["lin10"].mean()), color=C3, ls="--", lw=0.8, label=r"linear filter, 10 inputs")
    ax.axhline(float(d["lin27"].mean()), color=C2, ls="-.", lw=0.8, label=r"linear filter, 27 inputs")
    ax.set_xlabel(r"forgetting rate $\gamma$"); ax.set_ylabel(r"test NMSE")
    ax.set_xticks([0.01, 0.1, 1]); ax.set_xticklabels(["0.01", "0.1", "1"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.legend(loc="lower right", fontsize=6)
    save(fig, "paper_qrc_gamma.pdf")


def fig_qrc_part1():
    """Reservoir figure of Part I, supplement."""
    d = np.load(DATA/"fig_qrc_gamma.npz"); e = np.load(DATA/"fig_qrc_example.npz")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL, 1.45), layout="constrained", width_ratios=[1.35, 1])
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.005, wspace=0.01)
    tt = np.arange(len(e["yt"]))[:150]
    ax.plot(tt, e["yt"][:150], color=C1, lw=0.9, label=r"target")
    ax.plot(tt, e["yp"][:150], color=C4, lw=0.7, ls="--", label=r"prediction")
    ax.set_xlabel(r"test step"); ax.set_ylabel(r"NARMA-10 output")
    ax.legend(loc="upper right", ncol=2, fontsize=6); paneltags.tag(ax, "(a)")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi+0.25*(hi-lo))
    names = (("res", "reservoir", C4), ("lin10", "filter 10", C3), ("esn", "ESN", C2), ("lin27", "filter 27", C1))
    for i, (k, lab, col) in enumerate(names):
        v = d[k]
        bx.hlines(i, 0, v.mean(), color=col, lw=0.8)
        bx.errorbar(v.mean(), i, xerr=v.std(ddof=1), fmt="o", color=col, ms=3, elinewidth=0.6)
    bx.set_yticks(range(4)); bx.set_yticklabels([n[1] for n in names]); bx.invert_yaxis()
    bx.set_xlabel(r"test NMSE"); bx.set_xlim(0, 0.6); bx.set_xticks([0, 0.2, 0.4, 0.6]); bx.set_xticklabels(["0", "0.2", "0.4", "0.6"])
    paneltags.tag(bx, "(b)")
    save(fig, "paper_qrc.pdf")


if __name__ == "__main__":
    fig_concept(); fig_network(); fig_qhnm(); fig_dense(); fig_qrc(); fig_qrc_part1()
    print("figures done")
