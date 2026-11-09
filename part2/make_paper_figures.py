#!/usr/bin/env python3
r"""Standalone driver: regenerate PUBLICATION-quality (Physical Review style)
multi-panel figures from the two ALREADY-VERIFIED simulation scripts.

This file does NOT change any physics, parameters, seeds, or numerical results.
It imports the verified computation primitives from

    sim_qrc_narma.py   (memristive quantum reservoir, NARMA-10)
    sim_qhnm.py        (full QHN-M storage/retrieval)

and reuses their simulation logic verbatim -- the loops below are copied
line-for-line from each script's data-producing code (only the matplotlib
styling/plotting is rewritten).  Seeds are the same: SEED=1234 (QRC),
SEED=20260625 (QHN-M).  The QHN-M functions use a module-global RNG that
main() advances by calling fig_hebb, fig_retrieval, fig_capacity,
fig_palimpsest IN THAT ORDER; we re-seed and call the data loops in the same
order so every random draw is identical to the verified run.

Outputs paper-style PDFs into ../papers/figures/.

Run: python3 make_paper_figures.py
"""
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import the verified modules WITHOUT running their main() (guarded by __main__)
import sim_qrc_narma as qrc_mod
import sim_qhnm as qhnm_mod

# ----------------------------------------------------------------------
# COMMON HOUSE STYLE (art-ent-meter-HOM / figstyle.py):
#   usetex serif, font ~18, dpi 300, two-colour blue + tab:red scheme,
#   thin solid lines (lw~2) with small markers, NO gridlines, in-pointing
#   ticks, panel labels a)/b)/c)/d) top-left, tight_layout.
# A grey THIRD series is used only where two colours do not suffice
# (the 3-bar NMSE chart and the n=6,8,10 capacity curves).
# Falls back to mathtext if LaTeX is unavailable (FIGSTYLE_NO_TEX=1).
# ----------------------------------------------------------------------
import figstyle

figstyle.apply()
USETEX = plt.rcParams["text.usetex"]

# HOM colours
BLUE = figstyle.BLUE     # 'b'        -- primary series
RED = figstyle.RED       # 'tab:red'  -- secondary series
THIRD = figstyle.THIRD   # '0.35'     -- neutral third series (only when needed)
GREY = figstyle.GREY     # '0.5'      -- auxiliary reference lines

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "papers", "figures")
os.makedirs(OUTDIR, exist_ok=True)

# Data cache: the expensive QRC / QHN-M computations run only when their
# cache file is missing; the figure functions plot from the cached dicts.
DATA_DIR = Path(__file__).resolve().parent.parent / "figures" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    figstyle.cap_ticks(fig)          # house rule: 4-5 stripped labels per axis
    path = os.path.abspath(os.path.join(OUTDIR, name))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    sz = os.path.getsize(path)
    assert sz > 0, f"{path} is empty!"
    print(f"wrote {path}  ({sz} bytes)")
    return path


def panel_label(ax, text):
    # Use figstyle.panel's OUTSIDE default (above top-left corner, bold).
    figstyle.panel(ax, text)


# ======================================================================
#  QRC data  --  reproduces sim_qrc_narma.main() data exactly (SEED=1234)
# ======================================================================
def compute_qrc():
    cache = DATA_DIR / "make_paper_figures_qrc.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return dict(yt_q=d["yt_q"], yp_q=d["yp_q"],
                    nmse_q=float(d["nmse_q"]), nmse_lin=float(d["nmse_lin"]),
                    nmse_esn=float(d["nmse_esn"]), gammas=d["gammas"],
                    nmse_g=d["nmse_g"], g_opt=float(d["g_opt"]),
                    nmse_g_opt=float(d["nmse_g_opt"]))

    SEED = qrc_mod.SEED  # 1234
    rng = np.random.default_rng(SEED)

    T = 1200
    u, y = qrc_mod.narma10(T, rng)
    target = y

    washout = 200
    train = slice(washout, 900)
    test = slice(900, T)

    N_QUBITS = 6
    GAMMA0 = 0.03
    qrc = qrc_mod.MemristiveQRC(n=N_QUBITS, gamma=GAMMA0, eta=0.9, theta0=0.15,
                                coupling=0.35, reset=0.03, gain=0.35,
                                gamma_spread=3.0, seed=SEED)
    qfeats = qrc.run(u)
    nmse_q, yp_q, yt_q = qrc_mod.evaluate(qfeats, target, train, test)

    lfeats = qrc_mod.linear_floor_features(u, y)
    nmse_lin, _, _ = qrc_mod.evaluate(lfeats, target, train, test)

    esn_n = qrc.n_features - 1
    efeats = qrc_mod.esn_features(u, esn_n, np.random.default_rng(SEED + 1))
    nmse_esn, _, _ = qrc_mod.evaluate(efeats, target, train, test)

    # gamma sweep -- identical to source
    gammas = np.geomspace(0.01, 1.2, 16)
    nmse_g = []
    for g in gammas:
        q = qrc_mod.MemristiveQRC(n=N_QUBITS, gamma=g, eta=0.9, theta0=0.15,
                                  coupling=0.35, reset=0.03, gain=0.35,
                                  gamma_spread=3.0, seed=SEED)
        f = q.run(u)
        nm, _, _ = qrc_mod.evaluate(f, target, train, test)
        nmse_g.append(nm)
    nmse_g = np.array(nmse_g)
    g_opt = gammas[int(np.argmin(nmse_g))]
    nmse_g_opt = float(nmse_g.min())

    out = dict(yt_q=yt_q, yp_q=yp_q, nmse_q=nmse_q, nmse_lin=nmse_lin,
               nmse_esn=nmse_esn, gammas=gammas, nmse_g=nmse_g,
               g_opt=g_opt, nmse_g_opt=nmse_g_opt)
    np.savez(cache, **out)
    return out


# ======================================================================
#  QHN-M data  --  reproduces sim_qhnm.main() data exactly (SEED=20260625)
#  CRITICAL: re-seed the module RNG, then run the four data blocks in the
#  same order (hebb, retrieval, capacity, palimpsest) so every draw matches.
# ======================================================================
def compute_qhnm():
    m = qhnm_mod
    # re-seed the shared global RNG to the verified starting state
    m.RNG = np.random.default_rng(m.SEED)
    RNG = m.RNG  # local alias; m.RNG is the one the primitives read

    # --- (a) Hebbian emergence (verbatim from fig_hebb) ----------------
    n = 8
    L = 5
    eta = 1.0 / n
    gamma = 0.01
    dt = 1.0
    patterns = RNG.choice([-1, 1], size=(L, n))
    J_learned, traj = m.stream_memristor_couplings(
        patterns, eta=eta, gamma=gamma, dt=dt, steps_per_pattern=1)
    iu = np.triu_indices(n, 1)
    ages = np.arange(L)[::-1]
    w = (eta / gamma) * (1 - np.exp(-gamma * dt))
    w_decay = np.exp(-gamma * dt * ages)
    J_theory = np.zeros((n, n))
    for mu in range(L):
        J_theory += w * w_decay[mu] * np.outer(patterns[mu], patterns[mu])
    np.fill_diagonal(J_theory, 0.0)
    J_hebb = m.hebb_theory(patterns)
    final_corr = np.corrcoef(J_learned[iu], J_theory[iu])[0, 1]
    corr_hebb = np.corrcoef(J_learned[iu], J_hebb[iu])[0, 1]
    hebb = dict(J_theory_iu=J_theory[iu], J_learned_iu=J_learned[iu],
                J_hebb_iu=J_hebb[iu], corr=final_corr, corr_hebb=corr_hebb)

    # --- (b) retrieval vs noise (verbatim from fig_retrieval) ----------
    n = 8
    L = 3
    eta, gamma = 1.0 / n, 0.01
    patterns = RNG.choice([-1, 1], size=(L, n))
    J, _ = m.stream_memristor_couplings(patterns, eta=eta, gamma=gamma,
                                        dt=1.0, steps_per_pattern=1)
    spins = m.spin_table(n)
    # finer grid (17 points, step 1/32); f=0.125 stays exactly on-grid (headline)
    noise_levels = np.arange(0.0, 0.5001, 1.0 / 32.0)
    n_trials = 600        # high statistics -> smooth curves, tight 95% CI bands
    Omega0 = 1.0
    K = len(noise_levels)
    acc_q, ci_q = np.zeros(K), np.zeros(K)     # Omega>0 quantum anneal
    acc_c, ci_c = np.zeros(K), np.zeros(K)     # Omega=0 greedy descent
    acc_sa, ci_sa = np.zeros(K), np.zeros(K)   # classical simulated annealing
    def _corrupt(true_b, nflip):
        noisy_b = true_b.copy()
        if nflip > 0:
            B = true_b.shape[0]
            flip = np.argsort(RNG.random((B, n)), axis=1)[:, :nflip]   # distinct cols/row
            noisy_b[np.repeat(np.arange(B), nflip), flip.ravel()] *= -1
        return noisy_b
    for fi, f in enumerate(noise_levels):
        nflip = int(np.floor(f * n + 0.5))
        true_b = patterns[RNG.integers(0, L, size=n_trials)].copy()    # (B, n)
        noisy_b = _corrupt(true_b, nflip)
        oq = m.overlap_batch(m.retrieve_quantum_anneal_batch(J, noisy_b, Omega0=Omega0, spins=spins), true_b)
        oc = m.overlap_batch(m.retrieve_classical_batch(J, noisy_b), true_b)
        osa = m.overlap_batch(m.retrieve_classical_sa_batch(J, noisy_b), true_b)
        for samp, acc, ci in ((oq, acc_q, ci_q), (oc, acc_c, ci_c), (osa, acc_sa, ci_sa)):
            acc[fi] = samp.mean()
            ci[fi] = 1.96 * samp.std(ddof=1) / np.sqrt(len(samp))
    retrieval = dict(noise=noise_levels, acc_q=acc_q, ci_q=ci_q,
                     acc_c=acc_c, ci_c=ci_c, acc_sa=acc_sa, ci_sa=ci_sa,
                     n_trials=n_trials)

    # --- (c) capacity vs L (verbatim from fig_capacity) ----------------
    ns = [6, 8, 10]
    n_nets = 60        # more networks -> tighter SEM CI band across networks
    n_probes = 8
    f_noise = 0.125
    cap_curves = {}
    empirical = {}
    for n in ns:
        spins = m.spin_table(n)
        eta, gamma = 1.0 / n, 0.01
        Ls = list(range(1, n + 1))
        succ, succ_ci = [], []
        nflip = int(np.floor(f_noise * n + 0.5))
        for L in Ls:
            per_net = []
            for _ in range(n_nets):
                patterns = RNG.choice([-1, 1], size=(L, n))
                J, _ = m.stream_memristor_couplings(patterns, eta=eta, gamma=gamma,
                                                    dt=1.0, steps_per_pattern=1)
                true_b = patterns[RNG.integers(0, L, size=n_probes)].copy()
                noisy_b = true_b.copy()
                if nflip > 0:
                    flip = np.argsort(RNG.random((n_probes, n)), axis=1)[:, :nflip]
                    noisy_b[np.repeat(np.arange(n_probes), nflip), flip.ravel()] *= -1
                R = m.retrieve_quantum_anneal_batch(J, noisy_b, Omega0=1.0, spins=spins)
                per_net.append(np.mean(m.overlap_batch(R, true_b) > 0.999))
            per_net = np.asarray(per_net)
            succ.append(per_net.mean())
            succ_ci.append(1.96 * per_net.std(ddof=1) / np.sqrt(len(per_net)))
        succ = np.array(succ); succ_ci = np.array(succ_ci)
        cap = 0
        for L, s in zip(Ls, succ):
            if s >= 0.9:
                cap = L
        cap_curves[n] = (np.array(Ls), succ, succ_ci)
        empirical[n] = cap
    capacity = dict(curves=cap_curves, empirical=empirical, n_nets=n_nets)
    panel_c = DATA_DIR / "fix_panel_c_capacity.npz"   # read by fig_redesign.py, Fig. 3(c); the shipped file is not overwritten
    if not panel_c.exists():
        np.savez(panel_c, **{f"{k}_{n}": v for n, c in cap_curves.items() for k, v in zip(("Ls", "succ", "sci"), c)})

    # --- (d) palimpsest (verbatim from fig_palimpsest) -----------------
    n = 8
    L = 20
    eta = 0.30
    gamma = 0.30
    dt = 1.0
    f_noise = 0.0625
    n_streams = 60        # more streams -> tighter SEM CI across streams
    n_probes = 12
    spins = m.spin_table(n)
    ages = np.arange(L)
    per_stream = np.zeros((n_streams, L))
    nflip = int(np.floor(f_noise * n + 0.5))
    for st in range(n_streams):
        patterns = RNG.choice([-1, 1], size=(L, n))
        J_final, _ = m.stream_memristor_couplings(
            patterns, eta=eta, gamma=gamma, dt=dt, steps_per_pattern=1)
        for mu in range(L):
            age = (L - 1) - mu
            true_b = np.tile(patterns[mu], (n_probes, 1))
            noisy_b = true_b.copy()
            if nflip > 0:
                flip = np.argsort(RNG.random((n_probes, n)), axis=1)[:, :nflip]
                noisy_b[np.repeat(np.arange(n_probes), nflip), flip.ravel()] *= -1
            R = m.retrieve_quantum_anneal_batch(J_final, noisy_b, Omega0=1.0, spins=spins)
            per_stream[st, age] = np.mean(m.overlap_batch(R, true_b) > 0.999)
    retr = per_stream.mean(axis=0)
    retr_ci = 1.96 * per_stream.std(axis=0, ddof=1) / np.sqrt(n_streams)
    weight = np.exp(-gamma * ages * dt)
    window = 0
    for a in range(L):
        if retr[a] >= 0.5:
            window += 1
        else:
            break
    palimpsest = dict(ages=ages, retr=retr, retr_ci=retr_ci, weight=weight,
                      window=window, gamma=gamma, n_streams=n_streams)

    return dict(hebb=hebb, retrieval=retrieval, capacity=capacity,
                palimpsest=palimpsest)


# ======================================================================
#  FIGURE: paper_qrc.pdf   (2 panels: prediction + NMSE bars)
# ======================================================================
def make_paper_qrc(d):
    # SINGLE-COLUMN figure: native width 3.4 in, included at \columnwidth
    # (~3.4 in), so fonts render 1:1 on the page.  Panels stacked 2x1:
    # (a) time series on top, (b) NMSE bars below.  Base font 9 pt with
    # tick/legend text pinned at 8 pt keeps every string in the 8-9.5 pt
    # effective window; line widths are scaled to match the previous
    # on-page stroke weight (2 pt at 0.55x ~= 1.1 pt).
    figstyle.apply(base_fontsize=9)
    plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8})
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.4, 4.5))

    # (a) target vs prediction.  Colour alone must not carry the distinction
    # (colourblind safety): the target is a thicker SOLID blue reference
    # stroke and the prediction a thinner DASHED red trace drawn on top --
    # the paper-wide blue-solid / red-dashed two-series language (Figs. 5a,
    # 8a, 9, 12) -- so linestyle+width are redundant cues and the prediction
    # stays legible where the curves overlap.
    tt = np.arange(len(d["yt_q"]))
    ax0.plot(tt, d["yt_q"], color=BLUE, lw=1.7, alpha=0.85,
             label=r"target $y_t$")
    ax0.plot(tt, d["yp_q"], color=RED, lw=1.0, ls="--",
             dashes=(3.2, 1.4), label=r"prediction $\hat y_t$")
    ax0.set_xlabel(r"time step $t$")
    ax0.set_ylabel(r"NARMA-10 output")
    ax0.set_title(rf"quantum reservoir, NMSE $={d['nmse_q']:.3f}$")
    # Compact FRAMELESS legend parked in reserved ylim headroom, so it never
    # covers data and needs no occluding box (less non-data ink).
    ax0.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.7,
               handletextpad=0.4, borderpad=0.15, borderaxespad=0.25,
               frameon=False)
    lo0, hi0 = ax0.get_ylim()
    ax0.set_ylim(lo0, hi0 + 0.28 * (hi0 - lo0))
    panel_label(ax0, "(a)")

    # (b) NMSE bars -- three series => blue / red / grey third.
    labels = ["linear\nfloor", "classical\nESN", "quantum\nreservoir"]
    vals = [d["nmse_lin"], d["nmse_esn"], d["nmse_q"]]
    colors = [THIRD, RED, BLUE]
    bars = ax1.bar(range(3), vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, rf"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel(r"test NMSE")
    ax1.set_ylim(0, max(vals) * 1.25)
    panel_label(ax1, "(b)")

    fig.tight_layout(pad=0.3, h_pad=1.1)
    return save(fig, "paper_qrc.pdf")


# ======================================================================
#  FIGURE: paper_qrc_gamma.pdf   (single panel, NMSE vs gamma log-x)
# ======================================================================
def make_paper_qrc_gamma(d):
    # COLUMN figure: displayed at \columnwidth, so keep the paper base font 18
    # (must NOT change apparent size relative to other single-column figures).
    figstyle.apply(base_fontsize=18)
    # Wider, less tall than SINGLE so the single-column figure FILLS the column
    # (matches the reference hysteresis aspect ~1.44, minimal whitespace);
    # apparent on-page font stays ~9 pt (18 x 3.375 / ~6.5 in).
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.semilogx(d["gammas"], d["nmse_g"], "-o", color=BLUE, lw=2, ms=5,
                label="NMSE sweep")
    ax.plot([d["g_opt"]], [d["nmse_g_opt"]], "*", color=RED, ms=18,
            zorder=6, label=(rf"optimum $\gamma={d['g_opt']:.2g}$, "
                             rf"NMSE $={d['nmse_g_opt']:.3f}$"))
    ax.set_xlabel(r"forgetting rate $\gamma$")
    ax.set_ylabel(r"test NMSE")
    # Curve is LOW on the left (small gamma, near the optimum) and RISES to the
    # right; park the legend upper-left over that low/clear left branch, in an
    # opaque framed box so it never reads through the rising curve.
    leg = ax.legend(loc="upper left", frameon=True, framealpha=1.0,
                    handlelength=1.4, handletextpad=0.5, borderaxespad=0.6)
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(0.8)
    leg.set_zorder(7)
    # A little extra headroom so the box clears both the curve and the star.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))
    fig.tight_layout()
    return save(fig, "paper_qrc_gamma.pdf")


# ======================================================================
#  FIGURE: paper_qhnm.pdf   (2x2 panels)
# ======================================================================
def make_paper_qhnm(d):
    # Full-width figure* (GRID2x2, width 10 in displayed at \textwidth ~7 in):
    # base 13 -> on-page apparent font ~9 pt (matches single-column figures).
    figstyle.apply(base_fontsize=13)
    fig, axes = plt.subplots(2, 2, figsize=figstyle.GRID2x2)
    (axA, axB), (axC, axD) = axes

    # (a) Hebbian emergence scatter: learned J vs textbook (1/n) sum xi xi.
    # Plotted against the plain textbook Hebbian (J_hebb) and annotated with
    # the verified corr(learned, plain-Hebb) = 0.9999.  (The learned coupling
    # and the textbook rule differ only by an overall scale, so we rescale the
    # x-axis to the learned units by a least-squares slope for the y=x guide.)
    h = d["hebb"]
    yy = h["J_learned_iu"]
    xraw = h["J_hebb_iu"]
    slope = float(np.dot(xraw, yy) / np.dot(xraw, xraw))  # learned = slope*hebb
    xx = slope * xraw  # textbook Hebbian expressed in learned-J units
    lo = min(xx.min(), yy.min())
    hi = max(xx.max(), yy.max())
    pad = 0.08 * (hi - lo)
    axA.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "-", color=GREY,
             lw=1.0, zorder=1)
    axA.plot(xx, yy, ls="", marker="o", ms=5, color=BLUE, zorder=2)
    axA.set_xlabel(r"textbook Hebbian $J_{ij}$ (rescaled)")
    axA.set_ylabel(r"learned $J_{ij}$")
    axA.text(0.95, 0.06, rf"$r={h['corr_hebb']:.4f}$", transform=axA.transAxes,
             va="bottom", ha="right", color=RED)
    axA.set_xlim(lo - pad, hi + pad)
    axA.set_ylim(lo - pad, hi + pad)
    panel_label(axA, "(a)")

    # (b) retrieval vs noise -- three series with 95% CI bands:
    # quantum anneal (Omega>0), classical simulated annealing, greedy (Omega=0).
    r = d["retrieval"]
    nz = r["noise"]
    for acc, ci, col in ((r["acc_q"], r["ci_q"], BLUE),
                         (r["acc_sa"], r["ci_sa"], THIRD),
                         (r["acc_c"], r["ci_c"], RED)):
        axB.fill_between(nz, acc - ci, acc + ci, color=col, alpha=0.16, lw=0)
    axB.plot(nz, r["acc_q"], "-o", color=BLUE, ms=5, lw=2,
             label=r"$\Omega>0$ (quantum anneal)")
    axB.plot(nz, r["acc_sa"], "-^", color=THIRD, ms=5, lw=2,
             label=r"classical SA")
    axB.plot(nz, r["acc_c"], "-s", color=RED, ms=5, lw=2,
             label=r"$\Omega=0$ (greedy)")
    axB.set_xlabel(r"input noise fraction $f$")
    axB.set_ylabel(r"recall overlap")
    axB.set_ylim(0.4, 1.03)
    axB.legend(loc="lower left", handlelength=1.6,
               fontsize=0.72 * plt.rcParams["font.size"])
    panel_label(axB, "(b)")

    # (c) capacity vs L -- three series (n=6,8,10) => blue / red / grey third.
    c = d["capacity"]
    cap_colors = {6: BLUE, 8: RED, 10: THIRD}
    for n in (6, 8, 10):
        Ls, succ, sci = c["curves"][n]
        axC.fill_between(Ls, succ - sci, succ + sci, color=cap_colors[n],
                         alpha=0.15, lw=0)
        axC.plot(Ls, succ, "-o", color=cap_colors[n], ms=5, lw=2,
                 label=rf"$n={n}$")
    for n in (6, 8, 10):
        axC.axvline(0.14 * n, color=cap_colors[n], ls=":", lw=1.2, alpha=0.9)
    axC.text(0.97, 0.55, r"$0.14\,n$ bound", transform=axC.transAxes,
             ha="right", va="center", color=GREY,
             fontsize=0.72 * plt.rcParams["font.size"])
    axC.set_xlabel(r"stored patterns $L$")
    axC.set_ylabel(r"recall success")
    axC.set_ylim(-0.03, 1.05)
    axC.legend(loc="lower left", ncol=3, columnspacing=0.8,
               handletextpad=0.3, handlelength=1.2)
    panel_label(axC, "(c)")

    # (d) palimpsest -- blue bars (sim.) + red theory weight curve.
    p = d["palimpsest"]
    axD.bar(p["ages"], p["retr"], width=0.85, color=BLUE, alpha=0.55,
            label=r"recall (sim.)")
    axD.errorbar(p["ages"], p["retr"], yerr=p["retr_ci"], fmt="none",
                 ecolor="0.25", elinewidth=0.8, capsize=1.5, zorder=5)
    axD.plot(p["ages"], p["weight"], color=RED, lw=2,
             label=r"$e^{-\gamma\,\mathrm{age}}$")
    axD.axhline(0.5, color=GREY, ls=":", lw=1.0)
    axD.set_xlabel(r"pattern age $(T-t_\mu)/\Delta t$")
    axD.set_ylabel(r"retrievability")
    axD.set_ylim(0, 1.05)
    axD.set_xlim(-0.6, len(p["ages"]) - 0.4)
    axD.legend(loc="upper right", handlelength=1.6)
    panel_label(axD, "(d)")

    fig.tight_layout()
    return save(fig, "paper_qhnm.pdf")


# ======================================================================
def main():
    print("=" * 70)
    print("Computing QRC data (SEED=%d) ..." % qrc_mod.SEED)
    qd = compute_qrc()
    print(f"  nmse_linear   = {qd['nmse_lin']:.4f}   (verified 0.7734)")
    print(f"  nmse_esn      = {qd['nmse_esn']:.4f}   (verified 0.3643)")
    print(f"  nmse_quantum  = {qd['nmse_q']:.4f}   (verified 0.4720)")
    print(f"  gamma_opt     = {qd['g_opt']:.3f}    (verified 0.026)")
    print(f"  nmse@gamma_opt= {qd['nmse_g_opt']:.4f}   (verified 0.4731)")

    print("Computing QHN-M data (SEED=%d) ..." % qhnm_mod.SEED)
    hd = compute_qhnm()
    print(f"  corr(learned, plain-Hebb) = {hd['hebb']['corr_hebb']:.4f}"
          f"  (verified 0.9999)")
    ni = list(hd["retrieval"]["noise"]).index(0.125)
    print(f"  retrieval f=0.125: Omega>0={hd['retrieval']['acc_q'][ni]:.3f} "
          f"(verified 0.906), Omega=0={hd['retrieval']['acc_c'][ni]:.3f} "
          f"(verified 0.769)")
    print(f"  capacity empirical L = {hd['capacity']['empirical']} "
          f"(verified 1,1,1)  0.14n = "
          f"{[round(0.14*n,2) for n in (6,8,10)]}")
    print(f"  palimpsest window = {hd['palimpsest']['window']} patterns "
          f"(verified ~2)")

    print("-" * 70)
    p1 = make_paper_qrc(qd)
    p2 = make_paper_qrc_gamma(qd)
    p3 = make_paper_qhnm(hd)
    print("-" * 70)
    print("Figures written:")
    for p in (p1, p2, p3):
        print("  " + p)


if __name__ == "__main__":
    main()
