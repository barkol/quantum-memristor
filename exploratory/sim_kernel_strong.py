"""
Impulse-response test of the memory kernel deep in the nonlinear regime (Fig. S4).

The Nakajima-Zwanzig form of the paper is derived by expanding L(theta) to
first order around theta0, formally controlled for eta*Gamma/gamma <~ 1.
Here we test, without any expansion, whether the *closed-loop* memory
response of the full nonlinear mean-field dynamics (the strongly memristive
simulation regime eta=0.5, gamma=0.01) is still a single exponential, and
whether its rate is captured by cycle-averaged linear response about the
settled limit cycle:

    d(dtheta)/dt = -[gamma + 4*eta*sin(4*theta* - 2*alpha)] * dtheta
                 = -gamma_eff(t) * dtheta,
    gamma_eff_bar = gamma + 4*eta*<sin(4*theta* - 2*alpha)>_cycle .

Protocol: settle the Z-feedback limit cycle (basis_hysteresis.py model),
kick theta by dtheta0 at several phases of the settled cycle, follow the
difference trajectory, fit a single exponential to |dtheta(t)|, and compare
the fitted rate with gamma_eff_bar and with the bare gamma.

Deterministic (no RNG), about 15 s.  Data in figures/data/sim_kernel_strong.npz
(computed if missing); the drawing layer only reads it.
Output: figures/paper_kernel_strong.pdf at its final size (\\columnwidth = 3.40 in).
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# ── Parameters (identical to the strong-regime figures) ────────────────────
GAMMA = 0.01
THETA0 = 0.0
OMEGA = 0.12
A = 1.0
T_CYCLE = 2.0 * np.pi / OMEGA
KICK = 0.02          # small kick on top of the *nonlinear* limit cycle
N_SETTLE = 6         # settled cycles before kicking
N_PHASES = 8         # kick phases across one settled cycle
FIT_WINDOW = 1.5 * T_CYCLE   # cap on the fit range after the kick
ETA_STRONG = 0.5
ETAS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = Path(os.environ.get("FIG_DATA_DIR", REPO / "figures" / "data")) / "sim_kernel_strong.npz"
OUT = REPO / "figures" / "paper_kernel_strong.pdf"


def alpha(t):
    return 0.5 * A * np.sin(OMEGA * t)


def rhs(t, theta, eta):
    """Full nonlinear mean-field Z-feedback law (no expansion)."""
    return eta * np.cos(4.0 * theta - 2.0 * alpha(t)) - GAMMA * (theta - THETA0)


def settled_cycle(eta):
    """Integrate to the settled limit cycle; return (t_grid, theta*) over
    one cycle starting at t_s = N_SETTLE*T_CYCLE."""
    t_s = N_SETTLE * T_CYCLE
    t_grid = np.arange(0.0, t_s + T_CYCLE, 0.02)
    sol = solve_ivp(rhs, (0.0, t_grid[-1]), [THETA0], t_eval=t_grid,
                    args=(eta,), rtol=1e-10, atol=1e-12)
    mask = sol.t >= t_s
    return sol.t[mask], sol.y[0][mask]


def gamma_eff_prediction(eta):
    """Cycle-averaged linear-response rate about the settled limit cycle."""
    t_c, th_c = settled_cycle(eta)
    return GAMMA + 4.0 * eta * np.mean(np.sin(4.0 * th_c - 2.0 * alpha(t_c)))


def measure_gamma_eff(eta):
    """Kick at N_PHASES phases; fit exp decay of |dtheta|; return
    (mean_rate, std_rate, example_traces) where traces are (tau, |dtheta|)."""
    t_s = N_SETTLE * T_CYCLE
    window = min(FIT_WINDOW, 8.0 / max(gamma_eff_prediction(eta), 1e-3))
    rates, traces = [], []
    for k in range(N_PHASES):
        t0 = t_s + k * T_CYCLE / N_PHASES
        t_eval = np.arange(t0, t0 + window, window / 400.0)
        ref0 = solve_ivp(rhs, (0.0, t0), [THETA0], args=(eta,), rtol=1e-10, atol=1e-12)
        th_at_t0 = ref0.y[0][-1]
        ref = solve_ivp(rhs, (t0, t_eval[-1]), [th_at_t0], t_eval=t_eval,
                        args=(eta,), rtol=1e-10, atol=1e-12)
        kick = solve_ivp(rhs, (t0, t_eval[-1]), [th_at_t0 + KICK],
                         t_eval=t_eval, args=(eta,), rtol=1e-10, atol=1e-12)
        d = np.abs(kick.y[0] - ref.y[0])
        tau = t_eval - t0
        ok = d > KICK * 1e-3
        slope = np.polyfit(tau[ok], np.log(d[ok]), 1)[0]
        rates.append(-slope)
        traces.append((tau, d))
    return float(np.mean(rates)), float(np.std(rates)), traces


def compute():
    # panel (a): decay traces at eta = 0.5 (the paper's strong regime)
    mean_r, std_r, traces = measure_gamma_eff(ETA_STRONG)
    pred_r = gamma_eff_prediction(ETA_STRONG)
    # panel (b): renormalized rate versus feedback strength
    meas = np.array([measure_gamma_eff(e)[:2] for e in ETAS])
    preds = np.array([gamma_eff_prediction(e) for e in ETAS])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, mean_r=mean_r, std_r=std_r, pred_r=pred_r,
             n_traces=len(traces), etas=ETAS, meas=meas, preds=preds,
             **{f"trace_tau_{k}": tt for k, (tt, _) in enumerate(traces)},
             **{f"trace_d_{k}": dd for k, (_, dd) in enumerate(traces)})


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
    sys.path.insert(0, str(HERE))
    from paneltags import tag, place
    plt.rcdefaults()   # model modules may have changed rcParams (e.g. savefig.bbox)
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 7, "axes.labelsize": 7,
                         "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5,
                         "lines.linewidth": 0.9, "lines.markersize": 2.8, "legend.frameon": False,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    d = np.load(CACHE, allow_pickle=False)
    mean_r, pred_r = float(d["mean_r"]), float(d["pred_r"])
    traces = [(d[f"trace_tau_{k}"], d[f"trace_d_{k}"]) for k in range(int(d["n_traces"]))]
    etas, meas, preds = d["etas"], d["meas"], d["preds"]
    blue, red, grey = plt.cm.turbo(0.12), plt.cm.turbo(0.88), "0.45"
    fmt = FuncFormatter(lambda x, _: rf"${x:g}$")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(3.40, 1.45), layout="constrained")
    tau_max = max(t[-1] for t, _ in traces)
    for k, (tau, dd) in enumerate(traces):
        axa.semilogy(tau, dd, color=blue, lw=0.8, alpha=0.55, label="kicks" if k == 0 else None)
    tf = np.linspace(0.0, tau_max, 200)
    axa.semilogy(tf, KICK * np.exp(-mean_r * tf), color=red, lw=0.9, ls="--",
                 label=rf"fit, $\gamma_\mathrm{{eff}}={mean_r:.3f}$")
    axa.semilogy(tf, KICK * np.exp(-GAMMA * tf), color=grey, lw=0.8, ls=":",
                 label=rf"bare $\gamma={GAMMA:g}$")
    axa.set_ylim(KICK * 1e-4, KICK * 3)
    axa.set_xlim(0, tau_max)
    axa.yaxis.set_major_locator(LogLocator(numticks=10))
    axa.yaxis.set_minor_formatter(NullFormatter())
    axa.xaxis.set_major_formatter(fmt)
    axa.set_xlabel(r"$t-t_0$")
    axa.set_ylabel(r"$|\delta\theta(t)|$")
    axa.legend(loc="lower left", handlelength=1.6, handletextpad=0.4, labelspacing=0.15,
               borderaxespad=0.2, borderpad=0.1)
    tag(axa, "(a)")

    rel = np.abs(meas[:, 0] / preds - 1.0)
    rel_err = meas[:, 1] / preds
    # upper-sided bars: the phase scatter exceeds the mean, so the lower half
    # is unrepresentable on the log axis
    axb.errorbar(etas, rel, yerr=[np.zeros_like(rel), rel_err], fmt="o", ms=2.8,
                 color=blue, lw=0.8, capsize=1.5, capthick=0.6)
    axb.set_yscale("log")
    axb.set_ylim(4e-7, 5e-2)
    axb.set_xlim(-0.02, 0.86)
    axb.xaxis.set_major_locator(FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8]))
    axb.xaxis.set_major_formatter(fmt)
    axb.yaxis.set_major_locator(FixedLocator([1e-6, 1e-4, 1e-2]))
    axb.yaxis.set_minor_formatter(NullFormatter())
    axb.axhline(1e-3, color=grey, lw=0.8, ls=":")
    axb.text(0.84, 1.35e-3, r"$10^{-3}$", color="0.3", fontsize=6, ha="right", va="bottom")
    axb.set_xlabel(r"feedback strength $\eta$")
    axb.set_ylabel(r"$|\gamma_\mathrm{eff}^\mathrm{fit}/\bar{\gamma}_\mathrm{eff}-1|$")
    tag(axb, "(b)")
    place(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"Saved: {OUT}")
    print(f"eta=0.5: measured gamma_eff = {mean_r:.4f} +- {float(d['std_r']):.4f}, "
          f"prediction = {pred_r:.4f}, bare gamma = {GAMMA}")
    for e, (m, s), p in zip(etas, meas, preds):
        print(f"  eta={e:4.2f}: measured {m:.4f}+-{s:.4f}  predicted {p:.4f}")


if __name__ == "__main__":
    if not CACHE.exists():
        compute()
    draw()
