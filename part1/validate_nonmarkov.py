"""
Validate quantum memristor mean-field ODE against stochastic quantum trajectories
(Fig. S3, "register_validation.pdf" in the paper).

A sinusoidal polarisation angle alpha(t) = (A/2)*sin(omega*t) drives a HWP whose
angle theta evolves via measurement-based feedback.  The mean-field (deterministic)
and trajectory-averaged (stochastic) descriptions must agree as N_traj -> inf.

Data in figures/data/validate_nonmarkov.npz (computed if missing); the drawing
layer only reads it.  The 2e5-record Monte Carlo writes a resumable checkpoint
(figures/data/validate_nonmarkov.part.npz: running sums + RNG state) after every
batch, so an interrupted run continues where it stopped with bit-identical
results; the part file is removed when the cache is complete.
Output: figures/memristor_nonmarkov_validation.pdf at its final size
(\\columnwidth = 3.40 in).

Reference: Sanz et al., APL Photonics 3, 080801 (2018); our manuscript.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# ── Parameters ──────────────────────────────────────────────────────────────
eta = 0.5
gamma = 0.01
theta0 = 0.0
omega = 0.12
A = 1.0
T_cycle = 2.0 * np.pi / omega
t_max = 3.0 * T_cycle
dt = 0.05
N_traj = 200_000   # batched accumulation -> high statistics, tight band
BATCH = 10_000
SEED = 42

t_eval = np.arange(0.0, t_max + 0.5 * dt, dt)
N_steps = len(t_eval)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_DIR = Path(os.environ.get("FIG_DATA_DIR", REPO / "figures" / "data"))
CACHE = DATA_DIR / "validate_nonmarkov.npz"
PART = DATA_DIR / "validate_nonmarkov.part.npz"
OUT = REPO / "figures" / "memristor_nonmarkov_validation.pdf"


def alpha(t):
    """Input polarisation angle: alpha(t) = (A/2)*sin(omega*t)."""
    return 0.5 * A * np.sin(omega * t)


def input_signal(t):
    """I(t) = sin(2*alpha) = sin(A*sin(omega*t))."""
    return np.sin(A * np.sin(omega * t))


def output_signal(theta, t):
    """O(t) = cos(4*theta - 2*alpha(t))."""
    return np.cos(4.0 * theta - 2.0 * alpha(t))


def compute(stop_after=None):
    """Mean-field ODE + batched trajectories.  stop_after: number of batches
    after which to stop (test of the resume path); None runs to completion."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Method 1: mean-field ODE (deterministic)
    def rhs(t, theta):
        return eta * np.cos(4.0 * theta - 2.0 * alpha(t)) - gamma * (theta - theta0)

    sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], [theta0], t_eval=t_eval,
                    method="RK45", rtol=1e-10, atol=1e-12)
    theta_mf = sol.y[0]
    output_mf = output_signal(theta_mf, t_eval)

    # Method 2: quantum trajectories, accumulated online in batches; resumable
    rng = np.random.default_rng(seed=SEED)
    if PART.exists():
        p = np.load(PART, allow_pickle=False)
        done = int(p["done"])
        sum_th, sumsq_th = p["sum_th"].copy(), p["sumsq_th"].copy()
        sum_out, sumsq_out = p["sum_out"].copy(), p["sumsq_out"].copy()
        rng.bit_generator.state = json.loads(str(p["rng_state"]))
        print(f"resuming from checkpoint: {done} records done")
    else:
        done = 0
        sum_th = np.zeros(N_steps); sumsq_th = np.zeros(N_steps)
        sum_out = np.zeros(N_steps); sumsq_out = np.zeros(N_steps)
    nb = 0
    while done < N_traj:
        B = min(BATCH, N_traj - done)
        th_b = np.empty((B, N_steps)); out_b = np.empty((B, N_steps))
        theta_cur = np.full(B, theta0)
        for k in range(N_steps):
            a_k = alpha(t_eval[k])
            P_H = 0.5 * (1.0 + np.cos(4.0 * theta_cur - 2.0 * a_k))
            outcome = np.where(rng.random(B) < P_H, 1.0, -1.0)
            out_b[:, k] = outcome
            th_b[:, k] = theta_cur
            if k < N_steps - 1:
                theta_cur = (theta_cur + eta * outcome * dt
                             - gamma * (theta_cur - theta0) * dt)
        sum_th += th_b.sum(0);  sumsq_th += (th_b ** 2).sum(0)
        sum_out += out_b.sum(0); sumsq_out += (out_b ** 2).sum(0)
        done += B
        nb += 1
        tmp = PART.with_name(PART.stem + ".tmp.npz")
        np.savez(tmp, done=done, sum_th=sum_th, sumsq_th=sumsq_th, sum_out=sum_out,
                 sumsq_out=sumsq_out, rng_state=json.dumps(rng.bit_generator.state))
        os.replace(tmp, PART)
        print(f"  {done}/{N_traj} records (checkpoint)", flush=True)
        if stop_after is not None and nb >= stop_after and done < N_traj:
            print("stopping early (resume test)")
            return False

    # Ensemble statistics
    theta_traj_mean = sum_th / N_traj
    theta_traj_std = np.sqrt(np.maximum(sumsq_th / N_traj - theta_traj_mean ** 2, 0.0))
    output_traj_mean = sum_out / N_traj
    output_traj_std = np.sqrt(np.maximum(sumsq_out / N_traj - output_traj_mean ** 2, 0.0))
    theta_sem = theta_traj_std / np.sqrt(N_traj)
    output_sem = output_traj_std / np.sqrt(N_traj)

    residual = theta_mf - theta_traj_mean
    rms_error = np.sqrt(np.mean(residual ** 2))

    # hysteresis: skip the first cycle (transient), keep the last two
    mask_hyst = t_eval >= T_cycle
    I_hyst = input_signal(t_eval[mask_hyst])
    O_mf_hyst = output_mf[mask_hyst]
    O_traj_hyst = output_traj_mean[mask_hyst]

    np.savez(CACHE, theta_mf=theta_mf, output_mf=output_mf,
             theta_traj_mean=theta_traj_mean, theta_sem=theta_sem,
             output_traj_mean=output_traj_mean, output_sem=output_sem,
             residual=residual, rms_error=rms_error,
             I_hyst=I_hyst, O_mf_hyst=O_mf_hyst, O_traj_hyst=O_traj_hyst)
    PART.unlink()
    return True


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
    blue, red = plt.cm.turbo(0.12), plt.cm.turbo(0.88)
    band = dict(color=plt.cm.turbo(0.30), alpha=0.6, linewidth=0)
    fmt = FuncFormatter(lambda x, _: rf"${x:g}$")
    tlocs = FixedLocator([0, 50, 100, 150])

    fig, axs = plt.subplots(2, 2, figsize=(3.40, 2.45), layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axs.flat

    # (a) theta(t)
    ax = ax_a
    th, sem = d["theta_traj_mean"], d["theta_sem"]
    ax.fill_between(t_eval, th - sem, th + sem, label=r"$\pm\sigma/\sqrt{N}$", **band)
    ax.plot(t_eval, th, color=blue, label=r"record avg.")
    ax.plot(t_eval, d["theta_mf"], color=red, ls="--", label=r"mean field")
    ax.set_ylabel(r"$\theta(t)$")
    ax.set_ylim(-0.02, 0.72)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6]))
    tag(ax, "(a)")

    # (b) output signal O(t)
    ax = ax_b
    o, osem = d["output_traj_mean"], d["output_sem"]
    ax.fill_between(t_eval, o - osem, o + osem, **band)
    ax.plot(t_eval, o, color=blue)
    ax.plot(t_eval, d["output_mf"], color=red, ls="--")
    ax.set_ylabel(r"$O(t)$")
    ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
    # one legend for all panels, in the empty upper part of (b) (O(t) is small after the first step)
    ax.legend(*ax_a.get_legend_handles_labels(), loc="upper right", handlelength=1.4, handletextpad=0.4,
              labelspacing=0.15, borderaxespad=0.2, borderpad=0.1)
    tag(ax, "(b)")

    # (c) hysteresis O versus I, last two input cycles
    ax = ax_c
    ax.plot(d["I_hyst"], d["O_traj_hyst"], color=blue)
    ax.plot(d["I_hyst"], d["O_mf_hyst"], color=red, ls="--")
    ax.set_xlabel(r"$I(t)$")
    ax.set_ylabel(r"$O(t)$")
    ax.xaxis.set_major_locator(FixedLocator([-0.8, 0, 0.8]))
    ax.yaxis.set_major_locator(FixedLocator([-0.04, 0, 0.04]))
    tag(ax, "(c)")

    # (d) residual, plotted in units of 1e-3
    ax = ax_d
    res = d["residual"]
    ax.fill_between(t_eval, 1e3 * (res - sem), 1e3 * (res + sem), **band)
    ax.plot(t_eval, 1e3 * res, color=blue)
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_ylabel("residual\n" r"($\times 10^{-3}$)")   # theta_MF - <theta>_traj, see caption
    ax.set_ylim(-3.6, 1.6)
    ax.yaxis.set_major_locator(FixedLocator([-3, -2, -1, 0, 1]))
    rms = float(d["rms_error"])
    e = int(np.floor(np.log10(rms)))
    ax.text(0.96, 0.06, rf"RMS $= {rms / 10.0 ** e:.1f}\times 10^{{{e}}}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6)
    tag(ax, "(d)")

    for ax in (ax_a, ax_b, ax_d):
        ax.set_xlim(0, t_max)
        ax.xaxis.set_major_locator(tlocs)
        ax.set_xlabel(r"$t$")
    for ax in axs.flat:
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
    place(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"Saved: {OUT}  (RMS = {rms:.3e})")


if __name__ == "__main__":
    if not CACHE.exists():
        stop = os.environ.get("VALIDATE_STOP_AFTER")
        if not compute(int(stop) if stop else None):
            sys.exit(0)
    draw()
