"""
Basis-dependent quantum hysteresis loops for a quantum memristor (Fig. 4 of Part I).

Mean-field (deterministic) simulation of a photonic quantum memristor
driven by sinusoidal input polarization and measured in X, Y, Z bases.
Each basis produces a different feedback signal and therefore a different
hysteresis loop shape.

Data in figures/data/basis_hysteresis.npz (computed if missing); the drawing
layer only reads it.  Output: figures/memristor_basis_dependent.pdf, drawn at
its final size in the paper (\\columnwidth = 3.40 in).

Authors: K. Sierant et al.
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
ETA = 0.5           # feedback strength
GAMMA = 0.01        # decay rate toward rest angle
THETA0 = 0.0        # rest angle
OMEGA = 0.12        # angular frequency of the input polarization
A = 1.0             # amplitude of input polarization angle

T_CYCLE = 2 * np.pi / OMEGA          # period of one cycle
T_MAX = 5 * T_CYCLE                  # 5 full cycles
DT = 0.01                            # time step

BASES = ["Z", "X", "Y"]
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = Path(os.environ.get("FIG_DATA_DIR", REPO / "figures" / "data")) / "basis_hysteresis.npz"
OUT = REPO / "figures" / "memristor_basis_dependent.pdf"


def alpha(t):
    """Input polarization angle: alpha(t) = (A/2) * sin(omega * t)."""
    return (A / 2.0) * np.sin(OMEGA * t)


def input_signal(t):
    """Observable input: I(t) = sin(2*alpha) = sin(A * sin(omega * t))."""
    return np.sin(A * np.sin(OMEGA * t))


def expected_signal(basis, theta, t):
    """Mean measurement outcome <m_B>: Z cos, X sin, Y 0 (of 4 theta - 2 alpha)."""
    arg = 4.0 * theta - 2.0 * alpha(t)
    if basis == "Z":
        return np.cos(arg)
    elif basis == "X":
        return np.sin(arg)
    elif basis == "Y":
        return 0.0 * theta  # works for both scalar and array
    raise ValueError(f"Unknown basis: {basis}")


def output_observable(basis, theta, t):
    """Output observable: ALWAYS the Z-basis observable cos(4*theta - 2*alpha),
    evaluated on the theta trajectory produced by feedback basis B."""
    return np.cos(4.0 * theta - 2.0 * alpha(t))


def theta_rhs(t, theta, basis):
    """d(theta)/dt = eta * <m_B>(theta, alpha(t)) - gamma * (theta - theta0)."""
    return ETA * expected_signal(basis, theta, t) - GAMMA * (theta - THETA0)


def solve_theta(basis):
    t_eval = np.arange(0.0, T_MAX, DT)
    sol = solve_ivp(theta_rhs, (0.0, T_MAX), [THETA0], t_eval=t_eval,
                    args=(basis,), method="RK45", rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0]


def shoelace_area(x, y):
    xc = np.append(x, x[0])
    yc = np.append(y, y[0])
    return 0.5 * np.abs(np.sum(xc[:-1] * yc[1:] - xc[1:] * yc[:-1]))


def compute():
    """Mean-field runs for the three bases; last two cycles, areas, indices."""
    print("Running mean-field simulations for X, Y, Z bases...")
    loop_data, areas = {}, {}
    t_start = 3 * T_CYCLE          # last 2 cycles (first 3 are transient)
    for basis in BASES:
        t, theta = solve_theta(basis)
        mask = t >= t_start
        loop_data[basis] = (input_signal(t)[mask], output_observable(basis, theta, t)[mask])
        areas[basis] = shoelace_area(*loop_data[basis])
        print(f"  A_{basis} = {areas[basis]:.4f}")
    A_X, A_Y, A_Z = areas["X"], areas["Y"], areas["Z"]
    S_XZ = abs(A_X - A_Z) / (A_X + A_Z) if (A_X + A_Z) > 0 else 0.0
    A_ref = (A_X + A_Y + A_Z) / 3.0
    W_Q = abs(A_X * A_Y * A_Z / A_ref**3 - 1.0) if A_ref > 0 else 1.0
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, S_XZ=S_XZ, W_Q=W_Q,
             **{f"I_{b}": loop_data[b][0] for b in BASES},
             **{f"O_{b}": loop_data[b][1] for b in BASES})


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    sys.path.insert(0, str(HERE))
    plt.rcdefaults()   # model modules may have changed rcParams (e.g. savefig.bbox)
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 7, "axes.labelsize": 7,
                         "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5,
                         "lines.linewidth": 0.9, "lines.markersize": 2.8, "legend.frameon": False,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    d = np.load(CACHE, allow_pickle=False)
    S_XZ = float(d["S_XZ"])
    col = {"Z": plt.cm.turbo(0.12), "X": plt.cm.turbo(0.88), "Y": "0.45"}
    sty = {"Z": "-", "X": "-", "Y": "--"}   # X solid: its near-flat loop retraces itself
    fig, ax = plt.subplots(figsize=(3.40, 1.55), layout="constrained")
    ax.axhline(0.0, color="0.75", lw=0.4, zorder=0)
    ax.axvline(0.0, color="0.75", lw=0.4, zorder=0)
    for b in BASES:
        I, O = d[f"I_{b}"], d[f"O_{b}"]
        if b == "Y":   # single-valued curve: draw it in one pass so the dashes stay visible
            o = np.argsort(I)
            I, O = I[o], O[o]
        ax.plot(I, O, color=col[b], ls=sty[b], lw=1.1)
    # direct labels right of the vertical zero line, in the clear region of each curve
    ax.text(0.50, 0.66, r"$Y$-basis", ha="center", va="center")
    ax.text(0.50, 0.24, r"$Z$-basis", ha="center", va="center")
    ax.text(0.50, -0.80, r"$X$-basis", ha="center", va="center")
    ax.text(-0.50, -0.42, rf"$\mathcal{{M}}_{{X,Z}} = {S_XZ:.2f}$", ha="center", va="center")
    fmt = FuncFormatter(lambda x, _: rf"${x:g}$")
    for a, lim in ((ax.xaxis, (-1.0, 1.0)), (ax.yaxis, (-1.15, 1.15))):
        a.set_major_locator(FixedLocator([-1, -0.5, 0, 0.5, 1]))
        a.set_major_formatter(fmt)
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"Input $I(t) = \sin 2\chi$")
    ax.set_ylabel(r"Output $\langle\hat{\sigma}_z\rangle$")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"Figure saved to {OUT}  (M_XZ = {S_XZ:.4f}, W_Q = {float(d['W_Q']):.4f})")


if __name__ == "__main__":
    if not CACHE.exists():
        compute()
    draw()
