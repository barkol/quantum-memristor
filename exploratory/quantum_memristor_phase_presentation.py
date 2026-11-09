"""
Generate the six main quantum-memristor figures for the device paper:

    memristor_main_hysteresis.pdf
    memristor_time_evolution.pdf
    memristor_hwp_angle.pdf
    memristor_effect_frequency.pdf
    memristor_effect_coupling.pdf
    memristor_effect_relaxation.pdf

Pinched-hysteresis (memristive-fingerprint) model.  The device is driven by a
signed input control v(t) = V0 sin(omega t).  The half-wave-plate angle theta is
a leaky integrator of the drive (the measurement-fed memory),

    dtheta/dt = eta v(t) - gamma (theta - theta0),

and the memory-dependent power transmission of the HWP is G(theta) = sin^2(2 theta).
The output (the photocurrent / H-V detector imbalance) is the memory-modulated
response

    i(t) = G(theta(t)) v(t).

Because i is proportional to the drive, i = 0 whenever v = 0: the loop is PINCHED
at the origin (the defining memristor signature, cf. manuscript Sec. II and the
R_out proportional-to-input-flux argument).  The memory lag opens the loop into a
figure-eight; at high omega the memory cannot follow, theta freezes at theta0, and
the loop collapses to the single-valued line i = G(theta0) v.  The loop area
follows eta*omega/(gamma^2+omega^2), peaking at omega = gamma, reproducing the
Lorentzian A(omega) of the manuscript.

Styling follows the shared house style in figstyle.py (LaTeX serif, blue/red,
thin solid curves, no grid).
"""

import os

import numpy as np
from scipy.integrate import solve_ivp

import figstyle

figstyle.apply()
import matplotlib.pyplot as plt

# -- parameters (illustrative; chosen for a clean memristive fingerprint) -----
ETA = 0.15           # feedback strength (rad per unit drive)
GAMMA = 0.10         # relaxation / forgetting rate
THETA0 = np.pi / 8   # rest angle: most sensitive HWP-transmission bias
OMEGA = 0.30         # drive angular frequency (omega ~ a few x gamma)
V0 = 1.0             # input drive amplitude

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# -- model --------------------------------------------------------------------
def drive(t, omega=OMEGA):
    """Signed input control v(t) = V0 sin(omega t)."""
    return V0 * np.sin(omega * t)


def transmission(theta):
    """Memory-dependent HWP power transmission G(theta) in [0, 1]."""
    return np.sin(2.0 * theta) ** 2


def solve_theta(eta=ETA, gamma=GAMMA, omega=OMEGA, n_cycles=8):
    """Integrate the leaky-integrator memory dynamics; return (t, theta)."""
    t_max = n_cycles * 2.0 * np.pi / omega
    t_eval = np.linspace(0.0, t_max, n_cycles * 500)

    def rhs(t, theta):
        return eta * drive(t, omega) - gamma * (theta - THETA0)

    sol = solve_ivp(rhs, (0.0, t_max), [THETA0], t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-11)
    return sol.t, sol.y[0]


def loop_last_cycles(eta=ETA, gamma=GAMMA, omega=OMEGA, n_cycles=8, skip=6):
    """Return (v, i) over the final settled cycles."""
    t, theta = solve_theta(eta, gamma, omega, n_cycles)
    mask = t >= skip * (2.0 * np.pi / omega)
    v = drive(t[mask], omega)
    i = transmission(theta[mask]) * v
    return v, i


def memoryless_line(v):
    """Frozen-memory reference i = G(theta0) v (single-valued straight line)."""
    return transmission(THETA0) * v


def save(fig, name):
    figstyle.cap_ticks(fig)          # house rule: 4-5 stripped labels per axis
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def _arrows(ax, v, i, fracs=(0.20, 0.70), color=figstyle.BLUE):
    for frac in fracs:
        k = int(frac * (len(v) - 2))
        ax.annotate("", xy=(v[k + 1], i[k + 1]), xytext=(v[k], i[k]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))


# -- Figure 1: main pinched hysteresis loop -----------------------------------
def plot_main_hysteresis():
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)

    v, i = loop_last_cycles()
    vv = np.linspace(-V0, V0, 200)

    ax.plot(vv, memoryless_line(vv), color=figstyle.RED, lw=2,
            label=r"No memory ($\omega\!\to\!\infty$)")
    ax.plot(v, i, color=figstyle.BLUE, lw=2, label="Memristive loop")
    ax.plot(0.0, 0.0, "o", color="k", ms=5, zorder=5)
    _arrows(ax, v, i)

    figstyle.zero_axes(ax)
    # Axis labels at normal weight/standard size (Adam: avoid bold/heavy labels).
    ax.set_xlabel(r"Input drive $v(t)=\sin\omega t$",
                  fontweight="normal", fontsize=plt.rcParams["xtick.labelsize"])
    ax.set_ylabel(r"Output $i(t)=G(\theta)\,v$",
                  fontweight="normal", fontsize=plt.rcParams["ytick.labelsize"])
    figstyle.unit_axis(ax, "both")
    ax.annotate("pinched", xy=(0.0, 0.0), xytext=(0.10, -0.18),
                fontsize=12, color="k")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "memristor_main_hysteresis.pdf")


# -- Figure 2: time traces of drive and output --------------------------------
def plot_time_evolution():
    fig, ax = plt.subplots(figsize=figstyle.WIDE)

    t, theta = solve_theta(n_cycles=4)
    ax.plot(t, drive(t), color=figstyle.GREY, lw=2, ls="--",
            label=r"input $v(t)$")
    ax.plot(t, transmission(theta) * drive(t), color=figstyle.BLUE, lw=2,
            label=r"output $i(t)$")

    figstyle.zero_axes(ax, y=True, x=False)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Signal")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.2, 1.2)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, "memristor_time_evolution.pdf")


# -- Figure 3: HWP memory-angle evolution -------------------------------------
def plot_hwp_angle():
    # COLUMN figure: displayed at \columnwidth, so use the SINGLE aspect (and the
    # paper base font 18) to FILL the column at an apparent ~9 pt, matching
    # memristor_main_hysteresis.pdf.  The memory theta(t) is a leaky integrator of
    # the drive v(t)=sin(omega t), so it LAGS the drive in phase; overlay a scaled
    # v(t) (dashed grey) on the same axis to make that phase lag visible.
    figstyle.apply(base_fontsize=15)   # column fig: base 11 was too small, 18 too
                                       # large -> 15 (base DOES work for this one)
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)

    t, theta = solve_theta(n_cycles=4)
    v = drive(t)

    # Scale the (dimensionless) drive onto the theta range so the two curves share
    # the axis sensibly: centre on theta0, amplitude = the settled swing of theta.
    settled = t >= 0.5 * (2.0 * np.pi / OMEGA)
    amp = 0.5 * (theta[settled].max() - theta[settled].min())
    v_scaled = THETA0 + amp * v

    ax.plot(t, v_scaled, color=figstyle.GREY, lw=1.6, ls="--",
            label=r"drive $\propto\sin\omega t$")
    ax.plot(t, theta, color=figstyle.BLUE, lw=2, label=r"$\theta(t)$")
    ax.axhline(THETA0, color=figstyle.GREY, lw=0.8, ls=":")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"HWP angle $\theta(t)$ (rad)")
    ax.set_xlim(0, t[-1])
    # Headroom so the legend clears the first theta/drive peak.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.16 * (hi - lo))
    figstyle.tidy_ticks(ax, "x")       # time axis: 4-5 stripped labels
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])   # theta: 5 labels
    ax.yaxis.set_major_formatter(figstyle.STRIP_FMT)
    ax.legend(loc="upper right", ncol=2, handlelength=1.6,
              handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "memristor_hwp_angle.pdf")
    figstyle.apply(base_fontsize=18)   # restore base for the remaining column figs


def _shade_family(n):
    """Blue-to-light family of n shades for monochrome parameter sweeps."""
    return [(s, s, 1.0) for s in np.linspace(0.0, 0.65, n)]


# -- Figure 4: frequency dependence (loop collapse) ---------------------------
def plot_effect_frequency():
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)

    omegas = [0.10, 0.30, 1.0, 3.0]
    for w, c in zip(omegas, _shade_family(len(omegas))):
        v, i = loop_last_cycles(omega=w, n_cycles=10, skip=7)
        ax.plot(v, i, color=c, lw=2, label=rf"$\omega={w:g}$")

    figstyle.zero_axes(ax)
    ax.set_xlabel(r"Input drive $v$")
    ax.set_ylabel(r"Output $i$")
    figstyle.unit_axis(ax, "both")
    ax.legend(loc="upper left", title=r"collapse as $\omega\!\uparrow$")
    fig.tight_layout()
    save(fig, "memristor_effect_frequency.pdf")


# -- Figure 5: feedback-strength dependence -----------------------------------
def plot_effect_coupling():
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)

    etas = [0.05, 0.10, 0.15, 0.20, 0.30]
    for e, c in zip(etas, _shade_family(len(etas))):
        v, i = loop_last_cycles(eta=e)
        ax.plot(v, i, color=c, lw=2, label=rf"$\eta={e:g}$")

    figstyle.zero_axes(ax)
    ax.set_xlabel(r"Input drive $v$")
    ax.set_ylabel(r"Output $i$")
    figstyle.unit_axis(ax, "both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "memristor_effect_coupling.pdf")


# -- Figure 6: relaxation-rate dependence --------------------------------------
def plot_effect_relaxation():
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)

    gammas = [0.03, 0.10, 0.30, 1.0]
    for g, c in zip(gammas, _shade_family(len(gammas))):
        v, i = loop_last_cycles(gamma=g, n_cycles=10, skip=7)
        ax.plot(v, i, color=c, lw=2, label=rf"$\gamma={g:g}$")

    figstyle.zero_axes(ax)
    ax.set_xlabel(r"Input drive $v$")
    ax.set_ylabel(r"Output $i$")
    figstyle.unit_axis(ax, "both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "memristor_effect_relaxation.pdf")


# -- Figure 7: phase-space portrait (memory vs drive) -------------------------
def plot_phase_space():
    """Phase-space portrait: memory angle theta vs input drive v over a settled
    cycle.  The closed loop (a tilted Lissajous ellipse) shows the memory lagging
    the drive -- the state-space view of the hysteresis."""
    fig, ax = plt.subplots(figsize=figstyle.SINGLE)

    t, theta = solve_theta(n_cycles=9)
    mask = t >= 7 * (2.0 * np.pi / OMEGA)
    v = drive(t[mask])
    th = theta[mask]

    ax.plot(v, th, color=figstyle.BLUE, lw=2)
    _arrows(ax, v, th, fracs=(0.25, 0.75), color=figstyle.BLUE)
    ax.axhline(THETA0, color=figstyle.GREY, lw=0.8, ls=":")
    ax.text(0.97, 0.06, r"$\theta_0$", transform=ax.transAxes,
            color=figstyle.GREY, ha="right", va="bottom")

    ax.set_xlabel(r"Input drive $v$")
    ax.set_ylabel(r"Memory angle $\theta$ (rad)")
    ax.set_xlim(-1.15, 1.15)
    fig.tight_layout()
    save(fig, "memristor_phase_space.pdf")


if __name__ == "__main__":
    print("Generating quantum-memristor figures (pinched-loop, publication style)...")
    plot_main_hysteresis()
    plot_time_evolution()
    plot_hwp_angle()
    plot_effect_frequency()
    plot_effect_coupling()
    plot_effect_relaxation()
    plot_phase_space()
    print("Done.")
