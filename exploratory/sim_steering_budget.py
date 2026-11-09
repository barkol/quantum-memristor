r"""Temporal-steering error budget for the coherent delay-loop architecture.

Physical model. Between the measurement of output photon $k$ (time $t_k$) and
that of photon $k{+}1$ (time $t_{k+1}$), the correlation-carrying photon
completes $n_p{=}2$ delay-loop passes and participates in $n_g{=}2$ joint
applications of the linear-optical controlled-phase gate.  Loop loss is
modelled by replacing a lost looping photon by a classical
feedback event, i.e. the memory qubit is traced out, which whitens the
assemblage.  Each gate application contributes a process-fidelity
visibility $V_g$.  The assemblage seen by the steering test is then

    sigma_{a|x}(v) = 1/2 [ v * Pi_{a|x} + (1-v) * I/2 ],
    v = V_g^{n_g} * eta_loop^{n_p},         eta_loop = 10^(-L_dB/10),

with $L_{dB}$ the per-pass loop loss.  Steering survives iff v > 1/sqrt(N)
for N mutually unbiased settings, so the loss budget is

    L_dB < (10 / n_p) * log10( V_g^{n_g} * sqrt(N) ).

The script computes the generalised temporal steering robustness R_T by SDP
(MOSEK) along physical loss axes and prints the budget table used in the
paper.  Deterministic.  Output: figures/paper_steering_budget.pdf

Data (SDP sweeps) in figures/data/sim_steering_budget.npz, computed only if missing
(checkpointed per sweep in figures/data/sim_steering_budget.part/); draw() only reads it.
"""
import itertools
from pathlib import Path

import numpy as np
import cvxpy as cp


N_PASSES = 2      # delay-loop passes between consecutive output measurements
N_GATES = 2       # joint controlled-phase gate applications

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
MUBS = {"X": SX, "Y": SY, "Z": SZ}


def projectors(pauli):
    return [(I2 + s * pauli) / 2 for s in (+1, -1)]


def assemblage(v, bases):
    out = {}
    for x, b in enumerate(bases):
        for a, Pi in enumerate(projectors(MUBS[b])):
            out[(x, a)] = 0.5 * (v * Pi + (1 - v) * (I2 / 2))
    return out


def _solver():
    return cp.MOSEK if "MOSEK" in cp.installed_solvers() else cp.SCS


def steering_robustness(sigma, N):
    strategies = list(itertools.product((0, 1), repeat=N))
    s = {lam: cp.Variable((2, 2), hermitian=True) for lam in strategies}
    cons = [s[lam] >> 0 for lam in strategies]
    for x in range(N):
        for a in (0, 1):
            lhs = sum(s[lam] for lam in strategies if lam[x] == a)
            cons.append(lhs - sigma[(x, a)] >> 0)
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(sum(s.values()))) - 1), cons)
    prob.solve(solver=_solver())
    return max(float(prob.value), 0.0)


def visibility(loss_db, v_gate):
    eta_loop = 10.0 ** (-loss_db / 10.0)
    return (v_gate ** N_GATES) * (eta_loop ** N_PASSES)


def loss_budget_db(N, v_gate):
    """Maximal per-pass loop loss (dB) with steering still certifiable."""
    v_thr = 1.0 / np.sqrt(N)
    arg = (v_gate ** N_GATES) / v_thr
    if arg <= 1.0:
        return -np.inf                      # gate alone already kills steering
    return (10.0 / N_PASSES) * np.log10(arg)


BASES = {2: ["X", "Z"], 3: ["X", "Y", "Z"]}

V_GATES = [1.00, 0.95, 0.90]

# ── Data: SDP sweeps -> figures/data/sim_steering_budget.npz (compute if missing) ──
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "figures" / "data"
CACHE = DATA_DIR / "sim_steering_budget.npz"
PART = DATA_DIR / "sim_steering_budget.part"      # per-sweep checkpoints, resumable


def _sweep(key, xs, fn):
    """One SDP sweep, checkpointed as PART/<key>.npy so an interrupted run resumes."""
    PART.mkdir(parents=True, exist_ok=True)
    f = PART / f"{key}.npy"
    if f.exists():
        return np.load(f)
    y = np.array([fn(x) for x in xs])
    np.save(f, y)
    return y


def compute():
    # universal curve R_T(v) (kept for reference, not drawn)
    v_grid = np.linspace(0.4, 1.0, 61)
    out = dict(v_grid=v_grid)
    for N, B in BASES.items():
        out[f"rt_v_{N}"] = _sweep(f"rt_v_{N}", v_grid, lambda v: steering_robustness(assemblage(v, B), N))
    # physical loss axis for three gate visibilities
    loss_grid = np.linspace(0.0, 1.6, 33)
    out["loss_grid"] = loss_grid
    for V in V_GATES:
        for N, B in BASES.items():
            k = f"rt_loss_{int(round(100 * V))}_{N}"
            out[k] = _sweep(k, loss_grid, lambda L: steering_robustness(assemblage(visibility(L, V), B), N))
    out["budget_closed_form"] = np.array([[loss_budget_db(N, V) for N in (2, 3)] for V in V_GATES])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)


def zero_crossing(x, y, V, tol=1e-6):
    """Loss where R_T reaches zero: R_T is linear in the visibility v above threshold, so the line through the
    last two grid points with R_T > 0 is extrapolated to zero in v and converted back to dB."""
    i = int(np.argmax(y <= tol))
    if i < 2:
        return np.nan
    v1, v2 = visibility(x[i - 2], V), visibility(x[i - 1], V)
    v0 = v2 - y[i - 1] * (v1 - v2) / (y[i - 2] - y[i - 1])
    return -(10.0 / N_PASSES) * np.log10(v0 / V ** N_GATES)


def draw():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import figstyle_pra as fs
    fs.apply()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
    d = np.load(CACHE)
    L = d["loss_grid"]
    col = {3: fs.C_BLUE, 2: fs.C_RED}
    styles = {1.00: "-", 0.95: "--", 0.90: ":"}
    fig = plt.figure(figsize=(fs.COLW, 1.75))
    ax = fig.add_axes([0.105, 0.19, 0.875, 0.775])
    for N in (3, 2):
        for V in V_GATES:
            ax.plot(L, d[f"rt_loss_{int(round(100 * V))}_{N}"], color=col[N], ls=styles[V], lw=1.0)
    ax.set_xlabel(r"per-pass loop loss (dB)")
    ax.set_ylabel(r"$\mathcal{R}_T$")
    h = ([Line2D([], [], color="0.15", ls=styles[V], lw=0.9, label=rf"$V_g={V:.2f}$") for V in V_GATES]
         + [Line2D([], [], color=col[3], lw=1.0, label=r"$N=3$"), Line2D([], [], color=col[2], lw=1.0, label=r"$N=2$")])
    ax.legend(handles=h, ncol=2, loc="upper right", columnspacing=1.0, handlelength=2.2, labelspacing=0.25)
    ax.set_xlim(-0.03, 1.63)
    ax.set_yticks([0.0, 0.1, 0.2])
    ax.set_yticklabels(["0", "0.1", "0.2"])
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xticks([0, 0.4, 0.8, 1.2, 1.6])
    ax.set_xticklabels(["0", "0.4", "0.8", "1.2", "1.6"])
    ax.set_ylim(-0.012, 0.285)
    out = REPO / "figures" / "paper_steering_budget.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    if not CACHE.exists():
        compute()
    d = np.load(CACHE)
    print("R_T at zero loss:", {N: float(d[f"rt_loss_100_{N}"][0]) for N in (2, 3)})
    print("\nBudget table: maximal per-pass loop loss (dB) with R_T > 0")
    print(f"(n_passes={N_PASSES}, n_gates={N_GATES}; threshold v > 1/sqrt(N))")
    for V in V_GATES:
        row = []
        for N in (2, 3):
            zc = zero_crossing(d["loss_grid"], d[f"rt_loss_{int(round(100 * V))}_{N}"], V)
            row.append(f"N={N}: closed form {loss_budget_db(N, V):6.3f} dB, SDP zero crossing {zc:6.3f} dB")
        print(f"  V_gate={V:.2f}  " + "   ".join(row))
    # CHSH (spatial witness) needs two-qubit visibility > 1/sqrt(2): same as N=2
    print("\nCHSH spatial witness threshold (visibility 1/sqrt2) == N=2 column.")
    draw()
