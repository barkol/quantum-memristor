r"""Temporal steering robustness of the photonic quantum memristor's output
qubit, computed by semidefinite programming (MOSEK).

Scenario (temporal steering, Chen et al. 2014): at time $t_k$ the output qubit
is measured in one of $N$ mutually unbiased bases (MUBs); the qubit then evolves
to $t_{k+1}$ under the device's loop decoherence -- modelled as an isotropic
depolarising (white-noise) channel of strength $d$, the regime in which "strong
loop decoherence effectively measures the looping photons" and returns the
classical-feedback limit.  The resulting temporal assemblage is

    sigma_{a|x}(eta) = 1/2 [ eta * Pi_{a|x} + (1-eta) * I/2 ],   eta = 1 - d,

with $\Pi_{a|x}$ the projector for outcome $a$ of MUB $x$ and reduced state I/2.

Temporal steering robustness (generalised robustness of steering, Cavalcanti &
Skrzypczyk): the minimal $t\ge0$ such that mixing the assemblage with weight $t$
of an arbitrary assemblage yields one with a local-hidden-state model.  As an SDP
over unnormalised LHS members sigma_lambda >= 0 (lambda a deterministic
outcome-assignment strategy):

    TSR = min  Tr( sum_lambda sigma_lambda ) - 1
    s.t. sigma_lambda >= 0,
         sum_{lambda: lambda_x = a} sigma_lambda  >=  sigma_{a|x}   for all a,x.

TSR > 0 certifies genuinely quantum temporal correlations (steerable memory);
TSR = 0 is the classical-feedback limit.  Solved with MOSEK (falls back to SCS
only if MOSEK is unavailable).

Run: python3 sim_temporal_steering.py
"""
import itertools
import os

import numpy as np
import cvxpy as cp

import figstyle
figstyle.apply()
import matplotlib.pyplot as plt

# ---- qubit MUB projectors -------------------------------------------------
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def projectors(pauli):
    """The two eigenprojectors {Pi_+, Pi_-} of a Pauli operator."""
    return [(I2 + s * pauli) / 2 for s in (+1, -1)]


MUBS = {"X": SX, "Y": SY, "Z": SZ}


def assemblage(eta, bases):
    """sigma_{a|x}(eta) for the depolarising-channel temporal scenario."""
    out = {}
    for x, b in enumerate(bases):
        for a, Pi in enumerate(projectors(MUBS[b])):
            out[(x, a)] = 0.5 * (eta * Pi + (1 - eta) * (I2 / 2))
    return out


def _solver():
    return cp.MOSEK if "MOSEK" in cp.installed_solvers() else cp.SCS


def steering_robustness(sigma, N):
    """Generalised temporal steering robustness via SDP (MOSEK)."""
    strategies = list(itertools.product((0, 1), repeat=N))   # 2^N det. strategies
    s = {lam: cp.Variable((2, 2), hermitian=True) for lam in strategies}
    cons = [s[lam] >> 0 for lam in strategies]
    for x in range(N):
        for a in (0, 1):
            lhs = sum(s[lam] for lam in strategies if lam[x] == a)
            cons.append(lhs - sigma[(x, a)] >> 0)
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(sum(s.values()))) - 1), cons)
    prob.solve(solver=_solver())
    return max(float(prob.value), 0.0)


def curve(bases, decoh_grid):
    N = len(bases)
    return np.array([steering_robustness(assemblage(1.0 - d, bases), N)
                     for d in decoh_grid])


def main():
    print("solver:", _solver(), "| installed:", cp.installed_solvers())
    d = np.linspace(0.0, 1.0, 41)            # loop-decoherence strength
    tsr3 = curve(["X", "Y", "Z"], d)         # 3 MUBs
    tsr2 = curve(["X", "Z"], d)              # 2 MUBs
    # report thresholds (last d with TSR>1e-6)
    for name, tsr in (("3 MUB", tsr3), ("2 MUB", tsr2)):
        thr = d[np.where(tsr > 1e-6)[0][-1]] if np.any(tsr > 1e-6) else 0.0
        print(f"  {name}: TSR(d=0)={tsr[0]:.4f}  steering vanishes near d~{thr:.3f}")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(d, tsr3, "-o", color=figstyle.BLUE, ms=4, lw=2,
            label=r"$N=3$ MUBs ($X,Y,Z$)")
    ax.plot(d, tsr2, "-s", color=figstyle.RED, ms=4, lw=2,
            label=r"$N=2$ MUBs ($X,Z$)")
    ax.axhline(0.0, color=figstyle.GREY, lw=0.8)
    ax.set_xlabel(r"loop-decoherence strength $d$")
    ax.set_ylabel(r"temporal steering robustness $\mathcal{R}_T$")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right")
    # shade the classical-feedback region where TSR collapses to 0
    figstyle.cap_ticks(fig)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "papers", "figures", "paper_tsr.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("saved", os.path.abspath(out))


if __name__ == "__main__":
    main()
