"""
CPHASE applied to the outputs of TWO INDEPENDENT quantum-memristor trajectories.

Uses the already-implemented quantum-trajectory simulator
(`memristor.QuantumMemristorSimulation`): each memristor emits a pure
single-qubit polarization state per step (stochastic measurement feedback).
We take the two output qubits A, B, send them into a controlled-phase gate
CZ = diag(1,1,1,-1) (phi = pi), and plot what the gate actually produces.

A CZ is diagonal in the computational basis, so it does NOT change the
Z-basis populations -- its effect is pure COHERENCE: it entangles the two
outputs. We therefore plot the entanglement (concurrence) and the resulting
maximal CHSH value of the post-gate two-qubit state along the trajectory.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np

import figstyle
figstyle.apply()
import matplotlib.pyplot as plt

import memristor as m

BLUE, RED, GREY = figstyle.BLUE, figstyle.RED, figstyle.GREY
THIRD = getattr(figstyle, "THIRD", "0.45")

# controlled-phase gate (phi = pi)  on |00>,|01>,|10>,|11>
CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)


def bloch_to_state(x, y, z):
    """Pure single-qubit state vector from a unit Bloch vector."""
    a = np.sqrt(max((1.0 + z) / 2.0, 0.0))          # |0> amplitude (real >= 0)
    if a < 1e-9:
        return np.array([0.0, 1.0], dtype=complex)  # z = -1  -> |1>
    b = (x + 1j * y) / (2.0 * a)
    psi = np.array([a, b], dtype=complex)
    return psi / np.linalg.norm(psi)


def concurrence_pure(psi4):
    """Concurrence of a pure 2-qubit state [c00,c01,c10,c11]."""
    c00, c01, c10, c11 = psi4
    return float(2.0 * abs(c00 * c11 - c01 * c10))


def cphase_outcomes(trA, trB):
    """Per-step concurrence and max-CHSH of CZ|psi_A>|psi_B>."""
    n = len(trA["t"])
    C = np.zeros(n)
    for i in range(n):
        pa = bloch_to_state(trA["sigma_x"][i], trA["sigma_y"][i], trA["sigma_z"][i])
        pb = bloch_to_state(trB["sigma_x"][i], trB["sigma_y"][i], trB["sigma_z"][i])
        psi = CZ @ np.kron(pa, pb)
        C[i] = concurrence_pure(psi)
    S = 2.0 * np.sqrt(1.0 + C ** 2)                 # Horodecki bound for pure states
    return C, S


from scipy.integrate import solve_ivp


def hwp_output_state(theta, alpha):
    """Output qubit of a memristor HWP U(theta) on input |R(alpha)> = [cos a, sin a].
    Closed form: |psi_out> = [cos(2 theta - alpha), sin(2 theta - alpha)]."""
    beta = 2.0 * theta - alpha
    return np.array([np.cos(beta), np.sin(beta)], dtype=complex)


def run_driven_memristor(t, A, omega, eta, gamma, theta0, phi=0.0):
    """Driven memristor with measurement-feedback memory (the verified
    basis_hysteresis Z-dynamics), modulated input angle alpha(t)=A sin(omega t+phi).
        d(theta)/dt = eta*cos(4 theta - 2 alpha) - gamma*(theta - theta0)
    Returns alpha(t), theta(t)."""
    def alpha(tt):
        return A * np.sin(omega * tt + phi)

    def rhs(tt, th):
        return eta * np.cos(4.0 * th - 2.0 * alpha(tt)) - gamma * (th - theta0)

    sol = solve_ivp(rhs, (t[0], t[-1]), [theta0], t_eval=t,
                    method="RK45", rtol=1e-9, atol=1e-11)
    return alpha(t), sol.y[0]


def main():
    # ---- two independent driven memristors: DIFFERENT modulation frequencies
    #      (the concurrence then beats at the difference frequency) ----
    t = np.linspace(0.0, 140.0, 3600)
    eta, gamma = 0.2, 0.035
    aA, thA = run_driven_memristor(t, A=1.0, omega=0.35, eta=eta, gamma=gamma, theta0=0.0, phi=0.0)
    aB, thB = run_driven_memristor(t, A=1.0, omega=0.55, eta=eta, gamma=gamma, theta0=0.0, phi=0.0)

    # ---- fixed, maximally-entangling controlled-phase gate (phi = pi) ----
    C = np.zeros(len(t))
    for i in range(len(t)):
        pa = hwp_output_state(thA[i], aA[i])
        pb = hwp_output_state(thB[i], aB[i])
        psi = CZ @ np.kron(pa, pb)
        C[i] = concurrence_pure(psi)
    S = 2.0 * np.sqrt(1.0 + C ** 2)

    tau = 1.0 / gamma                       # memory-relaxation (transient) timescale

    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(figstyle.SINGLE[0], figstyle.SINGLE[1] * 1.6),
        sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.5]))

    # (a) modulated inputs + memory build-up (the source of the transient)
    ax0.plot(t, aA, color=BLUE, lw=1.0, ls=":", label=r"input $\alpha_A(t)$")
    ax0.plot(t, thA, color=BLUE, lw=1.7, label=r"memory $\theta_A(t)$")
    ax0.plot(t, thB, color=RED, lw=1.7, label=r"memory $\theta_B(t)$")
    ax0.set_ylabel("angle (rad)")
    ax0.legend(loc="upper right", fontsize=7, ncol=3)
    ax0.set_title("Modulated inputs $\\to$ memory builds up (transient)", fontsize=10)

    # (b) concurrence after the FIXED maximally-entangling CZ
    ax1.axvspan(0, tau, color=THIRD, alpha=0.10)
    ax1.axvline(tau, color=GREY, ls=":", lw=1.0)
    ax1.text(tau, 1.06, r"  transient $\sim 1/\gamma$", color="0.35",
             fontsize=7.5, ha="left", va="top")
    ax1.plot(t, C, color=BLUE, lw=1.6)
    ax1.set_ylim(-0.05, 1.12)
    ax1.set_ylabel(r"concurrence $C$")
    ax1.set_xlabel(r"time $t$")
    ax1.set_title(r"After fixed CZ ($\phi=\pi$): "
                  r"$C=|\sin(4\theta_A-2\alpha_A)\,\sin(4\theta_B-2\alpha_B)|$",
                  fontsize=9.5)

    fig.tight_layout()
    import os
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures", "cphase_outcomes.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print("steady-state mean C (t>2/gamma):",
          round(float(C[t > 2 * tau].mean()), 3),
          " peak C:", round(float(C.max()), 3))
    print("saved:", out)


if __name__ == "__main__":
    main()
