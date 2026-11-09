#!/usr/bin/env python3
"""Two-panel fully-quantum CHSH figure for PRX Quantum.

Reuses VERIFIED computations:
 - LEFT  : concurrence C(t) of two driven memristors (different omega) entangled
           by a fixed CZ (phi=pi), beating at the difference frequency.
           Exactly anim_cphase / cphase_two_memristors.main().
 - RIGHT : Horodecki CHSH S vs CPHASE phase phi for |+>|+> -> CPHASE -> S=2sqrt(u1+u2).
           Exactly anim_chsh.
No physics/parameters/seeds changed.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local figstyle
import figstyle
# Single-column figure included at width=\columnwidth (~3.4 in in the REVTeX
# reprint). The figure is drawn at native ~3.4 in width so fonts print 1:1:
# body/labels 9.5 pt, tick labels raised to ~8.4 pt (above the 8 pt floor).
figstyle.apply(base_fontsize=9.5)
import matplotlib.pyplot as plt

import cphase_two_memristors as cp

BLUE, RED, GREY = figstyle.BLUE, figstyle.RED, figstyle.GREY
THIRD = getattr(figstyle, "THIRD", "0.35")

# ====================================================================
# DATA CACHE: the expensive panel data are computed only when the cache
# is missing; the plotting below uses only the cached arrays/scalars.
# ====================================================================
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / "figures" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "make_fq_figure.npz"

if CACHE.exists():
    d = np.load(CACHE, allow_pickle=False)
    t, C, phis, Sv = d["t"], d["C"], d["phis"], d["Sv"]
    tau = float(d["tau"]); peakC = float(d["peakC"]); peakS = float(d["peakS"])
else:
    # ================================================================
    # LEFT panel data: exact reuse of cphase_two_memristors.main / anim_cphase
    # ================================================================
    t = np.linspace(0.0, 140.0, 3600)
    eta, gamma = 0.2, 0.035
    aA, thA = cp.run_driven_memristor(t, A=1.0, omega=0.35, eta=eta, gamma=gamma, theta0=0.0, phi=0.0)
    aB, thB = cp.run_driven_memristor(t, A=1.0, omega=0.55, eta=eta, gamma=gamma, theta0=0.0, phi=0.0)

    C = np.zeros(len(t))
    for i in range(len(t)):
        pa = cp.hwp_output_state(thA[i], aA[i])
        pb = cp.hwp_output_state(thB[i], aB[i])
        psi = cp.CZ @ np.kron(pa, pb)
        C[i] = cp.concurrence_pure(psi)
    tau = 1.0 / gamma
    peakC = float(C.max())

    # ================================================================
    # RIGHT panel data: exact reuse of anim_chsh (Horodecki via T-matrix)
    # ================================================================
    import qutip as qt
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi_in = qt.tensor(plus, plus)
    paulis = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]

    def cphase(phi):
        return qt.Qobj(np.diag([1, 1, 1, np.exp(1j * phi)]), dims=[[2, 2], [2, 2]])

    phis = np.linspace(0.0, np.pi, 130)
    Sv = []
    for phi in phis:
        rho = qt.ket2dm(cphase(phi) * psi_in)
        T = np.array([[qt.expect(qt.tensor(a, b), rho).real for b in paulis] for a in paulis])
        ev = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
        Sv.append(2.0 * np.sqrt(max(ev[0] + ev[1], 0.0)))
    Sv = np.array(Sv)
    peakS = float(Sv.max())

    np.savez(CACHE, t=t, C=C, phis=phis, Sv=Sv, tau=tau,
             peakC=peakC, peakS=peakS)

TS = 2.0 * np.sqrt(2.0)

# ====================================================================
# FIGURE  (single-column two-panel stack, house style)
# ====================================================================
# cphase_two_memristors calls figstyle.apply() (18 pt default) at import
# time, clobbering the size set above -- re-apply the print size here.
figstyle.apply(base_fontsize=9.5)
# Keep the smallest type (tick labels) at/above the 8 pt effective floor
# when printed 1:1 at \columnwidth.
plt.rcParams["xtick.labelsize"] = 8.4
plt.rcParams["ytick.labelsize"] = 8.4
fig, axL = plt.subplots(figsize=(3.44, 2.4))  # single panel

# ---- LEFT: concurrence beating in as the memory builds ----
axL.axvspan(0, tau, color=THIRD, alpha=0.10)
axL.axvline(tau, color=GREY, ls=":", lw=1.1, ymax=0.84)  # stop below the 'transient' label band
axL.axhline(1.0, color=GREY, lw=0.9, zorder=0)
axL.plot(t, C, color=BLUE, lw=2.0)
_F = plt.rcParams["font.size"]
axL.text(2, 1.10, r"transient $\sim 1/\gamma$", color="black",
         ha="left", va="center", fontsize=0.92 * _F)
axL.text(0.97, 1.005, r"$C=1$", color="black", ha="right", va="bottom",
         transform=axL.get_yaxis_transform(), fontsize=0.92 * _F)
axL.set_xlim(0, 140)
axL.set_ylim(0, 1.2)          # concurrence >= 0; round limits so cap_ticks keeps 0..1.2
axL.set_xlabel(r"time $t$")
axL.set_ylabel(r"concurrence $C(t)$")


# ---- RIGHT: CHSH S vs CPHASE phase ----


fig.tight_layout(pad=0.25, h_pad=1.3)
figstyle.cap_ticks(fig)          # 4-5 round, stripped ticks (1 not 1.00)
_OUT = str(Path(__file__).resolve().parent.parent / "figures" / "paper_fully_quantum.pdf")
fig.savefig(_OUT, bbox_inches="tight", pad_inches=0.03)
fig.savefig(_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

print(f"PEAK C = {peakC:.4f}")
print(f"PEAK S = {peakS:.4f}   (Tsirelson {TS:.4f})")
print(f"S(phi=0) = {Sv[0]:.4f}, S(phi=pi) = {Sv[-1]:.4f}")
print("saved", _OUT)
