r"""Figure 1 (thesis figure): criterion -> device -> memory law -> steering test.

Four mini-panels built from the paper's actual models, joined by arrows:
 (a) the operational criterion (classical-wave regime vs single-photon regime);
 (b) the feedback device (HWP whose angle is the leaky-integrator memory);
 (c) the exponential memory kernel of the mean register and the pinched hysteresis loop it
     generates (i = G(theta) v, leaky integrator, settled cycle);
 (d) the steering test: steering parameter S_3 = 3 eta^2 versus per-pass loop loss
     (N = 3 MUBs, ideal gates) with the budget of 1.19 dB for no-clicks given random outcomes.

Deterministic. Output: figures/paper_thesis.pdf (3.40 in = \columnwidth, fonts at final size).
Data (ODE + SDP sweep) in figures/data/make_fig1_thesis.npz, computed only if missing; draw() only reads it.
"""
import itertools
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import cvxpy as cp


# ---- data cache: ODE + SDP sweep run only if the cache is missing ---------
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "figures" / "data"
CACHE = DATA_DIR / "make_fig1_thesis.npz"

# ---- panel (d) machinery: R_T vs loss (N=3, V_g=1) ------------------------
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

def steering_robustness(v, paulis):
    N = len(paulis)
    sig = {}
    for x, B in enumerate(paulis):
        for a, s in enumerate((+1, -1)):
            Pi = (I2 + s * B) / 2
            sig[(x, a)] = 0.5 * (v * Pi + (1 - v) * I2 / 2)
    strategies = list(itertools.product((0, 1), repeat=N))
    sv = {lam: cp.Variable((2, 2), hermitian=True) for lam in strategies}
    cons = [sv[lam] >> 0 for lam in strategies]
    for x in range(N):
        for a in (0, 1):
            cons.append(sum(sv[lam] for lam in strategies if lam[x] == a) - sig[(x, a)] >> 0)
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(sum(sv.values()))) - 1), cons)
    prob.solve(solver=cp.MOSEK if "MOSEK" in cp.installed_solvers() else cp.SCS)
    return max(float(prob.value), 0.0)

def compute():
    # ---- panel (c) data: kernel + pinched loop (app:iv model) -------------
    eta, gamma, om, th0 = 0.15, 0.10, 0.30, np.pi / 8
    tau = np.linspace(0, 40, 200)
    kernel = eta * np.exp(-gamma * tau)

    def rhs(t, th):
        return eta * np.sin(om * t) - gamma * (th - th0)
    T = 2 * np.pi / om
    t_eval = np.linspace(0, 10 * T, 4000)
    sol = solve_ivp(rhs, (0, t_eval[-1]), [th0], t_eval=t_eval, rtol=1e-9)
    mask = sol.t >= 9 * T
    v_dr = np.sin(om * sol.t[mask])
    i_out = np.sin(2 * sol.y[0][mask]) ** 2 * v_dr

    # ---- panel (d) data: R_T vs loss (N=3, V_g=1), consistency check of the drawn S_3 budget ----
    loss = np.linspace(0, 1.6, 17)
    rt = np.array([steering_robustness((10 ** (-L / 10)) ** 2, [SX, SY, SZ]) for L in loss])

    # ---- panel (d) drawn curve: S_3 = 3 eta_loop^2 and its crossing with sqrt(3) ----
    loss_fine = np.linspace(0, 1.6, 321)
    S3 = 3 * 10 ** (-2 * loss_fine / 10)
    thr = 5 * np.log10(np.sqrt(3))          # S_3 = sqrt(3)  <=>  1.193 dB
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, tau=tau, kernel=kernel, v_dr=v_dr, i_out=i_out, loss=loss, rt=rt,
             loss_fine=loss_fine, S3=S3, thr=thr)


def draw():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import figstyle_pra as fs
    from paneltags_rows import tag, place
    fs.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
    d = np.load(CACHE)
    BLUE, RED = fs.C_BLUE, fs.C_RED
    W, H = fs.COLW, 2.42
    fig = plt.figure(figsize=(W, H))

    def axes_in(x0, y0, w, h):
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H])

    # ---------------- row 1: (a) criterion card, (b) device ----------------
    YC0, YC1 = 1.52, 2.24                       # card band (inches)
    axa = axes_in(0.04, YC0, 1.42, YC1 - YC0)
    axa.set_xlim(0, 1.42); axa.set_ylim(0, YC1 - YC0); axa.axis("off")
    axa.add_patch(FancyBboxPatch((0.03, 0.03), 1.36, 0.66, boxstyle="round,pad=0.02",
                                 fc="none", ec="0.45", lw=0.6))
    axa.text(0.71, 0.56, r"$\langle n\rangle \sim 10^{6}$--$10^{7}$", ha="center", va="center", fontsize=7)
    axa.text(0.71, 0.42, "classical wave", ha="center", va="center", fontsize=7, style="italic")
    axa.plot([0.16, 1.26], [0.335, 0.335], color="0.55", lw=0.6, ls=(0, (4, 3)))
    axa.text(0.71, 0.235, "nonclassical light,", ha="center", va="center", fontsize=7)
    axa.text(0.71, 0.11, "entanglement", ha="center", va="center", fontsize=7)
    tag(axa, "(a)", row=0, title="criterion")

    axb = axes_in(1.82, YC0, 1.56, YC1 - YC0)
    axb.set_xlim(0, 1.56); axb.set_ylim(0, YC1 - YC0); axb.set_aspect("equal"); axb.axis("off")
    yb = 0.53                                     # optical axis of (b)
    ak = dict(arrowstyle="-|>,head_width=0.18,head_length=0.32", color="0.2", lw=0.8, shrinkA=0, shrinkB=0)
    axb.annotate("", xy=(0.36, yb), xytext=(0.0, yb), arrowprops=ak)
    axb.add_patch(FancyBboxPatch((0.38, yb - 0.11), 0.56, 0.22, boxstyle="round,pad=0.02",
                                 fc=fs.tint(BLUE, 0.85), ec=BLUE, lw=0.8))
    axb.text(0.66, yb, r"HWP$(\theta)$", ha="center", va="center", fontsize=7)
    axb.annotate("", xy=(1.30, yb), xytext=(0.96, yb), arrowprops=ak)
    axb.add_patch(Circle((1.37, yb), 0.05, fc="0.2", ec="none"))
    axb.text(1.37, yb + 0.09, "det.", ha="center", va="bottom", fontsize=7)
    axb.annotate("", xy=(0.80, yb - 0.145), xytext=(1.37, yb - 0.065),
                 arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.32", color=RED, lw=0.8,
                                 shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=-0.35"))
    axb.text(1.12, yb - 0.30, "feedback", ha="center", va="center", fontsize=7)
    axb.text(0.78, 0.07, r"$\dot\theta=\eta N(t)-\gamma(\theta-\theta_0)$", ha="center", va="center", fontsize=7)
    tag(axb, "(b)", row=0, title="device", x0=1.74 / W)

    # flow arrow (a) -> (b), on the optical axis of (b)
    yab = (YC0 + yb) / H
    fig.add_artist(FancyArrowPatch((1.50 / W, yab), (1.76 / W, yab), transform=fig.transFigure,
                                   arrowstyle="-|>,head_width=0.18,head_length=0.32", mutation_scale=10,
                                   color="0.3", lw=0.9))

    # ---------------- row 2: (c) memory law, (d) steering test ----------------
    YP0, YP1 = 0.30, 1.18
    axc = axes_in(0.20, YP0, 1.26, YP1 - YP0)
    axc.plot(d["tau"], d["kernel"], color=BLUE, lw=1.0)
    axc.set_xlabel(r"$\tau$", labelpad=1)
    axc.set_ylabel(r"$\mathcal{K}(\tau)$", labelpad=2)
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_xlim(-1.5, 41.5); axc.set_ylim(0, 0.16)
    axi = axc.inset_axes([0.47, 0.42, 0.50, 0.53])
    axi.plot(d["v_dr"], d["i_out"], color=RED, lw=0.8)
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color("0.6"); sp.set_linewidth(0.5)
    axc.text(0.97, 0.39, "open loop,\nrate $\\gamma$", ha="right", va="top", ma="right",
             transform=axc.transAxes, fontsize=6, linespacing=1.1)
    tag(axc, "(c)", row=1, title="memory law")

    axd = axes_in(2.08, YP0, 1.25, YP1 - YP0)
    thr = float(d["thr"])
    axd.plot(d["loss_fine"], d["S3"], color=BLUE, lw=1.0)
    axd.axhline(np.sqrt(3), color="0.4", lw=0.5, ls=":")
    axd.axvline(thr, color="0.4", lw=0.5, ls=":")
    axd.text(thr - 0.05, 2.98, "no-clicks:\nrandom", ha="right", va="top", ma="right", fontsize=6, linespacing=1.1)
    axd.set_xlabel("loop loss (dB)", labelpad=1)
    axd.set_ylabel(r"$\mathcal{S}_3$", labelpad=1)
    axd.set_xlim(0, 1.6); axd.set_ylim(1.3, 3.1)
    axd.set_xticks([0, thr]); axd.set_xticklabels(["0", "1.19"])
    axd.set_yticks([np.sqrt(3), 3]); axd.set_yticklabels([r"$\sqrt{3}$", "3"])
    tag(axd, "(d)", row=1, title="steering test")

    # flow arrow (b) -> (c): down through the gap between the rows, between the columns
    xm = 1.62 / W
    fig.add_artist(FancyArrowPatch((xm, (YC0 - 0.02) / H), (xm, (YP1 + 0.12) / H), transform=fig.transFigure,
                                   arrowstyle="-|>,head_width=0.18,head_length=0.32", mutation_scale=10,
                                   color="0.3", lw=0.9))
    place(fig, ytop={0: (H - 0.13) / H})
    out = REPO / "figures" / "paper_thesis.pdf"
    fig.savefig(out)
    print("Saved:", out)


if __name__ == "__main__":
    if not CACHE.exists():
        compute()
    d = np.load(CACHE)
    i = int(np.argmax(d["rt"] <= 1e-6))
    print(f"R_T(0 dB) = {d['rt'][0]:.4f}; R_T first zero on the 0.1 dB grid at {d['loss'][i]:.2f} dB;"
          f" S_3 = sqrt(3) at {float(d['thr']):.3f} dB")
    draw()
