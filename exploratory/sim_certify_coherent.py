r"""Certification of the COHERENT delay-loop memristor completed: the ideal
assemblage visibility is DERIVED (not assumed) from the pair-run protocol,
and the certificate is verified end-to-end against the loss budget.

Pair run (dedicated certification run of the paper): photons k and k+1 are
injected into an otherwise empty loop and interact through n_g = 2 joint
controlled-phase gate applications before both are switched out; the two
applications compose to a single CZ(Phi) with Phi = phi_1 + phi_2 set by the
program. The ideal pre-measurement state is

    |psi(Phi)> = CZ(Phi) |+>|+>.

At Phi = pi this is maximally entangled: measuring photon k in any basis
leaves photon k+1 in a PURE conditional state, i.e. the temporal assemblage
has visibility v = 1 EXACTLY. The script:

  1. computes R_T(Phi) for the ideal pair run (N = 2, 3 MUBs) -- maximal at
     Phi = pi, matching R_T(v=1) of the budget model;
  2. applies the budget noise channel (v = V_g^2 eta_loop^2) to the stored
     photon at Phi = pi and confirms the R_T zero crossings reproduce the
     closed-form loss budget of Eq. (loss_budget);
  3. prints the certified memory depth m* = floor( ln sqrt(N) / (-ln v_pass) )
     implied by the geometric decay v(m) = v_pass^m of the coherent device's
     memory kernel.

Deterministic. Output: figures/paper_device_certification.pdf
(panel (a): coherent pair run; panel (b): classical-feedback device under
the same certificate, from part1/sim_certify_feedback.py).

Data in figures/data/sim_certify_coherent.npz, computed only if missing (each stage checkpointed in
figures/data/sim_certify_coherent.part/, resumable); draw() only reads it.
"""
import itertools
from pathlib import Path

import numpy as np
import cvxpy as cp


I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
MUBS = {"X": SX, "Y": SY, "Z": SZ}
BASES = {2: ["X", "Z"], 3: ["X", "Y", "Z"]}


def projectors(pauli):
    return [(I2 + s * pauli) / 2 for s in (+1, -1)]


def steering_robustness(sigma, N):
    strategies = list(itertools.product((0, 1), repeat=N))
    s = {lam: cp.Variable((2, 2), hermitian=True) for lam in strategies}
    cons = [s[lam] >> 0 for lam in strategies]
    for x in range(N):
        for a in (0, 1):
            lhs = sum(s[lam] for lam in strategies if lam[x] == a)
            cons.append(lhs - sigma[(x, a)] >> 0)
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(sum(s.values()))) - 1), cons)
    prob.solve(solver=cp.MOSEK if "MOSEK" in cp.installed_solvers() else cp.SCS)
    return max(float(prob.value), 0.0)


def pair_state(phi):
    """|psi(Phi)> = CZ(Phi)|+>|+> as a 4x4 density matrix."""
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi = np.kron(plus, plus)
    cz = np.diag([1, 1, 1, np.exp(1j * phi)]).astype(complex)
    psi = cz @ psi
    return np.outer(psi, psi.conj())


def depolarize_B(rho2, v):
    """Isotropic depolarizing channel with retained visibility v on qubit B."""
    rho_B_white = np.kron(np.trace(rho2.reshape(2, 2, 2, 2), axis1=1, axis2=3),
                          I2 / 2)
    # partial trace over B, then re-tensor with I/2: Tr_B rho (x) I/2
    rho_A = np.trace(rho2.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    white = np.kron(rho_A, I2 / 2)
    return v * rho2 + (1 - v) * white


def assemblage_from_state(rho2, bases):
    """sigma_{a|x} = Tr_A[(Pi_{a|x} (x) I) rho] with p(x)=uniform."""
    sig = {}
    for x, b in enumerate(bases):
        for a, Pi in enumerate(projectors(MUBS[b])):
            M = np.kron(Pi, I2)
            R = (M @ rho2 @ M).reshape(2, 2, 2, 2)
            sig[(x, a)] = np.trace(R, axis1=0, axis2=2)   # trace out A
    return sig


# ── Data: SDP sweeps, bisections and the feedback Monte Carlo -> npz (compute if missing) ──
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "figures" / "data"
CACHE = DATA_DIR / "sim_certify_coherent.npz"
PART = DATA_DIR / "sim_certify_coherent.part"


def _stage(name, fn):
    """Run one stage once; its result dict is checkpointed as PART/<name>.npz so a restart skips it."""
    PART.mkdir(parents=True, exist_ok=True)
    f = PART / f"{name}.npz"
    if f.exists():
        return dict(np.load(f))
    r = fn()
    np.savez(f, **r)
    return r


def _stage_phi():
    # 1. ideal pair run: R_T versus total gate phase
    phis = np.linspace(0.0, 2 * np.pi, 41)
    out = dict(phis=phis)
    for N in (2, 3):
        out[f"rt_phi_{N}"] = np.array([steering_robustness(assemblage_from_state(pair_state(phi), BASES[N]), N)
                                       for phi in phis])
        out[f"rt_pi_{N}"] = steering_robustness(assemblage_from_state(pair_state(np.pi), BASES[N]), N)
    return out


def _stage_bisect():
    # 2. noisy pair run at Phi=pi: SDP zero crossing of R_T vs closed-form budget
    out = {}
    for Vg in (1.00, 0.90):
        for N in (2, 3):
            lo, hi = 0.0, 3.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                v = (Vg ** 2) * (10 ** (-mid / 10.0)) ** 2
                rho = depolarize_B(pair_state(np.pi), v)
                rt = steering_robustness(assemblage_from_state(rho, BASES[N]), N)
                if rt > 1e-7:
                    lo = mid
                else:
                    hi = mid
            k = f"{int(round(100 * Vg))}_{N}"
            out["thr_sdp_" + k] = lo
            out["thr_closed_" + k] = (10.0 / 2) * np.log10((Vg ** 2) * np.sqrt(N))
    return out


def _stage_feedback():
    # classical-feedback device, trajectory model of part1/sim_certify_feedback.py (seeded, about 30 s)
    import importlib.util
    spec = importlib.util.spec_from_file_location("scf", REPO / "part1" / "sim_certify_feedback.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    r = m.main()
    return dict(fb_eta=r["eta"], fb_S3=r["S3_eta"], fb_RT=r["rt_3"], fb_sig3=r["sig_3"], fb_RT2=r["rt_2"],
                fb_sig_reconf=r["sig_reconf"])


def compute():
    out = {}
    for name, fn in (("phi", _stage_phi), ("bisect", _stage_bisect), ("feedback", _stage_feedback)):
        out.update(_stage(name, fn))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)


def report():
    d = np.load(CACHE)
    phis = d["phis"]
    for N in (2, 3):
        i = int(np.argmax(d[f"rt_phi_{N}"]))
        print(f"ideal pair run N={N}: max R_T = {d[f'rt_phi_{N}'][i]:.4f} at Phi = {phis[i]/np.pi:.2f} pi;"
              f" R_T(Phi=pi) = {float(d[f'rt_pi_{N}']):.4f}")
    print("\nnoisy pair run at Phi=pi (v = V_g^2 eta_loop^2):")
    for Vg in (1.00, 0.90):
        for N in (2, 3):
            k = f"{int(round(100 * Vg))}_{N}"
            print(f"  V_g={Vg:.2f} N={N}: SDP threshold {float(d['thr_sdp_' + k]):.3f} dB/pass "
                  f"vs closed-form budget {float(d['thr_closed_' + k]):.3f} dB/pass")
    print("\nclassical-feedback device: S_3(eta) =", np.round(d["fb_S3"], 3), " R_T(N=3) =", f"{float(d['fb_RT']):.2e}")
    # 3. certified memory depth of the coherent device (closed form)
    print("\ncertified memory depth m* = floor(ln sqrt(N) / (-ln v_pass)):")
    for Vg, Ldb in ((1.0, 0.3), (0.95, 0.3), (0.90, 0.1)):
        v_pass = (Vg ** 2) * (10 ** (-Ldb / 10.0)) ** 2
        for N in (2, 3):
            mstar = int(np.floor(np.log(np.sqrt(N)) / (-np.log(v_pass))))
            print(f"  V_g={Vg:.2f}, {Ldb:.1f} dB/pass, N={N}: v_pass={v_pass:.3f}, m* = {mstar}")


def draw():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import figstyle_pra as fs
    from paneltags_rows import tag, place
    fs.apply()
    import matplotlib.pyplot as plt
    d = np.load(CACHE)
    x = d["phis"] / np.pi
    fig = plt.figure(figsize=(fs.COLW, 1.50))
    axa = fig.add_axes([0.095, 0.225, 0.375, 0.635])
    axb = fig.add_axes([0.610, 0.225, 0.375, 0.635])
    # (a) two curves, direct labels in matching colours
    axa.plot(x, d["rt_phi_3"], color=fs.C_BLUE, lw=1.0)
    axa.plot(x, d["rt_phi_2"], color=fs.C_RED, lw=1.0, ls="--")
    axa.axvline(1.0, color="0.55", lw=0.5, ls=":", zorder=0)
    axa.text(0.5, 0.262, r"$N=3$", color=fs.C_BLUE, ha="right", va="center", fontsize=6)
    axa.text(1.0, 0.085, r"$N=2$", color=fs.C_RED, ha="center", va="center", fontsize=6, zorder=3,
             bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    axa.set_xlabel(r"total gate phase $\Phi/\pi$")
    axa.set_ylabel(r"$\mathcal{R}_T$")
    axa.set_xlim(-0.05, 2.05)
    axa.set_ylim(-0.01, 0.30)
    axa.set_xticks([0, 0.5, 1, 1.5, 2])
    axa.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
    axa.set_yticks([0, 0.1, 0.2])
    axa.set_yticklabels(["0", "0.1", "0.2"])
    tag(axa, "(a)", title="coherent pair run (ideal)")
    # (b) data prominent; bound and noise floor as thin grey reference lines, labelled next to them
    eta = d["fb_eta"]
    axb.plot(eta, d["fb_S3"], "o-", color=fs.C_BLUE, lw=1.0, ms=2.6)
    axb.axhline(np.sqrt(3), color="0.45", lw=0.6, ls="--", zorder=1)
    axb.axhline(float(d["fb_RT"]), color="0.45", lw=0.6, ls=":", zorder=1)
    axb.text(0.45, 1.66, r"classical bound $\sqrt{3}$", color="0.3", ha="center", va="top", fontsize=6)
    axb.text(0.45, 1.00, r"$\mathcal{S}_3$ (trajectory model)", color=fs.C_BLUE, ha="center", va="bottom", fontsize=6)
    axb.text(0.45, 0.08, r"$\mathcal{R}_T$ (MC noise)", color="0.3", ha="center", va="bottom", fontsize=6)
    axb.set_xlabel(r"feedback strength $\eta$")
    axb.set_ylabel(r"$\mathcal{S}_3$, $\mathcal{R}_T$")
    axb.set_xlim(0.03, 0.87)
    axb.set_ylim(-0.08, 1.95)
    axb.set_xticks([0.2, 0.4, 0.6, 0.8])
    axb.set_xticklabels(["0.2", "0.4", "0.6", "0.8"])
    axb.set_yticks([0, 0.5, 1, 1.5])
    axb.set_yticklabels(["0", "0.5", "1", "1.5"])
    tag(axb, "(b)", title="classical-feedback device")
    place(fig)
    out = REPO / "figures" / "paper_device_certification.pdf"
    fig.savefig(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    if not CACHE.exists():
        compute()
    report()
    draw()
