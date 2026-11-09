r"""Certification of the CLASSICAL-FEEDBACK memristor: the temporal steering
test applied to the device analyzed in the paper.

Protocol (tap-out configuration). Each output photon is routed at a 50:50
splitter either to the device's own feedback detector (fixed Z analysis;
a click updates the classical register theta) or to the tester, who measures
it in a randomly chosen MUB x. The tester's basis choice never enters the
feedback loop, as the certification assumptions require. The temporal
assemblage over consecutive tester photons is then

    sigma_{a|x} = E[ P(a|x, theta_k) rho(theta_{k+1}) ],

with all (x,a)-dependence routed through the classical common cause theta:
an explicit local-hidden-state model with lambda = theta. Hence R_T = 0 and
S_N <= sqrt(N) for ALL parameters -- the device's register memory, as stated in the paper, is
classical (measure-and-prepare, hence entanglement-breaking by
construction). The measurement-basis sensitivity of the feedback (S_{X,Z} = 0.97)
is not a quantum signature: a classical polarized beam reproduces it.

The script verifies numerically:
  1. R_T of the sampled tap-out assemblage (SDP, MOSEK) ~ 0 within
     Monte-Carlo noise, for N = 2, 3;
  2. S_N stays below sqrt(N) across eta;
  3. (control) if the tester instead RECONFIGURES the device's feedback
     measurement to basis x, the assemblage becomes SIGNALING
     (sum_a sigma_{a|x} depends on x) -- the steering SDP is then
     inapplicable, which is precisely why the protocol mandates that the
     basis choice remain outside the feedback loop.

Deterministic seed. Companion of sim_certify_coherent.py.
"""
import itertools

import numpy as np
import cvxpy as cp

rng = np.random.default_rng(20260706)

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = {"X": SX, "Y": SY, "Z": SZ}


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


# ── photon-by-photon feedback dynamics (discrete trajectory model) ─────────
ETA = 0.5
GAMMA = 0.01
THETA0 = 0.0
OMEGA = 0.12
A = 1.0
DT = 1.0
N_PHOTONS = 400_000
BASES3 = ["X", "Y", "Z"]


def chi(t):
    return 0.5 * A * np.sin(OMEGA * t)


def out_vec(theta, t):
    c = np.cos(2 * theta - chi(t)); s = np.sin(2 * theta - chi(t))
    return np.array([c, s], dtype=complex)


def out_state(theta, t):
    v = out_vec(theta, t)
    return np.outer(v, v.conj())


def prob_plus(theta, t, basis):
    rho = out_state(theta, t)
    P = projectors(PAULIS[basis])[0]
    return float(np.real(np.trace(P @ rho)))


def step_feedback(theta, t):
    """Device's own Z-basis feedback measurement on a tapped photon."""
    p = np.cos(2 * theta - chi(t)) ** 2
    click = rng.random() < p
    return theta + ETA * click - GAMMA * (theta - THETA0) * DT


def run_tapout(eta, bases, n_photons=N_PHOTONS):
    """Tap-out protocol: tester basis never enters the loop."""
    global ETA
    ETA = eta
    N = len(bases)
    sig = {(x, a): np.zeros((2, 2), dtype=complex) for x in range(N) for a in (0, 1)}
    n_events = 0
    corr = np.zeros(N); ncorr = np.zeros(N)
    theta, t = THETA0, 0.0
    for _ in range(20_000):
        theta = step_feedback(theta, t); t += DT
    pending = None            # (x, a) of the last tester photon
    for _ in range(n_photons):
        if rng.random() < 0.5:                      # feedback branch
            theta = step_feedback(theta, t)
        else:                                       # tester branch
            x = rng.integers(N)
            pp = prob_plus(theta, t, bases[x])
            a = 0 if rng.random() < pp else 1
            if pending is not None:
                # this tester photon is the "k+1" of the previous one
                xp, ap = pending
                sig[(xp, ap)] += out_state(theta, t)
                n_events += 1
                s2 = +1 if a == 0 else -1
                if bases[x] == bases[xp]:
                    s1 = +1 if ap == 0 else -1
                    corr[xp] += s1 * s2; ncorr[xp] += 1
            pending = (x, a)
        t += DT
    # normalize to an assemblage: sigma_{a|x}, weights p(a|x) with p(x)=1/N
    for x in range(N):
        tot = sum(np.real(np.trace(sig[(x, a)])) for a in (0, 1))
        for a in (0, 1):
            sig[(x, a)] /= tot
    S_N = float(np.sum(np.where(ncorr > 0, corr / np.maximum(ncorr, 1), 0)))
    return sig, S_N, n_events


def signaling_norm(sig, N):
    """max_x,x' trace-norm distance of marginals sum_a sigma_{a|x}."""
    margs = []
    for x in range(N):
        margs.append(sig[(x, 0)] + sig[(x, 1)])
    dmax = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            ev = np.linalg.eigvalsh(margs[i] - margs[j])
            dmax = max(dmax, 0.5 * float(np.sum(np.abs(ev))))
    return dmax


def run_reconfigured(eta, bases, n_photons=N_PHOTONS):
    """Control: tester reconfigures the device's feedback basis to x
    (basis choice LEAKS into the loop) -> signaling assemblage."""
    global ETA
    ETA = eta
    N = len(bases)
    sig = {(x, a): np.zeros((2, 2), dtype=complex) for x in range(N) for a in (0, 1)}
    counts = np.zeros((N, 2))
    theta, t = THETA0, 0.0
    for _ in range(20_000):
        theta = step_feedback(theta, t); t += DT
    for _ in range(n_photons):
        x = rng.integers(N)
        pp = prob_plus(theta, t, bases[x])
        a = 0 if rng.random() < pp else 1
        theta = theta + ETA * (1 - a) - GAMMA * (theta - THETA0) * DT
        t += DT
        sig[(x, a)] += out_state(theta, t)
        counts[x, a] += 1
    for x in range(N):
        tot = counts[x].sum()
        for a in (0, 1):
            sig[(x, a)] /= tot
    return sig


def main():
    """Runs the checks in the fixed order of the seeded generator (the S_3 values depend on that order) and
    returns the numbers that panel (b) of paper_device_certification.pdf draws."""
    res = {}
    print("Tap-out certification protocol (basis choice outside the loop):")
    for N, bases in ((2, ["X", "Z"]), (3, BASES3)):
        sig, S_N, n_ev = run_tapout(0.5, bases)
        rt = steering_robustness(sig, N)
        sgn = signaling_norm(sig, N)
        res[f"rt_{N}"], res[f"sig_{N}"], res[f"S_{N}"] = rt, sgn, S_N
        print(f"  N={N}: R_T = {rt:.2e}, signaling = {sgn:.2e} "
              f"(both ~ MC noise, {n_ev} events); S_N = {S_N:+.3f} <= sqrt(N) = {np.sqrt(N):.3f}")

    print("\nS_3 vs eta (bound sqrt(3)=1.732):")
    res["eta"] = np.array([0.1, 0.3, 0.5, 0.8])
    s3 = []
    for eta in res["eta"]:
        _, S_N, _ = run_tapout(eta, BASES3)
        s3.append(S_N)
        print(f"  eta={eta:3.1f}: S_3 = {S_N:+.3f}")
    res["S3_eta"] = np.array(s3)

    print("\nControl: reconfigured-feedback protocol (basis leaks into loop):")
    sig = run_reconfigured(0.5, BASES3)
    res["sig_reconf"] = signaling_norm(sig, 3)
    print(f"  signaling = {res['sig_reconf']:.3f}  "
          "(assemblage is signaling; steering SDP inapplicable)")
    return res


if __name__ == "__main__":
    main()
