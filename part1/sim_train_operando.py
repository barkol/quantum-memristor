r"""Certification IN OPERANDO: temporal steering of a consecutive output pair
during continuous operation of the coherent delay-loop memristor, without a
dedicated pair run.

Model. In continuous operation the two joint gate applications between every
pair of consecutive train photons compose to a controlled phase Phi (as in
the pair run); at the certification program Phi = pi the ideal output train
is a 1D cluster chain,

    |C_n> = prod_{m} CZ_{m,m+1}(pi) |+>^{(x) n}.

The tester measures every output photon anyway; for the certified pair
(k, k+1) the outcomes of the NEIGHBORING photons, measured in the X basis,
localize the entanglement into the pair (standard one-way-computing
entanglement localization): conditioned on each neighbor branch, the pair is
maximally entangled up to known Pauli byproducts, which the tester undoes in
software (allowed: frame corrections conditioned on its own records).
The correction is implemented below as the fidelity-optimal two-qubit Pauli
frame; this coincides with the predetermined cluster-state byproduct rule in
every branch, at and below the quoted noise levels (the ideal branches reach
unit fidelity, and isotropic depolarization commutes with Pauli frames), so
no data-dependent optimization enters the protocol.

The script verifies:
  1. ideal train (n = 6, interior pair): every neighbor branch has pair
     concurrence 1; after Pauli correction the branch-averaged assemblage
     equals the dedicated pair-run assemblage -> R_T identical (0.268 / 0.172);
  2. noisy train: every photon independently depolarized with visibility
     v_q = V_g^{n_g} eta_loop^{n_p} (same per-photon noise model as the
     budget); R_T of the corrected in-operando assemblage versus per-pass
     loss -> the in-operando loss threshold, to compare with the pair-run
     budget of Eq. (loss_budget).

Deterministic. Output: printed thresholds (quoted in the manuscript).
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
PAULI2 = [I2, SX, SY, SZ]


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


# ── cluster-chain construction ─────────────────────────────────────────────
N_QUBITS = 6
PAIR = (2, 3)                      # 0-indexed interior pair (photons k, k+1)
NEIGHBORS = [q for q in range(N_QUBITS) if q not in PAIR]


def op_on(op, q, n=N_QUBITS):
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, op if i == q else I2)
    return out


def cz_pair(q1, q2, phi, n=N_QUBITS):
    P1_a = op_on((I2 + SZ) / 2, q1, n); P1_b = op_on((I2 - SZ) / 2, q1, n)
    P2_a = op_on((I2 + SZ) / 2, q2, n); P2_b = op_on((I2 - SZ) / 2, q2, n)
    return (P1_a @ P2_a + P1_a @ P2_b + P1_b @ P2_a
            + np.exp(1j * phi) * P1_b @ P2_b)


def cluster_state(n=N_QUBITS):
    plus = np.ones(2, dtype=complex) / np.sqrt(2)
    psi = np.array([1.0 + 0j])
    for _ in range(n):
        psi = np.kron(psi, plus)
    for m in range(n - 1):
        psi = cz_pair(m, m + 1, np.pi) @ psi
    return psi


def depolarize_qubit(rho, q, v, n=N_QUBITS):
    """Isotropic depolarizing with retained visibility v on qubit q."""
    out = v * rho
    for P in (SX, SY, SZ):
        U = op_on(P, q, n)
        out += (1 - v) / 4 * (U @ rho @ U.conj().T)
    out += (1 - v) / 4 * rho
    return out


def x_branch_projector(outcomes, n=N_QUBITS):
    """Projector for X-basis outcomes on NEIGHBORS (list of 0/1)."""
    M = np.eye(2 ** n, dtype=complex)
    for q, o in zip(NEIGHBORS, outcomes):
        Pi = (I2 + (1 - 2 * o) * SX) / 2
        M = M @ op_on(Pi, q, n)
    return M


def reduce_to_pair(rho_full, n=N_QUBITS):
    """Trace out all qubits except PAIR (assumes projected neighbors)."""
    t = rho_full.reshape([2] * (2 * n))
    keep = list(PAIR)
    # trace out qubits not in keep, highest index first
    for q in sorted([q for q in range(n) if q not in keep], reverse=True):
        t = np.trace(t, axis1=q, axis2=q + t.ndim // 2)
        # after tracing, axes shift: rebuild index bookkeeping by recursion
        # simpler: handled by shrinking dims; adjust keep indices
        keep = [k - 1 if k > q else k for k in keep]
    d = int(np.sqrt(t.size))
    return t.reshape(d, d)


def pauli_correction(rho_pair, ref_rho):
    """Best two-qubit Pauli (P (x) Q) aligning rho_pair with the reference."""
    best, best_f = None, -1.0
    for P in PAULI2:
        for Q in PAULI2:
            U = np.kron(P, Q)
            cand = U @ rho_pair @ U.conj().T
            f = float(np.real(np.trace(cand @ ref_rho)))
            if f > best_f:
                best_f, best = f, cand
    return best, best_f


def assemblage_from_state(rho2, bases):
    sig = {}
    for x, b in enumerate(bases):
        for a, Pi in enumerate(projectors(MUBS[b])):
            M = np.kron(Pi, I2)
            R = (M @ rho2 @ M).reshape(2, 2, 2, 2)
            sig[(x, a)] = np.trace(R, axis1=0, axis2=2)
    return sig


def operando_pair_state(v_q):
    """Branch-averaged, Pauli-corrected pair state in continuous operation."""
    psi = cluster_state()
    rho = np.outer(psi, psi.conj())
    if v_q < 1.0:
        for q in range(N_QUBITS):
            rho = depolarize_qubit(rho, q, v_q)
    # reference: ideal pair-run state CZ(pi)|++>
    plus2 = np.ones(4, dtype=complex) / 2.0
    ref = np.diag([1, 1, 1, -1]).astype(complex) @ plus2
    ref_rho = np.outer(ref, ref.conj())
    avg = np.zeros((4, 4), dtype=complex)
    fids = []
    for outcomes in itertools.product((0, 1), repeat=len(NEIGHBORS)):
        M = x_branch_projector(outcomes)
        rb = M @ rho @ M.conj().T
        p = float(np.real(np.trace(rb)))
        if p < 1e-12:
            continue
        pair = reduce_to_pair(rb) / p
        corrected, f = pauli_correction(pair, ref_rho)
        avg += p * corrected
        fids.append(f)
    return avg / np.real(np.trace(avg)), (min(fids), max(fids))


# ── 1. ideal in-operando certification ─────────────────────────────────────
rho_pair, (fmin, fmax) = operando_pair_state(1.0)
print(f"ideal train n={N_QUBITS}: corrected branch fidelity to pair-run state: "
      f"min {fmin:.6f}, max {fmax:.6f}")
for N in (2, 3):
    rt = steering_robustness(assemblage_from_state(rho_pair, BASES[N]), N)
    print(f"  in-operando R_T (N={N}) = {rt:.4f}   (pair run: "
          f"{0.2679 if N==3 else 0.1716})")

# ── 2. noisy in-operando threshold ─────────────────────────────────────────
print("\nin-operando loss threshold (V_g=1, v_q = eta_loop^2 per photon):")
for N in (2, 3):
    lo, hi = 0.0, 2.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        v_q = (10 ** (-mid / 10.0)) ** 2
        rho_pair, _ = operando_pair_state(v_q)
        rt = steering_robustness(assemblage_from_state(rho_pair, BASES[N]), N)
        if rt > 1e-7:
            lo = mid
        else:
            hi = mid
    print(f"  N={N}: threshold {lo:.3f} dB/pass  (pair run: "
          f"{1.193 if N==3 else 0.753} dB/pass)")
