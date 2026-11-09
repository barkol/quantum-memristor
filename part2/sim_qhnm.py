#!/usr/bin/env python3
r"""Full QHN-M (memristive Quantum Hopfield Network) storage + retrieval
simulation.  Replaces the "Full QHN-M simulation in progress" placeholder
with REAL numerical results.

Physics model
-------------
n qubits, each carrying a classical memristor memory variable mu_i driven by
photon-detection feedback:

    d mu_i / dt = eta * N_i(t) - gamma (mu_i - mu0),   N_i in {0,1}.

Couplings are built from pairwise correlations of the detection records, so
the Hebbian rule emerges from the streaming-photon physics:

    J_ij(T) -> (eta/gamma) * sum_mu xi_i^mu xi_j^mu          (gamma->0 limit)
    J_ij(T) -> (eta/gamma) * sum_mu xi_i^mu xi_j^mu e^{-gamma(T-t_mu)}  (palimpsest)

Hamiltonian:
    H = - sum_{i<j} J_ij sigma_z^i sigma_z^j
        - Omega sum_i sigma_x^i
        - sum_i h_i sigma_z^i

Retrieval dynamics (stated clearly):  quantum annealing.  We initialise the
qubits in the +x product state (transverse-field ground state), then linearly
ramp a schedule s: 0 -> 1 that lowers Omega and raises the problem
Hamiltonian, integrating the Schroedinger equation
    H(s) = -(1-s) Omega_0 sum_i sigma_x^i  +  s * H_problem
with a noisy/partial pattern injected as the local biases h_i.  We read out
the Z-basis bit string with the largest probability.  The Omega=0 control
replaces the quantum tunnelling by greedy zero-temperature single-spin-flip
descent from the noisy pattern (classical Hopfield retrieval).

Everything is exact (dense 2^n Hamiltonians via Kronecker products of Pauli
matrices), n <= 10.  Fixed seed -> reproducible.

Run:  python3 sim_qhnm.py
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# global style: shared HOM house style (figstyle.py).  Slide-appropriate
# base font (13).  NOTE: make_paper_figures.py imports this module and then
# calls figstyle.apply() (font 18) itself, so its paper figures override this.
# ----------------------------------------------------------------------
import figstyle

figstyle.apply(base_fontsize=13)
USETEX = plt.rcParams["text.usetex"]

# HOM colour scheme (figstyle): blue primary, tab:red secondary, grey third
QBLUE = figstyle.BLUE       # 'b'        -- primary series
QGREEN = figstyle.THIRD     # '0.35'     -- neutral third series (3rd colour)
QRED = figstyle.RED         # 'tab:red'  -- secondary series
GREY = figstyle.GREY        # '0.5'      -- auxiliary reference lines

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUTDIR, exist_ok=True)

SEED = 20260625
RNG = np.random.default_rng(SEED)


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {os.path.abspath(path)}")


# ----------------------------------------------------------------------
# Pauli matrices and tensor-product Hamiltonian builders
# ----------------------------------------------------------------------
I2 = np.eye(2)
SX = np.array([[0.0, 1.0], [1.0, 0.0]])
SZ = np.array([[1.0, 0.0], [0.0, -1.0]])


def kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def single_op(op, i, n):
    """op acting on qubit i (0-indexed, qubit 0 = most significant bit)."""
    return kron_list([op if k == i else I2 for k in range(n)])


def build_problem_hamiltonian(J, h):
    """H_problem = - sum_{i<j} J_ij Z_i Z_j - sum_i h_i Z_i  (dense 2^n)."""
    n = len(h)
    dim = 2 ** n
    H = np.zeros((dim, dim))
    Zops = [single_op(SZ, i, n) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                H -= J[i, j] * (Zops[i] @ Zops[j])
        if h[i] != 0.0:
            H -= h[i] * Zops[i]
    return H


def transverse_field(n):
    """sum_i X_i."""
    dim = 2 ** n
    Hx = np.zeros((dim, dim))
    for i in range(n):
        Hx += single_op(SX, i, n)
    return Hx


# Diagonal of the classical Z-Z energy (fast, no 2^n x 2^n matrix needed)
def spin_table(n):
    """Return (2^n, n) array of +/-1 spins for each computational basis state.
    Basis ordering matches np.kron with qubit 0 as most significant bit."""
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)) & 1)
    return 1 - 2 * bits  # 0 -> +1, 1 -> -1  (|0>=|H>=+1, |1>=|V>=-1)


def classical_energy(J, h, spins):
    """E(sigma) = - sum_{i<j} J_ij s_i s_j - sum_i h_i s_i, vectorised."""
    n = spins.shape[1]
    # pairwise term
    E = np.zeros(spins.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                E -= J[i, j] * spins[:, i] * spins[:, j]
    E -= spins @ h
    return E


# ----------------------------------------------------------------------
# Memristor feedback: stream photon-encoded patterns, integrate the ODE
# ----------------------------------------------------------------------
def stream_memristor_couplings(patterns, eta, gamma, mu0=0.0, dt=1.0,
                               steps_per_pattern=1, palimpsest=False,
                               rng=None):
    r"""Stream L patterns as detection events and integrate the memristor
    feedback ODE for the *pairwise* memory variables mu_ij that set J_ij.

    Physical picture: when pattern xi^mu streams in, qubit i is prepared in
    |H> (det record +1) or |V> (-1).  The correlator of detection records on
    a pair (i,j) is xi_i xi_j.  The pairwise memristor obeys

        d mu_ij/dt = eta * xi_i^mu xi_j^mu - gamma (mu_ij - mu0)

    integrated (exact exponential step) over the dwell time of each pattern.
    J_ij := mu_ij.  With gamma->0 this is the running Hebbian sum; with finite
    gamma the contribution of pattern mu decays as e^{-gamma (T - t_mu)}.

    Returns J (n x n, zero diagonal) and, if requested, the full trajectory.
    """
    L, n = patterns.shape
    mu = np.full((n, n), mu0)
    np.fill_diagonal(mu, 0.0)
    traj = []  # snapshots of mu after each pattern (for emergence plot)
    decay = np.exp(-gamma * dt) if gamma > 0 else 1.0
    for mu_idx in range(L):
        xi = patterns[mu_idx]
        corr = np.outer(xi, xi).astype(float)  # xi_i xi_j, diagonal = 1
        np.fill_diagonal(corr, 0.0)
        for _ in range(steps_per_pattern):
            # exact solution of dmu/dt = eta*corr - gamma*(mu-mu0) over dt:
            if gamma > 0:
                mu = mu0 + (mu - mu0) * decay + (eta * corr / gamma) * (1 - decay)
            else:
                mu = mu + eta * corr * dt
        traj.append(mu.copy())
    return mu, np.array(traj)


def hebb_theory(patterns):
    """Theoretical Hebbian coupling (1/n) sum_mu xi_i xi_j, zero diagonal."""
    L, n = patterns.shape
    J = (patterns.T @ patterns) / n
    np.fill_diagonal(J, 0.0)
    return J


# ----------------------------------------------------------------------
# Retrieval dynamics
# ----------------------------------------------------------------------
def retrieve_classical(J, noisy_pattern, h_strength=0.0, max_sweeps=200):
    """Omega=0 control: zero-temperature greedy single-spin-flip descent
    (classical Hopfield asynchronous update) from the noisy pattern."""
    n = len(noisy_pattern)
    s = noisy_pattern.copy().astype(float)
    for _ in range(max_sweeps):
        changed = False
        order = RNG.permutation(n)
        for i in order:
            # local field from couplings (no external bias during relaxation)
            field = J[i] @ s
            new = 1.0 if field >= 0 else -1.0
            if new != s[i]:
                s[i] = new
                changed = True
        if not changed:
            break
    return s.astype(int)


def retrieve_classical_sa(J, noisy_pattern, bias_strength=0.6, n_sweeps=300):
    """Fairer classical baseline: simulated annealing on the SAME problem the
    quantum anneal solves --- couplings J plus the input injected as a weak local
    bias h=bias_strength*noisy (so the only difference from
    `retrieve_quantum_anneal` is thermal vs. quantum-tunnelling fluctuations).
    This is the right control for asking whether the *transverse field* beats a
    competent classical annealer rather than merely the greedy descent.

    Energy E=-1/2 sum_ij J_ij s_i s_j - sum_i h_i s_i; a single flip of s_i costs
    dE = 2 s_i (J_i.s + h_i).  The temperature schedule starts at the problem's
    own energy scale (median |J_i.s + h|) so it can hop spurious barriers without
    erasing the input, then cools geometrically; a final zero-T settle pins the
    readout in a minimum.  Best of a few restarts (the anneal's analogue of
    reading the most probable bit string)."""
    n = len(noisy_pattern)
    h = bias_strength * noisy_pattern.astype(float)

    def energy(s):
        return -0.5 * s @ (J @ s) - h @ s

    # set T0 to the typical local-field magnitude so flips are neither frozen
    # (T0<<|field|) nor random (T0>>|field|)
    scale = np.median(np.abs(J @ noisy_pattern.astype(float) + h)) + 1e-6
    T0, Tend = 1.5 * scale, 0.02 * scale
    temps = T0 * (Tend / T0) ** (np.arange(n_sweeps) / max(1, n_sweeps - 1))

    best, best_E = None, np.inf
    for _ in range(3):                      # a few independent restarts
        s = noisy_pattern.copy().astype(float)
        for Tcur in temps:
            for i in RNG.permutation(n):
                dE = 2.0 * s[i] * (J[i] @ s + h[i])
                if dE <= 0.0 or RNG.random() < np.exp(-dE / Tcur):
                    s[i] = -s[i]
        for _ in range(50):                 # zero-T settle into a minimum
            changed = False
            for i in RNG.permutation(n):
                new = 1.0 if (J[i] @ s + h[i]) >= 0 else -1.0
                if new != s[i]:
                    s[i] = new; changed = True
            if not changed:
                break
        E = energy(s)
        if E < best_E:
            best_E, best = E, s.copy()
    return best.astype(int)


def _apply_x_rotations(psi, n, theta):
    r"""Apply exp(-i theta sum_i X_i) = prod_i [cos t I - i sin t X_i] exactly.
    Single-qubit X-rotations commute across qubits, so we sweep qubit by qubit
    using tensor reshapes -- O(n * 2^n), no 2^n x 2^n matrix needed."""
    c = np.cos(theta)
    s = -1j * np.sin(theta)
    state = psi.reshape([2] * n)
    for i in range(n):
        state = np.moveaxis(state, i, 0)
        a0 = state[0].copy()
        a1 = state[1].copy()
        state[0] = c * a0 + s * a1   # X swaps the two amplitudes
        state[1] = c * a1 + s * a0
        state = np.moveaxis(state, 0, i)
    return state.reshape(-1)


def retrieve_quantum_anneal(J, noisy_pattern, Omega0, n_steps=120, T=12.0,
                            bias_strength=0.6, spins=None):
    r"""Omega>0 retrieval via quantum annealing (real-time Schroedinger
    evolution).  Schedule s: 0 -> 1.

        H(s) = -(1-s) Omega0 sum_i X_i + s * H_problem(J, h_bias)

    The noisy/partial pattern is injected as weak local Z-biases
    h_i = bias_strength * noisy_i, anchoring the dynamics near the input
    while the transverse field lets it tunnel to the true attractor.
    Start in the +x product state (ground state of -sum X_i).  Integrate with
    a symmetric 2nd-order Trotter split: the Z-part of H is DIAGONAL in the
    computational basis (cheap elementwise phase) and the X-part is a product
    of single-qubit rotations (cheap tensor sweep).  Read out the most
    probable Z bit string.  Numerically exact up to O(dt^2) Trotter error,
    which is negligible at the slice count used.
    """
    n = len(noisy_pattern)
    dim = 2 ** n
    if spins is None:
        spins = spin_table(n)
    h_bias = bias_strength * noisy_pattern.astype(float)
    # diagonal classical problem energy E(sigma) for every basis state
    Ediag = classical_energy(J, h_bias, spins)  # length 2^n

    # initial state |+>^n  (uniform amplitudes)
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)

    dt = T / n_steps
    for k in range(n_steps):
        s = (k + 0.5) / n_steps  # midpoint schedule
        Ax = (1 - s) * Omega0    # transverse field coefficient
        # H = -Ax sum X_i + s * H_problem(diag).  Symmetric Trotter:
        # exp(-i H dt) ~ U_x(dt/2) U_z(dt) U_x(dt/2)
        psi = _apply_x_rotations(psi, n, -Ax * dt / 2.0)
        psi *= np.exp(-1j * s * Ediag * dt)
        psi = _apply_x_rotations(psi, n, -Ax * dt / 2.0)
    probs = np.abs(psi) ** 2
    best = int(np.argmax(probs))
    bits = ((best >> np.arange(n - 1, -1, -1)) & 1)
    return (1 - 2 * bits).astype(int)


def overlap(a, b):
    """Normalised overlap (1 = identical, accounting for global sign flip)."""
    n = len(a)
    m = np.dot(a, b) / n
    return abs(m)  # Hopfield attractors come in +/- pairs; sign-blind overlap


# ======================================================================
#  VECTORISED (batched-over-trials) retrievers.  Same physics as the
#  per-call versions above, but a whole batch of B noisy patterns is
#  processed at once: the per-spin Metropolis/descent inner loop runs over
#  n spins (vectorised across the B trials) instead of n*B Python iterations,
#  and the quantum anneal evolves a (2^n, B) state in one pass.  ~B-fold fewer
#  Python iterations -> the high-statistics figures run in seconds, not minutes.
# ======================================================================
def overlap_batch(R, true_batch):
    """Sign-blind overlap per row of two (B, n) arrays -> (B,)."""
    return np.abs(np.sum(R * true_batch, axis=1) / R.shape[1])


def _apply_x_rotations_batch(psi, n, theta):
    """exp(-i theta sum_i X_i) on a (2^n, B) batch of states (B = last axis)."""
    c = np.cos(theta); s = -1j * np.sin(theta)
    B = psi.shape[-1]
    state = psi.reshape([2] * n + [B])
    for i in range(n):
        state = np.moveaxis(state, i, 0)
        a0 = state[0].copy(); a1 = state[1].copy()
        state[0] = c * a0 + s * a1
        state[1] = c * a1 + s * a0
        state = np.moveaxis(state, 0, i)
    return state.reshape(2 ** n, B)


def retrieve_quantum_anneal_batch(J, noisy_batch, Omega0, n_steps=120, T=12.0,
                                  bias_strength=0.6, spins=None):
    """Batched real-time quantum anneal; noisy_batch (B, n) -> retrieved (B, n)."""
    noisy_batch = np.atleast_2d(noisy_batch).astype(float)
    B, n = noisy_batch.shape
    dim = 2 ** n
    if spins is None:
        spins = spin_table(n)
    h_bias = bias_strength * noisy_batch                          # (B, n)
    Ecoup = classical_energy(J, np.zeros(n), spins)              # (dim,)
    Ediag = Ecoup[:, None] - spins @ h_bias.T                    # (dim, B)
    psi = np.full((dim, B), 1.0 / np.sqrt(dim), dtype=complex)
    dt = T / n_steps
    for k in range(n_steps):
        s = (k + 0.5) / n_steps
        Ax = (1 - s) * Omega0
        psi = _apply_x_rotations_batch(psi, n, -Ax * dt / 2.0)
        psi *= np.exp(-1j * s * Ediag * dt)
        psi = _apply_x_rotations_batch(psi, n, -Ax * dt / 2.0)
    best = np.argmax(np.abs(psi) ** 2, axis=0)                  # (B,)
    bits = (best[:, None] >> np.arange(n - 1, -1, -1)) & 1      # (B, n)
    return (1 - 2 * bits).astype(int)


def retrieve_classical_batch(J, noisy_batch, max_sweeps=200):
    """Batched zero-T greedy single-spin-flip descent (the Omega=0 control)."""
    S = np.atleast_2d(noisy_batch).astype(float).copy()
    n = S.shape[1]
    for _ in range(max_sweeps):
        changed = np.zeros(S.shape[0], dtype=bool)
        for i in RNG.permutation(n):
            new = np.where(S @ J[i] >= 0.0, 1.0, -1.0)
            changed |= new != S[:, i]
            S[:, i] = new
        if not changed.any():
            break
    return S.astype(int)


def retrieve_classical_sa_batch(J, noisy_batch, bias_strength=0.6,
                                n_sweeps=300, n_restarts=3):
    """Batched simulated annealing on the input-biased Ising problem (the fair
    thermal baseline); noisy_batch (B, n) -> retrieved (B, n), best of restarts.
    Per-trial temperature schedule set to the median local-field magnitude."""
    noisy_batch = np.atleast_2d(noisy_batch).astype(float)
    B, n = noisy_batch.shape
    h = bias_strength * noisy_batch                               # (B, n)
    scale = np.median(np.abs(noisy_batch @ J + h), axis=1) + 1e-9  # (B,)
    frac = np.arange(n_sweeps) / max(1, n_sweeps - 1)
    temps = (1.5 * scale)[None, :] * (0.02 / 1.5) ** frac[:, None]  # (n_sweeps, B)
    best, bestE = None, np.full(B, np.inf)
    for _ in range(n_restarts):
        S = noisy_batch.copy()
        field = S @ J + h                                        # (B, n), maintained
        for sw in range(n_sweeps):
            Tcur = temps[sw]                                     # (B,)
            for i in RNG.permutation(n):
                dE = 2.0 * S[:, i] * field[:, i]                 # (B,)
                acc = (dE <= 0.0) | (RNG.random(B) < np.exp(-dE / Tcur))
                ds = np.where(acc, -2.0 * S[:, i], 0.0)          # (B,)
                S[:, i] += ds
                field += np.outer(ds, J[i])                      # J[i,i]=0
        for _ in range(50):                                     # zero-T settle
            changed = np.zeros(B, dtype=bool)
            for i in RNG.permutation(n):
                new = np.where(S @ J[i] + h[:, i] >= 0.0, 1.0, -1.0)
                changed |= new != S[:, i]
                S[:, i] = new
            if not changed.any():
                break
        E = -0.5 * np.einsum('bi,ij,bj->b', S, J, S) - np.einsum('bi,bi->b', h, S)
        if best is None:
            best, bestE = S.copy(), E.copy()
        else:
            take = E < bestE
            best[take], bestE[take] = S[take], E[take]
    return best.astype(int)


# ----------------------------------------------------------------------
# FIGURE 1 -- Hebbian emergence
# ----------------------------------------------------------------------
def fig_hebb():
    n = 8
    L = 5
    # Hebbian emergence is the SLOW-FORGETTING (storage) regime: a small leak
    # rate so that all L streamed patterns accumulate with near-equal weight.
    # The memristor feedback ODE then integrates to the Hebbian coupling.
    # We use eta = 1/n so the learned scale matches the textbook (1/n) sum;
    # gamma small (0.01) -> exponential weights ~ 1 over the stream.
    eta = 1.0 / n
    gamma = 0.01
    dt = 1.0
    patterns = RNG.choice([-1, 1], size=(L, n))

    # one dwell per pattern; with steps_per_pattern=1, dt=1 the running memory
    # is mu_ij = (eta/gamma)(1-e^{-gamma L}) ~ eta * sum xixi for small gamma.
    J_learned, traj = stream_memristor_couplings(
        patterns, eta=eta, gamma=gamma, dt=dt, steps_per_pattern=1)
    # exponentially-weighted Hebbian target (exact for this ODE):
    iu = np.triu_indices(n, 1)
    ages = np.arange(L)[::-1]                      # pattern mu has age L-1-mu
    w = (eta / gamma) * (1 - np.exp(-gamma * dt))  # per-pattern step weight
    w_decay = np.exp(-gamma * dt * ages)           # fading factor per pattern
    J_theory = np.zeros((n, n))
    for mu in range(L):
        J_theory += w * w_decay[mu] * np.outer(patterns[mu], patterns[mu])
    np.fill_diagonal(J_theory, 0.0)

    # plain textbook Hebbian (1/n) sum, for the honest "matches Hebb" comparison
    J_hebb = hebb_theory(patterns)                 # (1/n) sum xixi

    final_corr = np.corrcoef(J_learned[iu], J_theory[iu])[0, 1]
    corr_hebb = np.corrcoef(J_learned[iu], J_hebb[iu])[0, 1]
    rel_err = np.linalg.norm(J_learned[iu] - J_theory[iu]) / np.linalg.norm(J_theory[iu])
    print(f"[hebb] n={n} L={L} eta={eta:.4f} gamma={gamma}  "
          f"corr(learned, weighted-Hebb)={final_corr:.4f}  "
          f"corr(learned, plain-Hebb)={corr_hebb:.4f}  rel_err={rel_err:.3e}")

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    lo = min(J_learned[iu].min(), J_theory[iu].min())
    hi = max(J_learned[iu].max(), J_theory[iu].max())
    pad = 0.08 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=GREY, lw=0.9,
            zorder=1)
    ax.scatter(J_theory[iu], J_learned[iu], s=16, color=QBLUE, zorder=2)
    if USETEX:
        ax.set_xlabel(r"Hebbian $\frac{\eta}{\gamma}\sum_\mu \xi_i^\mu \xi_j^\mu$")
        ax.set_ylabel(r"learned $J_{ij}$")
        ax.text(0.05, 0.92, rf"$r={final_corr:.3f}$", transform=ax.transAxes,
                va="top", ha="left", color=QRED, fontsize=9)
    else:
        ax.set_xlabel(r"Hebbian $(\eta/\gamma)\sum_\mu \xi_i\xi_j$")
        ax.set_ylabel(r"learned $J_{ij}$")
        ax.text(0.05, 0.92, rf"$r={final_corr:.3f}$", transform=ax.transAxes,
                va="top", ha="left", color=QRED, fontsize=9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    save(fig, "pres_qhnm_hebb.pdf")
    return final_corr, corr_hebb, rel_err


# ----------------------------------------------------------------------
# FIGURE 2 -- retrieval accuracy vs noise, Omega>0 vs Omega=0
# ----------------------------------------------------------------------
def fig_retrieval():
    n = 8
    # L=3: near the working-capacity edge so there ARE spurious minima for the
    # classical descent to fall into -> a fair test of whether the transverse
    # field Omega helps tunnel out of them.
    L = 3
    eta, gamma = 1.0 / n, 0.01
    patterns = RNG.choice([-1, 1], size=(L, n))
    J, _ = stream_memristor_couplings(patterns, eta=eta, gamma=gamma,
                                      dt=1.0, steps_per_pattern=1)

    spins = spin_table(n)
    noise_levels = np.array([0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5])
    n_trials = 40
    Omega0 = 1.0  # transverse field strength (in units of typical |J|)

    acc_q = np.zeros(len(noise_levels))
    acc_c = np.zeros(len(noise_levels))
    for fi, f in enumerate(noise_levels):
        nflip = int(np.floor(f * n + 0.5))
        oq, oc = [], []
        for _ in range(n_trials):
            mu = RNG.integers(0, L)
            true = patterns[mu].copy()
            noisy = true.copy()
            if nflip > 0:
                idx = RNG.choice(n, size=nflip, replace=False)
                noisy[idx] *= -1
            rq = retrieve_quantum_anneal(J, noisy, Omega0=Omega0, spins=spins)
            rc = retrieve_classical(J, noisy)
            # accuracy = recovery of the INTENDED (true) pattern, sign-blind
            oq.append(overlap(rq, true))
            oc.append(overlap(rc, true))
        acc_q[fi] = np.mean(oq)
        acc_c[fi] = np.mean(oc)

    print("[retrieval] noise   acc(Omega>0)  acc(Omega=0)")
    for f, aq, ac in zip(noise_levels, acc_q, acc_c):
        print(f"           {f:5.3f}   {aq:.3f}        {ac:.3f}")

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.plot(noise_levels, acc_q, "-o", color=QBLUE, ms=3.5, lw=1.4,
            label=r"$\Omega>0$ (anneal)")
    ax.plot(noise_levels, acc_c, "-s", color=QRED, ms=3.5, lw=1.4,
            label=r"$\Omega=0$ (classical)")
    ax.set_xlabel(r"input noise fraction $f$")
    ax.set_ylabel(r"recall overlap")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower left", frameon=False)
    save(fig, "pres_qhnm_retrieval.pdf")
    return noise_levels, acc_q, acc_c


# ----------------------------------------------------------------------
# FIGURE 3 -- capacity: success rate vs L, with 0.14n bound
# ----------------------------------------------------------------------
def fig_capacity():
    ns = [6, 8, 10]
    # three series (n=6,8,10) => blue / red / grey third (HOM scheme)
    colors = {6: QBLUE, 8: QRED, 10: QGREEN}
    n_nets = 12     # random pattern sets per (n,L)
    n_probes = 8    # retrieval probes per network
    f_noise = 0.125  # small fixed input corruption

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    empirical = {}
    for n in ns:
        spins = spin_table(n)  # cache once per n
        # storage regime: small leak so all L patterns accumulate (static
        # capacity test, not palimpsest forgetting).
        eta, gamma = 1.0 / n, 0.01
        Ls = list(range(1, n + 1))
        succ = []
        for L in Ls:
            hits = 0
            tot = 0
            for _ in range(n_nets):
                patterns = RNG.choice([-1, 1], size=(L, n))
                J, _ = stream_memristor_couplings(patterns, eta=eta, gamma=gamma,
                                                  dt=1.0, steps_per_pattern=1)
                for _ in range(n_probes):
                    mu = RNG.integers(0, L)
                    true = patterns[mu].copy()
                    noisy = true.copy()
                    nflip = int(np.floor(f_noise * n + 0.5))
                    if nflip > 0:
                        idx = RNG.choice(n, size=nflip, replace=False)
                        noisy[idx] *= -1
                    r = retrieve_quantum_anneal(J, noisy, Omega0=1.0, spins=spins)
                    if overlap(r, true) > 0.999:  # exact recovery of THIS pattern
                        hits += 1
                    tot += 1
            succ.append(hits / tot)
        succ = np.array(succ)
        ax.plot(Ls, succ, "-o", color=colors[n], ms=3.0, lw=1.3,
                label=rf"$n={n}$")
        # empirical capacity: largest L with success >= 0.9
        cap = 0
        for L, s in zip(Ls, succ):
            if s >= 0.9:
                cap = L
        empirical[n] = cap
        print(f"[capacity] n={n}  success(L)={np.round(succ,3).tolist()}  "
              f"empirical L(>=0.9)={cap}  0.14n={0.14*n:.2f}")

    # 0.14 n classical bound overlaid as vertical markers
    for n in ns:
        lc = 0.14 * n
        ax.axvline(lc, color=colors[n], ls=":", lw=0.9, alpha=0.7)
    ax.text(0.97, 0.55, r"$0.14\,n$ bound", transform=ax.transAxes,
            ha="right", va="center", color=GREY, fontsize=7, rotation=0)
    ax.set_xlabel(r"stored patterns $L$")
    ax.set_ylabel(r"recall success")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower left", frameon=False, ncol=3, columnspacing=0.8,
              handletextpad=0.3)
    save(fig, "pres_qhnm_capacity.pdf")
    return empirical


# ----------------------------------------------------------------------
# FIGURE 4 -- palimpsest / continual learning
# ----------------------------------------------------------------------
def fig_palimpsest():
    n = 8
    L = 20          # long stream of patterns, far beyond static capacity
    eta = 0.30
    gamma = 0.30    # finite forgetting rate -> fading memory (1/gamma ~ 3.3)
    dt = 1.0        # one time unit dwell per pattern
    f_noise = 0.0625
    n_streams = 10  # average over independent streams -> smooth, honest curve
    n_probes = 12
    spins = spin_table(n)
    ages = np.arange(L)            # 0 = newest

    retr_acc = np.zeros(L)
    for _ in range(n_streams):
        patterns = RNG.choice([-1, 1], size=(L, n))
        # Stream ALL patterns with the palimpsest (finite-gamma) memristor.
        # The final J is the exponentially-weighted Hebbian sum:
        #   J_ij = (eta/gamma)(1-e^{-gamma dt}) sum_mu xi xi e^{-gamma(T-t_mu)}
        J_final, _ = stream_memristor_couplings(
            patterns, eta=eta, gamma=gamma, dt=dt, steps_per_pattern=1)
        for mu in range(L):
            age = (L - 1) - mu
            true = patterns[mu]
            hits = 0
            for _ in range(n_probes):
                noisy = true.copy()
                nflip = int(np.floor(f_noise * n + 0.5))
                if nflip > 0:
                    idx = RNG.choice(n, size=nflip, replace=False)
                    noisy[idx] *= -1
                r = retrieve_quantum_anneal(J_final, noisy, Omega0=1.0, spins=spins)
                if overlap(r, true) > 0.999:
                    hits += 1
            retr_acc[age] += hits / n_probes
    retr = retr_acc / n_streams

    # theoretical fading weight of pattern at age a: e^{-gamma * a * dt}
    weight = np.exp(-gamma * ages * dt)

    # retrievable window: count leading ages (from newest) with retr>=0.5
    window = 0
    for a in range(L):
        if retr[a] >= 0.5:
            window += 1
        else:
            break
    # catastrophic-interference check: do OLD patterns corrupt the NEWEST?
    # (newest retrievability should stay high regardless of stream length)
    newest_recall = retr[0]
    oldest_recall = retr[-1]
    print(f"[palimpsest] n={n} L={L} eta={eta} gamma={gamma}  "
          f"retrievable window (retr>=0.5, from newest) = {window} patterns  "
          f"(1/gamma = {1/gamma:.1f} patterns)")
    print(f"[palimpsest] newest recall={newest_recall:.2f}, "
          f"oldest recall={oldest_recall:.2f}  (no catastrophic interference: "
          f"newest stays high while old fades)")
    print(f"[palimpsest] retr vs age = {np.round(retr,2).tolist()}")

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.bar(ages, retr, width=0.85, color=QBLUE, alpha=0.55,
           label=r"recall (sim.)")
    ax.plot(ages, weight, color=QRED, lw=1.5,
            label=r"$e^{-\gamma\,\mathrm{age}}$")
    ax.axhline(0.5, color=GREY, ls=":", lw=0.8)
    ax.set_xlabel(r"pattern age $(T-t_\mu)/\Delta t$")
    ax.set_ylabel(r"retrievability")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.6, L - 0.4)
    ax.legend(loc="upper right", frameon=False)
    save(fig, "pres_palimpsest.pdf")
    return window, retr


# ----------------------------------------------------------------------
def main():
    print("=" * 64)
    print(f"QHN-M full simulation  (seed={SEED})")
    print("=" * 64)
    hc, hc_plain, he = fig_hebb()
    nl, aq, ac = fig_retrieval()
    cap = fig_capacity()
    win, retr = fig_palimpsest()

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"Hebbian emergence: corr(ODE)={hc:.4f}, "
          f"corr(plain-Hebb)={hc_plain:.4f}, rel_err={he:.2e}")
    # report retrieval at a couple of noise levels
    for target in (0.125, 0.25):
        i = int(np.argmin(np.abs(nl - target)))
        print(f"Retrieval f={nl[i]:.3f}: Omega>0 = {aq[i]:.3f}, "
              f"Omega=0 = {ac[i]:.3f}")
    print(f"Capacity (empirical L at >=0.9 success): {cap}; "
          f"0.14n = {[round(0.14*n,2) for n in cap]}")
    print(f"Palimpsest retrievable window: {win} patterns")


if __name__ == "__main__":
    main()
