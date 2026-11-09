#!/usr/bin/env python3
r"""Memristive quantum reservoir computing (QRC) on the NARMA-10 task.

A *real* numerical simulation that replaces the toy 30-feature classical
reservoir used as a placeholder in the talk.  The reservoir is a small network
of photonic quantum memristors: each memristor is a polarization qubit passing
through a half-wave plate (HWP) at angle theta_i(t), where theta_i carries a
classical leaky memory driven by the input AND by measurement feedback.  Only a
linear (ridge) readout is trained; all nonlinearity and fading memory come from
the physics.

Physics model (from the talk)
-----------------------------
HWP unitary               U(theta) = [[cos2theta,  sin2theta],
                                       [sin2theta, -cos2theta]]
memory ODE (per qubit i)  dtheta_i/dt = eta * N_i(t) - gamma (theta_i - theta0)
detection rate            N_i(t) = Gamma_ph * Tr[Pi_+^i rho]   (back-action feedback)
input drive               theta_i also kicked by the scalar input u_t (encoding)
fading-memory kernel       K(tau) = eta * Gamma_ph * exp(-gamma tau)

Each timestep we (i) update every memristor angle with a leaky-integrator step
(this is the exp(-gamma tau) fading memory), (ii) apply per-qubit HWP unitaries,
(iii) apply a fixed entangling layer (CPHASE chain) so the 2^n Hilbert space is
genuinely explored, (iv) apply a weak partial reset / measurement back-action
implementing the detection-feedback nonlinearity, then (v) read out single- and
two-qubit Pauli expectation values as reservoir features.

Outputs three figures into ../figures/ (exact names the deck expects) and prints
the actual NMSE numbers.  Pure numpy/scipy/matplotlib; states built by hand as
numpy density matrices via Kronecker products (qutip is not installed).

Run: python3 sim_qrc_narma.py
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
_USETEX = plt.rcParams["text.usetex"]

# HOM colour scheme (figstyle): blue primary, tab:red secondary, grey third
QBLUE = figstyle.BLUE       # 'b'        -- primary series
QGREEN = figstyle.THIRD     # '0.35'     -- neutral third series (3rd colour)
QRED = figstyle.RED         # 'tab:red'  -- secondary series
GREY = figstyle.GREY        # '0.5'      -- auxiliary reference lines

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUTDIR, exist_ok=True)

SEED = 1234


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {path}")


# ======================================================================
#  single-qubit operators
# ======================================================================
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PI_PLUS = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0| projector Pi_+


def hwp(theta):
    """Half-wave-plate unitary at angle theta (the memristor element)."""
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    return np.array([[c, s], [s, -c]], dtype=complex)


def kron_list(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def embed(op, site, n):
    """Embed single-qubit operator `op` acting on `site` into n-qubit space."""
    ops = [I2] * n
    ops[site] = op
    return kron_list(ops)


def embed2(opa, sa, opb, sb, n):
    ops = [I2] * n
    ops[sa] = opa
    ops[sb] = opb
    return kron_list(ops)


# ======================================================================
#  NARMA-10 target time series
# ======================================================================
def narma10(T, rng):
    u = rng.uniform(0.0, 0.5, T)
    y = np.zeros(T)
    for t in range(9, T - 1):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * y[t - 9 : t + 1].sum()
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )
    return u, y


# ======================================================================
#  the memristive quantum reservoir
# ======================================================================
class MemristiveQRC:
    r"""Network of n photonic quantum memristors driven by a scalar input.

    State is the full 2^n x 2^n density matrix rho.  Each qubit i carries a
    classical memory angle theta_i obeying a leaky integrator (fading memory):

        theta_i <- theta_i + dt*( eta*N_i - gamma*(theta_i - theta0) ) + drive_i*u

    with N_i = Gamma_ph * Tr[Pi_+^i rho] the local detection rate (back-action
    feedback).  Per step: angle update -> per-qubit HWP -> fixed CPHASE chain
    -> weak partial reset (measurement back-action nonlinearity).
    """

    def __init__(self, n=6, gamma=0.06, eta=0.9, theta0=0.15,
                 Gamma_ph=1.0, dt=1.0, coupling=0.35, reset=0.03,
                 gain=0.35, gamma_spread=3.0, seed=SEED):
        self.n = n
        self.dim = 2 ** n
        self.gamma = gamma          # nominal forgetting rate (sweep knob)
        self.eta = eta
        self.theta0 = theta0
        self.Gamma_ph = Gamma_ph
        self.dt = dt
        self.reset = reset
        self.gain = gain            # memory -> HWP-angle gain
        rng = np.random.default_rng(seed)
        # fixed random per-qubit input encoding weights (drive theta_i by u_t)
        self.drive = 0.6 + 0.8 * rng.random(n)
        self.phase0 = rng.uniform(0, 2 * np.pi, n)  # static encoding offset
        # HETEROGENEOUS memory timescales: each memristor has its own
        # relaxation rate spread (log-uniform) around the nominal gamma, giving
        # the reservoir a bank of timescales -- essential to recall u_{t-9} in
        # NARMA-10.  The nominal gamma still scales the whole bank (sweep knob).
        self.gamma_i = gamma * np.exp(
            np.log(gamma_spread) * (rng.random(n) - 0.5) * 2.0
        )

        # --- precompute embedded observables (the linear readout features) -
        # store transposes so that Tr[A @ rho] = sum(A.T * rho) is O(dim^2).
        self.SZi = [embed(SZ, i, n) for i in range(n)]
        self.SXi = [embed(SX, i, n) for i in range(n)]
        self.PIi = [embed(PI_PLUS, i, n) for i in range(n)]
        self.SZZ = [
            (self.SZi[i] @ self.SZi[j])
            for i in range(n) for j in range(i + 1, n)
        ]
        self.n_pairs = len(self.SZZ)
        # transposed observables for fast trace via elementwise product
        self._obs_T = [op.T.copy() for op in
                       (self.SZi + self.SXi + self.SZZ)]
        self._piT = [op.T.copy() for op in self.PIi]

        # --- fixed entangling layer: CPHASE chain (nearest neighbour) ------
        self.U_couple = self._build_cphase_chain(coupling)

        self.reset_state = self._ground()  # |0..0><0..0| for partial reset

    def _ground(self):
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[0, 0] = 1.0
        return rho

    def _build_cphase_chain(self, coupling):
        """Fixed Ising-like entangler exp(-i*coupling*sum_<ij> Z_i Z_j)."""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.n - 1):
            H += embed2(SZ, i, SZ, i + 1, self.n)
        # ring closure to genuinely explore the space for small n
        if self.n > 2:
            H += embed2(SZ, self.n - 1, SZ, 0, self.n)
        from scipy.linalg import expm

        return expm(-1j * coupling * H)

    def init_memory(self):
        # leaky memory variable m_i (drives the HWP angle); starts at rest
        return np.zeros(self.n)

    def step(self, rho, m, u):
        """Advance reservoir one timestep; return (new_rho, new_m, feats).

        `m` is the per-qubit leaky memory variable.  Its exact-discretisation
        update is the fading-memory kernel exp(-gamma tau): a stable leaky
        integrator that accumulates the input AND the local detection rate
        (measurement back-action feedback), with timescale 1/gamma.
        """
        n = self.n
        # detection rates N_i = Gamma_ph * Tr[Pi_+^i rho]  (real, in [0,Gamma])
        # Tr[A @ rho] = sum(A.T * rho) -- O(dim^2) instead of a full matmul.
        N = self.Gamma_ph * np.array(
            [np.real(np.sum(self._piT[i] * rho)) for i in range(n)]
        )
        # exact leaky-integrator step (always stable):
        #   m_i <- a*m_i + (1-a)*( drive_i*u + eta*N_i ),   a = exp(-gamma*dt)
        # this is precisely the e^{-gamma tau} fading-memory kernel; large gamma
        # = fast forgetting (short memory), small gamma = long memory.
        a = np.exp(-self.gamma_i * self.dt)
        m = a * m + (1.0 - a) * (self.drive * u + self.eta * N)
        # map bounded memory -> HWP angle; static per-qubit offset keeps the
        # qubits inequivalent so the entangler explores the full 2^n space.
        theta_eff = self.theta0 + self.gain * np.pi * m + 0.15 * np.sin(self.phase0)

        # per-qubit HWP unitaries -> tensor product (separable layer)
        U_layer = hwp(theta_eff[0])
        for i in range(1, n):
            U_layer = np.kron(U_layer, hwp(theta_eff[i]))
        rho = U_layer @ rho @ U_layer.conj().T
        # fixed entangling layer
        rho = self.U_couple @ rho @ self.U_couple.conj().T
        # weak partial reset toward ground state: measurement back-action /
        # dissipative nonlinearity (keeps trajectory bounded, adds nonlinearity)
        if self.reset > 0:
            rho = (1 - self.reset) * rho + self.reset * self.reset_state
        rho = 0.5 * (rho + rho.conj().T)  # numerical hermiticity guard

        feats = self._features(rho)
        return rho, m, feats

    def _features(self, rho):
        # single-qubit <sigma_z>,<sigma_x> and two-qubit <sigma_z sigma_z>
        # via fast traces; append a bias term.
        vals = [np.real(np.sum(AT * rho)) for AT in self._obs_T]
        vals.append(1.0)
        return np.array(vals)

    @property
    def n_features(self):
        return 2 * self.n + self.n_pairs + 1

    def run(self, u):
        T = len(u)
        feats = np.zeros((T, self.n_features))
        rho = self._ground()
        m = self.init_memory()
        for t in range(T):
            rho, m, f = self.step(rho, m, u[t])
            feats[t] = f
        return feats


# ======================================================================
#  ridge readout + NMSE
# ======================================================================
def ridge_fit(X, y, alpha):
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def nmse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def evaluate(feats, y, train, test, alpha=1e-6):
    """Fit ridge on train split, return (nmse_test, y_pred_test, y_true_test)."""
    Xtr, ytr = feats[train], y[train]
    W = ridge_fit(Xtr, ytr, alpha)
    yp = feats[test] @ W
    return nmse(y[test], yp), yp, y[test]


# ======================================================================
#  baselines
# ======================================================================
def linear_floor_features(u, y, mem=10):
    """(a) raw input window only -- linear sanity floor, no reservoir."""
    T = len(u)
    F = np.zeros((T, mem + 1))
    for t in range(mem, T):
        F[t, :mem] = u[t - mem + 1 : t + 1]
        F[t, mem] = 1.0
    return F


def esn_features(u, n_feat, rng, spectral=0.9, leak=0.3, scale_in=0.5):
    """(b) classical echo-state network / random-tanh reservoir."""
    T = len(u)
    Win = rng.standard_normal((n_feat, 2)) * scale_in
    W = rng.standard_normal((n_feat, n_feat))
    # rescale to target spectral radius
    rho_W = max(abs(np.linalg.eigvals(W)))
    W *= spectral / rho_W
    x = np.zeros(n_feat)
    F = np.zeros((T, n_feat + 1))
    for t in range(T):
        pre = Win @ np.array([u[t], 1.0]) + W @ x
        x = (1 - leak) * x + leak * np.tanh(pre)
        F[t, :n_feat] = x
        F[t, n_feat] = 1.0
    return F


# ======================================================================
#  main experiment
# ======================================================================
def main():
    rng = np.random.default_rng(SEED)

    # ---- generate NARMA-10 -------------------------------------------
    T = 1200
    u, y = narma10(T, rng)
    target = y  # predict y_t from reservoir state at step t (y already shifted)

    washout = 200
    train = slice(washout, 900)
    test = slice(900, T)
    print(f"NARMA-10: T={T}, washout={washout}, "
          f"train={train.stop-train.start}, test={test.stop-test.start}")

    # ---- (c) quantum memristive reservoir ----------------------------
    N_QUBITS = 6
    GAMMA0 = 0.03   # nominal forgetting rate (scales the per-qubit timescale bank)
    qrc = MemristiveQRC(n=N_QUBITS, gamma=GAMMA0, eta=0.9, theta0=0.15,
                        coupling=0.35, reset=0.03, gain=0.35,
                        gamma_spread=3.0, seed=SEED)
    print(f"quantum reservoir: n={qrc.n} qubits, dim={qrc.dim}, "
          f"features={qrc.n_features}")
    qfeats = qrc.run(u)
    nmse_q, yp_q, yt_q = evaluate(qfeats, target, train, test)

    # ---- (a) linear floor --------------------------------------------
    lfeats = linear_floor_features(u, y)
    nmse_lin, _, _ = evaluate(lfeats, target, train, test)

    # ---- (b) classical ESN with comparable feature count -------------
    esn_n = qrc.n_features - 1  # match feature count (minus shared bias)
    efeats = esn_features(u, esn_n, np.random.default_rng(SEED + 1))
    nmse_esn, _, _ = evaluate(efeats, target, train, test)

    print(f"\n=== NMSE on NARMA-10 test set ===")
    print(f"(a) linear floor (raw input window) : {nmse_lin:.4f}")
    print(f"(b) classical ESN ({esn_n} nodes)    : {nmse_esn:.4f}")
    print(f"(c) quantum memristive reservoir     : {nmse_q:.4f}")

    # ---- gamma sweep (fading-memory / forgetting-rate knob) ----------
    # gamma scales the whole per-qubit timescale bank: too small -> memory
    # never fades (saturates, no fresh nonlinearity); too large -> forgets the
    # 10-step NARMA history.  Expect an interior optimum (edge-of-chaos).
    gammas = np.geomspace(0.01, 1.2, 16)
    nmse_g = []
    for g in gammas:
        q = MemristiveQRC(n=N_QUBITS, gamma=g, eta=0.9, theta0=0.15,
                          coupling=0.35, reset=0.03, gain=0.35,
                          gamma_spread=3.0, seed=SEED)
        f = q.run(u)
        nm, _, _ = evaluate(f, target, train, test)
        nmse_g.append(nm)
        print(f"  gamma={g:6.4f}  NMSE={nm:.4f}")
    nmse_g = np.array(nmse_g)
    g_opt = gammas[int(np.argmin(nmse_g))]
    nmse_g_opt = float(nmse_g.min())
    print(f"\noptimal gamma = {g_opt:.3f}  ->  NMSE = {nmse_g_opt:.4f}")

    # ==================================================================
    #  FIGURE 1: test-set prediction
    # ==================================================================
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    tt = np.arange(len(yt_q))
    ax.plot(tt, yt_q, color=QBLUE, lw=1.4, label=r"target $y_{t}$")
    ax.plot(tt, yp_q, color=QRED, lw=1.4, ls="--",
            label=r"QRC $\hat y_{t}$")
    ax.set_xlabel(r"time step $t$")
    ax.set_ylabel(r"NARMA-10")
    ax.set_title(rf"quantum reservoir: NMSE $= {nmse_q:.3f}$")
    ax.legend(loc="upper right", ncol=2)
    save(fig, "pres_narma_pred.pdf")

    # ==================================================================
    #  FIGURE 2: NMSE comparison bars
    # ==================================================================
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    labels = [r"linear" + "\n" + r"floor",
              r"classical" + "\n" + r"ESN",
              r"quantum" + "\n" + r"memristor"]
    vals = [nmse_lin, nmse_esn, nmse_q]
    # three series: grey third / red / blue (HOM scheme)
    colors = [QGREEN, QRED, QBLUE]
    bars = ax.bar(range(3), vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, rf"{v:.3f}",
                ha="center", va="bottom")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"test NMSE")
    ax.set_ylim(0, max(vals) * 1.25)
    save(fig, "pres_qrc_nmse_bars.pdf")

    # ==================================================================
    #  FIGURE 3: gamma sweep (fading-memory optimum)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.semilogx(gammas, nmse_g, "o-", color=QBLUE, lw=1.6, ms=4)
    ax.plot([g_opt], [nmse_g_opt], "*", color=QRED, ms=13,
            label=rf"optimum $\gamma={g_opt:.2g}$")
    ax.set_xlabel(r"forgetting rate $\gamma$")
    ax.set_ylabel(r"test NMSE")
    ax.set_title(r"fading memory: NMSE vs $\gamma$")
    ax.legend(loc="upper left")
    save(fig, "pres_qrc_gamma_sweep.pdf")

    # ---- machine-readable summary ------------------------------------
    print("\n=== SUMMARY ===")
    print(f"seed={SEED} n_qubits={N_QUBITS} dim={qrc.dim} "
          f"features={qrc.n_features}")
    print(f"train={train.stop-train.start} test={test.stop-test.start} "
          f"washout={washout}")
    print(f"nmse_linear={nmse_lin:.4f} nmse_esn={nmse_esn:.4f} "
          f"nmse_quantum={nmse_q:.4f}")
    print(f"gamma_opt={g_opt:.3f} nmse_at_gamma_opt={nmse_g_opt:.4f}")


if __name__ == "__main__":
    main()
