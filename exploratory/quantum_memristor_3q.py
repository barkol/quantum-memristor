"""
Simulation of the fully-quantum photonic memristor with 3-qubit gate and delay loops.

Architecture:
    At each time step k, a 3-qubit programmable gate U acts on:
      - Port 1 (target):  fresh photon k
      - Port 2 (program): photon k-1 (one loop delay)
      - Port 3 (control): photon k-2 (two loop delays)
    After the gate, photon k-2 exits port 3 and is measured.
    The remaining 2-qubit state (photons k, k-1) is the quantum memory.

    Each photon passes through the gate exactly 3 times:
      step k:   as target  (interacts with k-1, k-2)
      step k+1: as program (interacts with k+1, k-1)
      step k+2: as control (interacts with k+2, k+1) -> then measured

Key outputs:
    1. Measurement record and effective memristive response
    2. Hysteresis loops under sinusoidal input modulation
    3. Concurrence (entanglement) of the 2-qubit loop state
    4. CHSH Bell correlations between consecutive exiting photons
    5. Short-term memory capacity

Author: K. Bartkiewicz, P. Tulewicz, A. Miranowicz (AMU Poznan)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt

# ============================================================
# Qubit and gate primitives
# ============================================================

I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)
KET_H = KET_0  # horizontal polarisation
KET_V = KET_1  # vertical polarisation


def ket_to_dm(psi):
    """Pure state vector -> density matrix."""
    psi = np.asarray(psi, dtype=complex).reshape(-1, 1)
    return psi @ psi.conj().T


def kron(*args):
    """Kronecker product of multiple arrays."""
    result = args[0]
    for m in args[1:]:
        result = np.kron(result, m)
    return result


def Ry(theta):
    """Single-qubit Y-rotation by angle theta."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(phi):
    """Single-qubit Z-rotation by angle phi."""
    return np.array([[np.exp(-1j * phi / 2), 0],
                     [0, np.exp(1j * phi / 2)]], dtype=complex)


def partial_trace_qubit(rho_3q, qubit_to_trace):
    """
    Partial trace of a 3-qubit density matrix over one qubit.

    Parameters
    ----------
    rho_3q : (8,8) array — 3-qubit density matrix
    qubit_to_trace : int — 0 (target), 1 (program), or 2 (control)

    Returns
    -------
    (4,4) array — 2-qubit density matrix of the remaining qubits
    """
    rho = rho_3q.reshape(2, 2, 2, 2, 2, 2)
    # axes: (t_row, p_row, c_row, t_col, p_col, c_col)
    if qubit_to_trace == 0:
        return np.trace(rho, axis1=0, axis2=3).reshape(4, 4)
    elif qubit_to_trace == 1:
        return np.trace(rho, axis1=1, axis2=4).reshape(4, 4)
    elif qubit_to_trace == 2:
        return np.trace(rho, axis1=2, axis2=5).reshape(4, 4)
    else:
        raise ValueError("qubit_to_trace must be 0, 1, or 2")


def measure_qubit(rho_3q, qubit, outcome, basis='Z'):
    """
    Projective measurement on one qubit of a 3-qubit state.

    Parameters
    ----------
    rho_3q : (8,8) array
    qubit : int — which qubit (0, 1, 2)
    outcome : int — 0 or 1
    basis : str — 'X', 'Y', or 'Z'

    Returns
    -------
    rho_post : (4,4) array — post-measurement 2-qubit state (unnormalised)
    prob : float — probability of this outcome
    """
    # Build projector in the measurement basis
    if basis == 'Z':
        proj = ket_to_dm(KET_0) if outcome == 0 else ket_to_dm(KET_1)
    elif basis == 'X':
        ket_plus = (KET_0 + KET_1) / np.sqrt(2)
        ket_minus = (KET_0 - KET_1) / np.sqrt(2)
        proj = ket_to_dm(ket_plus) if outcome == 0 else ket_to_dm(ket_minus)
    elif basis == 'Y':
        ket_py = (KET_0 + 1j * KET_1) / np.sqrt(2)
        ket_my = (KET_0 - 1j * KET_1) / np.sqrt(2)
        proj = ket_to_dm(ket_py) if outcome == 0 else ket_to_dm(ket_my)
    else:
        raise ValueError(f"Unknown basis: {basis}")

    # Build full 3-qubit projector
    ops = [I2, I2, I2]
    ops[qubit] = proj
    P = kron(*ops)

    # Project and trace
    rho_projected = P @ rho_3q @ P
    prob = np.real(np.trace(rho_projected))

    if prob < 1e-15:
        # Zero probability outcome
        rho_post = np.zeros((4, 4), dtype=complex)
        return rho_post, 0.0

    rho_post = partial_trace_qubit(rho_projected / prob, qubit)
    return rho_post, prob


def concurrence_2q(rho):
    """Concurrence of a 2-qubit density matrix (Wootters formula)."""
    sy = kron(Y, Y)
    rho_tilde = sy @ rho.conj() @ sy
    R = rho @ rho_tilde
    eigvals = np.sort(np.real(np.sqrt(np.maximum(np.linalg.eigvals(R), 0))))[::-1]
    return max(0.0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3])


# ============================================================
# 3-qubit gate constructors
# ============================================================

def make_gate_layered(phi_tp, phi_pc, phi_tc,
                      theta_t=0.0, theta_p=0.0, theta_c=0.0,
                      phi_t=0.0, phi_p=0.0, phi_c=0.0):
    """
    Programmable 3-qubit gate built from layers:
      1. Single-qubit Ry rotations
      2. CPhase between target-program
      3. CPhase between program-control
      4. Single-qubit Rz rotations
      5. CPhase between target-control

    Parameters
    ----------
    phi_tp, phi_pc, phi_tc : float
        Controlled-phase angles between qubit pairs.
    theta_t, theta_p, theta_c : float
        Ry rotation angles for each qubit.
    phi_t, phi_p, phi_c : float
        Rz rotation angles for each qubit.
    """
    dim = 8

    # Layer 1: Ry rotations
    R1 = kron(Ry(theta_t), Ry(theta_p), Ry(theta_c))

    # CPhase(phi) on qubits i,j: |11⟩ -> e^{iφ}|11⟩
    def cphase_3q(phi, q1, q2):
        U = np.eye(dim, dtype=complex)
        for idx in range(dim):
            bits = [(idx >> (2 - q)) & 1 for q in range(3)]
            if bits[q1] == 1 and bits[q2] == 1:
                U[idx, idx] = np.exp(1j * phi)
        return U

    CP_tp = cphase_3q(phi_tp, 0, 1)
    CP_pc = cphase_3q(phi_pc, 1, 2)
    CP_tc = cphase_3q(phi_tc, 0, 2)

    # Layer 2: Rz rotations
    R2 = kron(Rz(phi_t), Rz(phi_p), Rz(phi_c))

    return CP_tc @ R2 @ CP_pc @ CP_tp @ R1


def make_gate_ising_plus_rotations(J_tp, J_pc, J_tc,
                                    theta_t, theta_p, theta_c):
    """
    Gate from Ising Hamiltonian sandwiched between Ry rotations:
    U = Ry_layer · exp(-i H_Ising) · Ry_layer

    H_Ising = J_tp Z_t Z_p + J_pc Z_p Z_c + J_tc Z_t Z_c
    """
    # Ising diagonal
    H_diag = np.zeros(8, dtype=complex)
    for t, p, c in [(i, j, k) for i in range(2)
                     for j in range(2) for k in range(2)]:
        zt, zp, zc = 1 - 2*t, 1 - 2*p, 1 - 2*c
        idx = t*4 + p*2 + c
        H_diag[idx] = J_tp*zt*zp + J_pc*zp*zc + J_tc*zt*zc
    U_ising = np.diag(np.exp(-1j * H_diag))

    Ry_layer = kron(Ry(theta_t), Ry(theta_p), Ry(theta_c))
    return Ry_layer @ U_ising @ Ry_layer


# ============================================================
# Main simulator
# ============================================================

@dataclass
class StepResult:
    """Result of one simulation step."""
    time_step: int
    outcome: int           # measurement outcome (0 or 1)
    outcome_sign: float    # +1 or -1
    prob: float            # probability of the obtained outcome
    basis: str             # measurement basis used
    concurrence: float     # entanglement of the 2-qubit loop state
    input_angle: float     # Bloch angle of input state


class QuantumMemristorLoop:
    """
    Simulator for the 3-qubit gate quantum memristor with delay loops.

    The simulation tracks a 2-qubit density matrix (the loop state)
    and at each step:
      1. Tensor in a fresh input qubit
      2. Apply the 3-qubit gate
      3. Measure the control qubit (oldest photon)
      4. Update the loop state

    Parameters
    ----------
    gate : (8,8) ndarray
        The 3-qubit unitary gate.
    loop_loss : float
        Per-loop amplitude damping probability (0 = lossless).
    """

    def __init__(self, gate, loop_loss=0.0):
        self.gate = np.asarray(gate, dtype=complex)
        assert self.gate.shape == (8, 8), "Gate must be 8x8"
        self.loop_loss = loop_loss

        # Initialise loop state: both qubits in |0⟩
        self.rho_loop = kron(ket_to_dm(KET_0), ket_to_dm(KET_0))
        self.results = []
        self._step_count = 0

    def reset(self):
        """Reset to initial state."""
        self.rho_loop = kron(ket_to_dm(KET_0), ket_to_dm(KET_0))
        self.results = []
        self._step_count = 0

    def _apply_loss(self, rho_2q):
        """Apply amplitude damping to each loop qubit (simple model)."""
        if self.loop_loss <= 0:
            return rho_2q
        p = self.loop_loss
        # Kraus operators for amplitude damping on each qubit
        K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
        rho = np.zeros_like(rho_2q)
        for Ka in [K0, K1]:
            for Kb in [K0, K1]:
                K = kron(Ka, Kb)
                rho += K @ rho_2q @ K.conj().T
        return rho

    def step(self, input_state=None, input_angle=0.0, basis='Z'):
        """
        Execute one time step.

        Parameters
        ----------
        input_state : (2,) array or None
            Pure state of the fresh input photon. If None, uses |0⟩.
        input_angle : float
            Bloch-sphere polar angle (for record-keeping and hysteresis).
        basis : str
            Measurement basis for the exiting photon ('X', 'Y', 'Z').

        Returns
        -------
        StepResult
        """
        self._step_count += 1

        if input_state is None:
            input_state = KET_0

        # Apply loop losses
        self.rho_loop = self._apply_loss(self.rho_loop)

        # 1. Form 3-qubit state: |fresh⟩ ⊗ ρ_loop
        #    Qubit ordering: 0=target(fresh), 1=program(1-loop), 2=control(2-loop)
        rho_input = ket_to_dm(input_state)
        rho_3q = kron(rho_input, self.rho_loop)

        # 2. Apply gate
        rho_3q = self.gate @ rho_3q @ self.gate.conj().T

        # 3. Measure qubit 2 (control = oldest photon exiting)
        #    Sample outcome
        _, p0 = measure_qubit(rho_3q, qubit=2, outcome=0, basis=basis)
        _, p1 = measure_qubit(rho_3q, qubit=2, outcome=1, basis=basis)

        # Normalise probabilities
        p_total = p0 + p1
        p0, p1 = p0 / p_total, p1 / p_total

        outcome = 0 if np.random.random() < p0 else 1
        prob = p0 if outcome == 0 else p1

        # 4. Post-measurement state of qubits 0,1 (the new loop state)
        rho_post, _ = measure_qubit(rho_3q, qubit=2, outcome=outcome, basis=basis)
        self.rho_loop = rho_post

        # 5. Compute entanglement
        conc = concurrence_2q(self.rho_loop)

        result = StepResult(
            time_step=self._step_count,
            outcome=outcome,
            outcome_sign=1.0 - 2.0 * outcome,
            prob=prob,
            basis=basis,
            concurrence=conc,
            input_angle=input_angle,
        )
        self.results.append(result)
        return result

    def run(self, input_states, input_angles=None, bases=None):
        """
        Run the simulation for a sequence of input states.

        Parameters
        ----------
        input_states : list of (2,) arrays
        input_angles : list of floats (optional, for hysteresis)
        bases : list of str or single str

        Returns
        -------
        list of StepResult
        """
        N = len(input_states)
        if input_angles is None:
            input_angles = [0.0] * N
        if bases is None:
            bases = ['Z'] * N
        elif isinstance(bases, str):
            bases = [bases] * N

        results = []
        for k in range(N):
            r = self.step(input_states[k], input_angles[k], bases[k])
            results.append(r)
        return results


# ============================================================
# Input state generators
# ============================================================

def sinusoidal_input_states(N, omega, amplitude=np.pi/4, offset=np.pi/4):
    """
    Generate input states with sinusoidally modulated Bloch angle.
    θ(k) = offset + amplitude · sin(ω·k)
    |ψ(k)⟩ = cos(θ/2)|H⟩ + sin(θ/2)|V⟩
    """
    angles = offset + amplitude * np.sin(omega * np.arange(N))
    states = [np.array([np.cos(a/2), np.sin(a/2)], dtype=complex) for a in angles]
    return states, angles


def fixed_input_states(N, theta=np.pi/4):
    """All photons prepared in the same state on the Bloch sphere."""
    state = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=complex)
    return [state] * N, [theta] * N


def random_input_states(N):
    """Uniformly random input states on the Bloch sphere."""
    thetas = np.random.uniform(0, np.pi, N)
    phis = np.random.uniform(0, 2*np.pi, N)
    states = [np.array([np.cos(t/2), np.sin(t/2) * np.exp(1j*p)], dtype=complex)
              for t, p in zip(thetas, phis)]
    return states, thetas


# ============================================================
# Analysis functions
# ============================================================

def compute_hysteresis(results, window=1):
    """
    Compute the memristive response (running average of outcomes)
    and the input drive for hysteresis plotting.
    """
    outcomes = np.array([r.outcome_sign for r in results])
    angles = np.array([r.input_angle for r in results])

    # Running average as "effective θ"
    if window > 1:
        kernel = np.ones(window) / window
        response = np.convolve(outcomes, kernel, mode='same')
    else:
        response = np.cumsum(outcomes) / (np.arange(len(outcomes)) + 1)

    return angles, response


def compute_chsh(sim, gate, input_states, input_angles, n_runs=5000):
    """
    Estimate CHSH value from consecutive measurement outcomes.

    For each pair of basis settings (a, b), runs the full simulation
    and computes E(a,b) = ⟨outcome_k · outcome_{k+1}⟩.

    Uses 4 setting pairs: (a,b), (a,b'), (a',b), (a',b').
    a = Z, a' = X, b = diagonal (π/8), b' = anti-diagonal (3π/8).
    """
    # Measurement bases
    settings = {
        'a':  'Z',
        'a_prime': 'X',
    }
    # For b, b' we use rotated Z bases (diagonal/antidiagonal)
    # Since our measure_qubit supports X, Y, Z,
    # we use Z and X as the two settings for the exiting photon.
    # For a proper CHSH we need 4 combinations.
    # Strategy: alternate basis choices for consecutive photons.

    def run_with_alternating_bases(basis_even, basis_odd, n_steps):
        """Run simulation, measuring even-index exits in basis_even,
        odd-index exits in basis_odd."""
        sim_run = QuantumMemristorLoop(gate)
        bases = [basis_even if k % 2 == 0 else basis_odd
                 for k in range(n_steps)]
        sim_run.run(input_states[:n_steps], input_angles[:n_steps], bases)
        return sim_run.results

    n_steps = min(len(input_states), n_runs)

    # Compute E(a, b) for each setting pair
    correlators = {}
    for (name_a, basis_a), (name_b, basis_b) in [
        (('a', 'Z'), ('b', 'X')),
        (('a', 'Z'), ("b'", 'Y')),
        (("a'", 'X'), ('b', 'X')),
        (("a'", 'X'), ("b'", 'Y')),
    ]:
        results = run_with_alternating_bases(basis_a, basis_b, n_steps)
        # Correlations between consecutive outcomes
        signs = [r.outcome_sign for r in results]
        products = [signs[k] * signs[k+1] for k in range(len(signs) - 1)]
        E = np.mean(products)
        correlators[f"E({name_a},{name_b})"] = E

    E_ab = correlators["E(a,b)"]
    E_ab_prime = correlators["E(a,b')"]
    E_a_prime_b = correlators["E(a',b)"]
    E_a_prime_b_prime = correlators["E(a',b')"]

    chsh = abs(E_ab + E_ab_prime + E_a_prime_b - E_a_prime_b_prime)
    return chsh, correlators


def compute_memory_capacity(gate, n_steps=2000, max_delay=15, n_trials=20):
    """
    Short-term memory task (cf. Kobayashi et al. PRX Quantum 2024).

    Feed random inputs, try to recover the input from d steps ago
    from the measurement record.

    Returns R²_d for each delay d and total capacity C.
    """
    R2_values = []

    for trial in range(n_trials):
        # Random binary inputs
        input_bits = np.random.randint(0, 2, n_steps)
        input_states = [KET_H if b == 0 else KET_V for b in input_bits]
        input_angles = [0.0 if b == 0 else np.pi for b in input_bits]

        sim = QuantumMemristorLoop(gate)
        results = sim.run(input_states, input_angles, bases='Z')
        outcomes = np.array([r.outcome for r in results])

        trial_R2 = []
        for d in range(max_delay + 1):
            if d >= n_steps:
                trial_R2.append(0.0)
                continue
            # Target: input from d steps ago
            target = input_bits[d:n_steps]
            prediction = outcomes[d:n_steps]
            # Truncate to same length
            n = min(len(target), len(prediction))
            target = target[:n].astype(float)
            prediction = prediction[:n].astype(float)

            # Linear regression (single feature -> single weight + bias)
            if np.std(prediction) < 1e-10 or np.std(target) < 1e-10:
                trial_R2.append(0.0)
                continue

            # Use simple correlation as R²
            corr = np.corrcoef(target, prediction)[0, 1]
            trial_R2.append(corr**2)

        R2_values.append(trial_R2)

    R2_mean = np.mean(R2_values, axis=0)
    capacity = np.sum(R2_mean)
    return R2_mean, capacity


# ============================================================
# Plotting
# ============================================================

def plot_simulation_results(results, title="Quantum Memristor (3-qubit gate)"):
    """Comprehensive plot of simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    steps = [r.time_step for r in results]
    outcomes = [r.outcome_sign for r in results]
    concurrences = [r.concurrence for r in results]
    angles = [r.input_angle for r in results]

    # 1. Measurement outcomes over time
    ax = axes[0, 0]
    ax.plot(steps, outcomes, '.', markersize=2, alpha=0.5, color='C0')
    # Running average
    window = min(50, len(outcomes) // 5)
    if window > 1:
        kernel = np.ones(window) / window
        avg = np.convolve(outcomes, kernel, mode='same')
        ax.plot(steps, avg, '-', color='C3', linewidth=1.5,
                label=f'running avg (w={window})')
    ax.set_xlabel('time step k')
    ax.set_ylabel('outcome ±1')
    ax.set_title('Measurement outcomes')
    ax.legend()
    ax.set_ylim(-1.3, 1.3)

    # 2. Entanglement (concurrence) over time
    ax = axes[0, 1]
    ax.plot(steps, concurrences, '-', linewidth=0.8, color='C1')
    ax.set_xlabel('time step k')
    ax.set_ylabel('concurrence')
    ax.set_title('Entanglement of loop state')
    ax.set_ylim(-0.05, 1.05)

    # 3. Hysteresis loop
    ax = axes[1, 0]
    drive, response = compute_hysteresis(results, window=max(1, len(results)//20))
    n_warmup = len(results) // 4
    ax.plot(drive[n_warmup:], response[n_warmup:], '-', linewidth=0.5,
            alpha=0.7, color='C2')
    ax.set_xlabel('input angle θ_in')
    ax.set_ylabel('memristive response')
    ax.set_title('Hysteresis loop')

    # 4. Input angle over time
    ax = axes[1, 1]
    ax.plot(steps, angles, '-', linewidth=0.8, color='C4')
    ax.set_xlabel('time step k')
    ax.set_ylabel('input angle θ_in')
    ax.set_title('Input modulation')

    plt.tight_layout()
    return fig


def plot_memory_capacity(R2_values, capacity, title="Short-term memory"):
    """Plot memory capacity vs delay."""
    fig, ax = plt.subplots(figsize=(8, 5))
    delays = np.arange(len(R2_values))
    ax.bar(delays, R2_values, color='C0', alpha=0.7)
    ax.set_xlabel('delay d')
    ax.set_ylabel('R² (determination coefficient)')
    ax.set_title(f'{title} — total capacity C = {capacity:.2f}')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig


def plot_basis_comparison(gate, N=600, omega=0.15):
    """
    Run with the same input in X, Y, Z measurement bases
    and compare hysteresis loops (memristive basis dependence).
    """
    states, angles = sinusoidal_input_states(N, omega)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Basis-dependent hysteresis (3-qubit quantum memristor)', fontsize=13)

    for idx, basis in enumerate(['Z', 'X', 'Y']):
        sim = QuantumMemristorLoop(gate)
        results = sim.run(states, list(angles), bases=basis)
        drive, response = compute_hysteresis(results,
                                             window=max(1, N // 20))
        n_warmup = N // 4
        axes[idx].plot(drive[n_warmup:], response[n_warmup:],
                       '-', linewidth=0.5, alpha=0.8)
        axes[idx].set_xlabel('input angle')
        axes[idx].set_ylabel('response')
        axes[idx].set_title(f'{basis}-basis measurement')

        # Compute effective loop area
        area = np.abs(np.trapz(response[n_warmup:],
                               drive[n_warmup:]))
        axes[idx].text(0.05, 0.95, f'area ~= {area:.3f}',
                       transform=axes[idx].transAxes,
                       va='top', fontsize=10)

    plt.tight_layout()
    return fig


# ============================================================
# Main demonstration
# ============================================================

def main():
    print("=" * 60)
    print("3-Qubit Quantum Memristor Simulation")
    print("=" * 60)

    # --- Gate parameters ---
    # These produce a strongly entangling gate
    gate = make_gate_layered(
        phi_tp=np.pi/3,     # target-program coupling
        phi_pc=np.pi/4,     # program-control coupling
        phi_tc=np.pi/5,     # target-control coupling
        theta_t=np.pi/6,    # Ry rotation on target
        theta_p=np.pi/4,    # Ry rotation on program
        theta_c=np.pi/8,    # Ry rotation on control
    )

    # Verify unitarity
    assert np.allclose(gate @ gate.conj().T, np.eye(8), atol=1e-12), \
        "Gate is not unitary!"
    print("[OK] Gate is unitary")

    # --- 1. Basic run with sinusoidal input ---
    print("\n[1] Running sinusoidal input simulation...")
    N = 800
    omega = 2 * np.pi / 80  # period = 80 steps
    states, angles = sinusoidal_input_states(N, omega)
    sim = QuantumMemristorLoop(gate)
    results = sim.run(states, list(angles), bases='Z')

    avg_concurrence = np.mean([r.concurrence for r in results[20:]])
    print(f"    Mean concurrence: {avg_concurrence:.4f}")
    print(f"    Outcome variance: {np.var([r.outcome_sign for r in results]):.4f}")

    fig1 = plot_simulation_results(results,
        title="3-Qubit Quantum Memristor — Sinusoidal Input")
    fig1.savefig('../figures/3q_memristor_sinusoidal.pdf', bbox_inches='tight')
    print("    Saved: 3q_memristor_sinusoidal.pdf")

    # --- 2. Basis-dependent hysteresis ---
    print("\n[2] Computing basis-dependent hysteresis...")
    fig2 = plot_basis_comparison(gate, N=800, omega=omega)
    fig2.savefig('../figures/3q_memristor_basis_hysteresis.pdf', bbox_inches='tight')
    print("    Saved: 3q_memristor_basis_hysteresis.pdf")

    # --- 3. Entanglement vs gate strength ---
    print("\n[3] Scanning entanglement vs coupling strength...")
    phi_values = np.linspace(0, np.pi, 30)
    avg_conc = []
    for phi in phi_values:
        g = make_gate_layered(phi_tp=phi, phi_pc=phi*0.8, phi_tc=phi*0.6,
                              theta_t=np.pi/6, theta_p=np.pi/4, theta_c=np.pi/8)
        sim = QuantumMemristorLoop(g)
        states_short, angles_short = sinusoidal_input_states(200, omega)
        res = sim.run(states_short, list(angles_short), bases='Z')
        avg_conc.append(np.mean([r.concurrence for r in res[20:]]))

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(phi_values / np.pi, avg_conc, 'o-', color='C1')
    ax3.set_xlabel('coupling strength φ_tp / π')
    ax3.set_ylabel('mean concurrence')
    ax3.set_title('Entanglement vs gate coupling strength')
    fig3.savefig('../figures/3q_memristor_entanglement_scan.pdf', bbox_inches='tight')
    print("    Saved: 3q_memristor_entanglement_scan.pdf")

    # --- 4. CHSH Bell test ---
    print("\n[4] Computing CHSH correlations...")
    states_bell, angles_bell = fixed_input_states(2000, theta=np.pi/4)
    chsh_val, correlators = compute_chsh(
        sim, gate, states_bell, angles_bell, n_runs=2000)
    print(f"    CHSH value: {chsh_val:.4f}  (classical bound = 2.0)")
    for name, val in correlators.items():
        print(f"      {name} = {val:.4f}")
    if chsh_val > 2.0:
        print("    -> CHSH VIOLATION detected!")
    else:
        print("    -> No CHSH violation (may need different gate/input)")

    # --- 5. Memory capacity ---
    print("\n[5] Computing short-term memory capacity...")
    R2, capacity = compute_memory_capacity(gate, n_steps=1500,
                                            max_delay=15, n_trials=30)
    print(f"    Total capacity: C = {capacity:.2f}")
    fig5 = plot_memory_capacity(R2, capacity,
        title="Short-term memory — 3-qubit quantum memristor")
    fig5.savefig('../figures/3q_memristor_memory_capacity.pdf', bbox_inches='tight')
    print("    Saved: 3q_memristor_memory_capacity.pdf")

    # --- 6. Effect of loop loss ---
    print("\n[6] Scanning effect of loop loss...")
    loss_values = np.linspace(0, 0.3, 15)
    conc_vs_loss = []
    cap_vs_loss = []
    for loss in loss_values:
        g = make_gate_layered(phi_tp=np.pi/3, phi_pc=np.pi/4, phi_tc=np.pi/5,
                              theta_t=np.pi/6, theta_p=np.pi/4, theta_c=np.pi/8)
        sim_loss = QuantumMemristorLoop(g, loop_loss=loss)
        states_l, angles_l = sinusoidal_input_states(300, omega)
        res_l = sim_loss.run(states_l, list(angles_l), bases='Z')
        conc_vs_loss.append(np.mean([r.concurrence for r in res_l[20:]]))

        R2_l, cap_l = compute_memory_capacity(g, n_steps=500,
                                               max_delay=10, n_trials=10)
        cap_vs_loss.append(cap_l)

    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(12, 5))
    ax6a.plot(loss_values, conc_vs_loss, 'o-', color='C1')
    ax6a.set_xlabel('loop loss probability')
    ax6a.set_ylabel('mean concurrence')
    ax6a.set_title('Entanglement vs loop loss')

    ax6b.plot(loss_values, cap_vs_loss, 's-', color='C0')
    ax6b.set_xlabel('loop loss probability')
    ax6b.set_ylabel('memory capacity C')
    ax6b.set_title('Memory capacity vs loop loss')

    fig6.savefig('../figures/3q_memristor_loss_scan.pdf', bbox_inches='tight')
    print("    Saved: 3q_memristor_loss_scan.pdf")

    # --- 7. Classical limit (high loss -> classical feedback) ---
    print("\n[7] Verifying classical limit (high loss)...")
    g_cl = make_gate_layered(phi_tp=np.pi/3, phi_pc=np.pi/4, phi_tc=np.pi/5,
                             theta_t=np.pi/6, theta_p=np.pi/4, theta_c=np.pi/8)
    sim_classical = QuantumMemristorLoop(g_cl, loop_loss=0.99)
    states_cl, angles_cl = sinusoidal_input_states(300, omega)
    res_cl = sim_classical.run(states_cl, list(angles_cl), bases='Z')
    avg_conc_cl = np.mean([r.concurrence for r in res_cl[20:]])
    print(f"    Concurrence at loss=0.99: {avg_conc_cl:.6f}")
    print(f"    (Should be ~= 0, confirming classical limit)")

    plt.show()
    print("\n[Done]")


if __name__ == "__main__":
    main()
