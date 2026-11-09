#!/usr/bin/env python3
r"""Does the annealer's loss survive re-tuning its own schedule?

The Appendix reports the anneal at the parameters used throughout the paper
(Omega0 = 1, T = 12, K = 120).  A referee will ask the obvious question: is the
quantum schedule losing because quantum annealing does not help here, or because
it was run at badly chosen parameters?  This script answers it by scanning the
schedule over total time T and transverse-field scale Omega0, on exactly the
same networks and probe batches as the classical baselines, so the comparison is
paired and the disorder noise cancels.

Baselines: greedy descent on the input-biased problem (the like-for-like
control), greedy descent on the couplings alone (the control usually drawn), and
simulated annealing on the biased problem.

Checkpoint after every (T, Omega0) cell.
"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_qhnm as m
from fixes_compute import greedy_biased_batch, corrupt

DATA = Path(__file__).resolve().parent.parent / "figures" / "data"
DATA.mkdir(parents=True, exist_ok=True)
CKPT = DATA / "fix_anneal_tuning.npz"

N, L, F, BIAS = 8, 3, 0.1875, 0.6
K_STEPS = 120
TVALS = [6.0, 12.0, 25.0, 50.0]
OMVALS = [0.25, 0.5, 1.0, 2.0]
NET, TR = 60, 100
SEED = 20260912

done = {}
if CKPT.exists():
    z = np.load(CKPT, allow_pickle=True)
    done = {k: z[k] for k in z.files}
    print(f"resuming; {len([k for k in done if k.startswith('anneal_')])} cells done", flush=True)

spins = m.spin_table(N)
nflip = int(np.floor(F * N + 0.5))
rng = np.random.default_rng(SEED)

# fixtures: the same networks and probes for every schedule and every baseline
nets = []
for _ in range(NET):
    P = rng.choice([-1.0, 1.0], size=(L, N))
    J, _ = m.stream_memristor_couplings(P, eta=1.0 / N, gamma=0.01, dt=1.0,
                                        steps_per_pattern=1)
    tgt = rng.integers(0, L, TR)
    true_b = P[tgt]
    noisy = corrupt(rng, true_b, nflip, N)
    nets.append((J, true_b, noisy))

def ci(per_net):
    x = np.asarray(per_net, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))

def report(tag, per_net):
    mu, err = ci(per_net)
    done[tag] = np.array([mu, err])
    print(f"  {tag:28s} {mu:.4f} +- {err:.4f}", flush=True)

# ---- classical baselines (schedule-independent, computed once) -------------
if "greedy_biased" not in done:
    gb, gu, sa = [], [], []
    for J, true_b, noisy in nets:
        gb.append(m.overlap_batch(greedy_biased_batch(J, noisy, BIAS), true_b).mean())
        gu.append(m.overlap_batch(m.retrieve_classical_batch(J, noisy), true_b).mean())
        sa.append(m.overlap_batch(m.retrieve_classical_sa_batch(J, noisy, BIAS), true_b).mean())
    report("greedy_biased", gb); report("greedy_unbiased", gu); report("sa_biased", sa)
    np.savez(CKPT, **done)

# ---- the anneal, over its own schedule ------------------------------------
for T in TVALS:
    for Om in OMVALS:
        tag = f"anneal_T{T:g}_Om{Om:g}"
        if tag in done:
            print(f"  [skip] {tag}", flush=True); continue
        per_net, trot = [], []
        for J, true_b, noisy in nets:
            R = m.retrieve_quantum_anneal_batch(J, noisy, Omega0=Om, n_steps=K_STEPS,
                                                T=T, bias_strength=BIAS, spins=spins)
            per_net.append(m.overlap_batch(R, true_b).mean())
            Ed = m.classical_energy(J, np.zeros(N), spins)
            trot.append(np.abs(Ed).max() * (T / K_STEPS))
        report(tag, per_net)
        done[tag + "_trotter"] = np.array([float(np.mean(trot))])
        np.savez(CKPT, **done)

print("\nBest anneal cell vs the biased greedy control:")
gbm = done["greedy_biased"]
best = max((k for k in done if k.startswith("anneal_") and not k.endswith("_trotter")),
           key=lambda k: done[k][0])
print(f"  greedy_biased {gbm[0]:.4f} +- {gbm[1]:.4f}")
print(f"  {best} {done[best][0]:.4f} +- {done[best][1]:.4f}"
      f"   (max|Ediag|*dt = {done[best + '_trotter'][0]:.2f})")
print(f"  difference (anneal - greedy) = {done[best][0] - gbm[0]:+.4f}")
print(f"\nwrote {CKPT}")
