#!/usr/bin/env python3
r"""Cheap correctness gates, run BEFORE the expensive (n,f,h) sweep.

G1  h->0 identity : greedy_biased_batch(h=0) must be bit-identical to
                    retrieve_classical_batch when the RNG state is matched,
                    because the h term is then exactly zero.
G2  h->inf gate   : with a bias far above the coupling scale every rule must
                    return the noisy input itself, so the sign-blind overlap is
                    exactly |n - 2*nflip| / n.  Checked for all three rules.
G3  broadcasting  : a per-trial (B,1) bias must give the same answer as a scalar
                    bias when all its entries equal that scalar.
Exit code 0 only if every gate passes.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_qhnm as m
from fixes_compute import greedy_biased_batch

L, TR = 3, 200
fails = []

def fixture(n, nflip, seed):
    rng = np.random.default_rng(seed)
    P = rng.choice([-1, 1], size=(L, n))
    J, _ = m.stream_memristor_couplings(P, eta=1.0/n, gamma=0.01, dt=1.0, steps_per_pattern=1)
    tb = P[rng.integers(0, L, size=TR)].copy(); nb = tb.copy()
    fl = np.argsort(rng.random((TR, n)), axis=1)[:, :nflip]
    nb[np.repeat(np.arange(TR), nflip), fl.ravel()] *= -1
    return J, tb, nb

for n, nflip in ((8, 1), (8, 2), (12, 2), (12, 3), (14, 3)):
    J, tb, nb = fixture(n, nflip, 4242 + n*10 + nflip)
    spins = m.spin_table(n)

    # ---- G1: h -> 0 identity ------------------------------------------
    st = np.random.default_rng(99).bit_generator.state
    m.RNG = np.random.default_rng(99); m.RNG.bit_generator.state = st
    a = greedy_biased_batch(J, nb, bias_strength=0.0)
    m.RNG = np.random.default_rng(99); m.RNG.bit_generator.state = st
    b = m.retrieve_classical_batch(J, nb)
    ok1 = np.array_equal(a, b)
    if not ok1: fails.append(f"G1 n={n} nflip={nflip}: greedy(h=0) != greedy_unbiased")

    # ---- G2: h -> inf must return the noisy input ----------------------
    target = abs(n - 2*nflip) / n
    m.RNG = np.random.default_rng(7)
    BIG = 50.0
    r_gb = m.overlap_batch(greedy_biased_batch(J, nb, bias_strength=BIG), tb).mean()
    r_sa = m.overlap_batch(m.retrieve_classical_sa_batch(J, nb, bias_strength=BIG), tb).mean()
    r_qa = m.overlap_batch(m.retrieve_quantum_anneal_batch(
        J, nb, Omega0=1.0, n_steps=120, T=12.0, bias_strength=BIG, spins=spins), tb).mean()
    # also: the noisy input itself
    r_in = m.overlap_batch(nb, tb).mean()
    ok2 = all(abs(v - target) < 1e-9 for v in (r_gb, r_sa, r_qa, r_in))
    if not ok2:
        fails.append(f"G2 n={n} nflip={nflip}: target={target:.6f} gB={r_gb:.6f} "
                     f"SA={r_sa:.6f} QA={r_qa:.6f} input={r_in:.6f}")

    # ---- G3: (B,1) bias array == scalar bias ---------------------------
    hscalar = 0.3
    harr = np.full((TR, 1), hscalar)
    st2 = np.random.default_rng(5).bit_generator.state
    for fn, kw in ((greedy_biased_batch, {}),
                   (m.retrieve_classical_sa_batch, {}),
                   (m.retrieve_quantum_anneal_batch,
                    dict(Omega0=1.0, n_steps=120, T=12.0, spins=spins))):
        m.RNG = np.random.default_rng(5); m.RNG.bit_generator.state = st2
        x = fn(J, nb, bias_strength=hscalar, **kw)
        m.RNG = np.random.default_rng(5); m.RNG.bit_generator.state = st2
        y = fn(J, nb, bias_strength=harr, **kw)
        if not np.array_equal(x, y):
            fails.append(f"G3 n={n} nflip={nflip} {fn.__name__}: scalar != (B,1) array bias")
    print(f"  n={n:3d} nflip={nflip}: G1={'ok' if ok1 else 'FAIL'}  "
          f"G2={'ok' if ok2 else 'FAIL'} (target {target:.4f})  G3 checked", flush=True)

print()
if fails:
    print("GATES FAILED:"); [print("  " + f) for f in fails]; sys.exit(1)
print("ALL GATES PASSED")
