"""Annealer (T, Omega0) scan at f=0.125, n=8, L=3, h=0.6, with the Appendix
protocol of sweep_anneal_tuning.py (K=120 Trotter steps for every T), on two draws.
Also the paper's fixed schedule (12,1). Paired differences gb - q per network."""
import sys, os, numpy as np, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_qhnm as m
from fixes_compute import greedy_biased_batch, corrupt
N, L, F, BIAS, NET, TR = 8, 3, 0.125, 0.6, 60, 100
TV = [6., 12., 25., 50.]; OV = [0.25, 0.5, 1., 2.]
se = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
out = {}
for SEED in (20260912, 4242):
    t0 = time.time(); rng = np.random.default_rng(SEED); m.RNG = np.random.default_rng(SEED + 1)
    sp = m.spin_table(N); nf = int(np.floor(F * N + .5)); nets = []
    for _ in range(NET):
        P = rng.choice([-1., 1.], size=(L, N))
        J, _ = m.stream_memristor_couplings(P, eta=1. / N, gamma=.01, dt=1., steps_per_pattern=1)
        tg = rng.integers(0, L, TR); tb = P[tg]; nets.append((J, tb, corrupt(rng, tb, nf, N)))
    gb = np.array([m.overlap_batch(greedy_biased_batch(J, nb, BIAS), tb).mean() for J, tb, nb in nets])
    sa = np.array([m.overlap_batch(m.retrieve_classical_sa_batch(J, nb, BIAS), tb).mean() for J, tb, nb in nets])
    print(f'seed {SEED}: gb={gb.mean():.4f}+-{se(gb):.4f} sa={sa.mean():.4f}+-{se(sa):.4f}', flush=True)
    for T in TV:
        for O in OV:
            q = np.array([m.overlap_batch(m.retrieve_quantum_anneal_batch(J, nb, Omega0=O, n_steps=120, T=T,
                          bias_strength=BIAS, spins=sp), tb).mean() for J, tb, nb in nets])
            d = gb - q; out[f'{SEED}_{T}_{O}'] = [q.mean(), d.mean(), se(d)]
            print(f'   T={T:4.0f} Om={O:4.2f}  q={q.mean():.4f}  gb-q={d.mean():+.4f}+-{se(d):.4f}', flush=True)
    print(f'  [{time.time()-t0:.0f}s]', flush=True)
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'part2')
os.makedirs(R, exist_ok=True)
json.dump(out, open(os.path.join(R, 'sweep_anneal_tuning_f0125.json'), 'w'), indent=1)
