#!/usr/bin/env python3
r"""Does the Fig.2(b) ordering survive a larger register?

Fig.2(b) of the manuscript is n=8, L=3 and is the ONLY empirical basis for the
abstract's "associative recall of classical patterns yields no quantum
advantage".  Here we repeat it verbatim at n=12 and n=14 with the appendix
protocol (Omega0=1, T=12, K=120 symmetric Trotter, midpoint schedule, input
bias h_i = 0.6*xi_noisy, SA with 300 sweeps / 3 restarts).

Four rules per cell:
  greedy_unbiased : the Omega=0 control as drawn in the paper (no h term)
  greedy_biased   : same descent on the input-biased problem (like-for-like)
  sa              : simulated annealing on the input-biased problem
  anneal          : quantum anneal, appendix parameters

Everything paired on the SAME network and the SAME probe batch; confidence
intervals are taken across networks.  Checkpoint after every (n, f) cell.
"""
import numpy as np, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_qhnm as m
from fixes_compute import greedy_biased_batch, corrupt

DATA = Path(__file__).resolve().parent.parent / "figures" / "data"
DATA.mkdir(parents=True, exist_ok=True)
CKPT = DATA / "fix_scale_fig2b.npz"

L, BIAS, OMEGA0, T_ANNEAL, K_ANNEAL = 3, 0.6, 1.0, 12.0, 120
FGRID = [0.0625, 0.125, 0.1875, 0.25, 0.3125]
CFG = {12: (40, 100), 14: (30, 100)}          # n -> (n_nets, n_trials)

res = {}
if CKPT.exists():
    res = json.loads(str(np.load(CKPT, allow_pickle=True)["res"]))
    print(f"resuming; {len(res)} cells already done", flush=True)

def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96*x.std(ddof=1)/np.sqrt(len(x)))]

for n, (NET, TR) in CFG.items():
    spins = m.spin_table(n)
    eta, gamma = 1.0/n, 0.01
    for f in FGRID:
        key = f"{n}_{f}"
        if key in res:
            print(f"  [skip] n={n} f={f}"); continue
        nflip = int(np.floor(f*n + 0.5))
        rng = np.random.default_rng(770000 + 1000*n + int(f*10000))
        m.RNG = np.random.default_rng(880000 + 1000*n + int(f*10000))
        acc = {k: [] for k in ("gu", "gb", "sa", "qa")}
        d_sa_qa, d_qa_gu, d_sa_gu, d_gb_qa = [], [], [], []
        for _ in range(NET):
            P = rng.choice([-1, 1], size=(L, n))
            J, _ = m.stream_memristor_couplings(P, eta=eta, gamma=gamma,
                                                dt=1.0, steps_per_pattern=1)
            tb = P[rng.integers(0, L, size=TR)].copy()
            nb = tb.copy()
            if nflip > 0:
                fl = np.argsort(rng.random((TR, n)), axis=1)[:, :nflip]
                nb[np.repeat(np.arange(TR), nflip), fl.ravel()] *= -1
            o_qa = m.overlap_batch(m.retrieve_quantum_anneal_batch(
                J, nb, Omega0=OMEGA0, n_steps=K_ANNEAL, T=T_ANNEAL,
                bias_strength=BIAS, spins=spins), tb)
            o_sa = m.overlap_batch(m.retrieve_classical_sa_batch(J, nb), tb)
            o_gu = m.overlap_batch(m.retrieve_classical_batch(J, nb), tb)
            o_gb = m.overlap_batch(greedy_biased_batch(J, nb), tb)
            acc["qa"].append(o_qa.mean()); acc["sa"].append(o_sa.mean())
            acc["gu"].append(o_gu.mean()); acc["gb"].append(o_gb.mean())
            d_sa_qa.append((o_sa - o_qa).mean())
            d_qa_gu.append((o_qa - o_gu).mean())
            d_sa_gu.append((o_sa - o_gu).mean())
            d_gb_qa.append((o_gb - o_qa).mean())
        res[key] = dict(n=n, f=f, nflip=nflip, n_nets=NET, n_trials=TR,
                        greedy_unbiased=ci(acc["gu"]), greedy_biased=ci(acc["gb"]),
                        sa=ci(acc["sa"]), anneal=ci(acc["qa"]),
                        paired_sa_minus_anneal=ci(d_sa_qa),
                        paired_anneal_minus_greedyunb=ci(d_qa_gu),
                        paired_sa_minus_greedyunb=ci(d_sa_gu),
                        paired_greedybias_minus_anneal=ci(d_gb_qa))
        np.savez(CKPT, res=json.dumps(res))
        r = res[key]
        print(f"n={n} f={f:.4f} (nflip={nflip}, {NET}x{TR}):  "
              f"anneal={r['anneal'][0]:.4f}  SA={r['sa'][0]:.4f}  "
              f"greedy_unb={r['greedy_unbiased'][0]:.4f}  greedy_bias={r['greedy_biased'][0]:.4f}", flush=True)
        print(f"      PAIRED  SA-anneal={r['paired_sa_minus_anneal'][0]:+.4f}+-{r['paired_sa_minus_anneal'][1]:.4f}"
              f"   anneal-greedyUnb={r['paired_anneal_minus_greedyunb'][0]:+.4f}+-{r['paired_anneal_minus_greedyunb'][1]:.4f}"
              f"   greedyBias-anneal={r['paired_greedybias_minus_anneal'][0]:+.4f}+-{r['paired_greedybias_minus_anneal'][1]:.4f}", flush=True)
print("SCALE-FIG2B DONE", flush=True)
