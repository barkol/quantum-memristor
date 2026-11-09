#!/usr/bin/env python3
r"""Is the collapse of (anneal - greedy_Omega=0) at n=12 a size effect, or an
artefact of the fixed input bias h=0.6, which was tuned at n=8?

h is fed to the quantum anneal and to SA as well as to the biased greedy
control, so a mis-scaled h penalises all three and could fake a size effect.

Design note: the networks and probe batches are drawn ONCE per n and reused for
every h, so the h-dependence is measured on identical data and the disorder
noise cancels in the paired differences.  The 'scaled' variant sets
h_i = median_j |(J s_noisy)_j| * xi_noisy_i per trial, the coupling-only
local-field scale that SA already uses to pick its starting temperature.

Checkpoint after every (n, h) cell.
"""
import numpy as np, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_qhnm as m
from fixes_compute import greedy_biased_batch

DATA = Path(__file__).resolve().parent.parent / "figures" / "data"
DATA.mkdir(parents=True, exist_ok=True)
CKPT = DATA / "fix_sweep_bias.npz"
L, F, OMEGA0, T_ANNEAL, K_ANNEAL = 3, 0.1875, 1.0, 12.0, 120
HVALS = [0.1, 0.15, 0.2, 0.3, 0.6, 1.2, 2.4, "scaled"]
NET, TR = 20, 100

res = {}
if CKPT.exists():
    res = json.loads(str(np.load(CKPT, allow_pickle=True)["res"]))
    print(f"resuming; {len(res)} cells done", flush=True)

def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96*x.std(ddof=1)/np.sqrt(len(x)))]

for n in (8, 12, 14):
    spins = m.spin_table(n); eta, gamma = 1.0/n, 0.01
    nflip = int(np.floor(F*n + 0.5))
    # ---- draw the networks and probes ONCE for this n -------------------
    rng = np.random.default_rng(660000 + n)
    fixture = []
    for _ in range(NET):
        P = rng.choice([-1, 1], size=(L, n))
        J, _ = m.stream_memristor_couplings(P, eta=eta, gamma=gamma,
                                            dt=1.0, steps_per_pattern=1)
        tb = P[rng.integers(0, L, size=TR)].copy()
        nb = tb.copy()
        fl = np.argsort(rng.random((TR, n)), axis=1)[:, :nflip]
        nb[np.repeat(np.arange(TR), nflip), fl.ravel()] *= -1
        fixture.append((J, tb, nb))
    # greedy_unbiased does not see h at all: compute it once per network
    m.RNG = np.random.default_rng(700000 + n)
    o_gu_fixed = [m.overlap_batch(m.retrieve_classical_batch(J, nb), tb)
                  for J, tb, nb in fixture]

    for hi, hspec in enumerate(HVALS):
        key = f"{n}_{hspec}"
        if key in res:
            print(f"  [skip] n={n} h={hspec}"); continue
        m.RNG = np.random.default_rng(770000 + 1000*n + 17*hi)
        acc = {k: [] for k in ("gu", "gb", "sa", "qa")}
        d_sa_qa, d_qa_gu, d_gb_qa = [], [], []
        for (J, tb, nb), o_gu in zip(fixture, o_gu_fixed):
            if hspec == "scaled":
                bias = np.median(np.abs(nb.astype(float) @ J), axis=1)[:, None]
            else:
                bias = float(hspec)
            o_qa = m.overlap_batch(m.retrieve_quantum_anneal_batch(
                J, nb, Omega0=OMEGA0, n_steps=K_ANNEAL, T=T_ANNEAL,
                bias_strength=bias, spins=spins), tb)
            o_sa = m.overlap_batch(m.retrieve_classical_sa_batch(
                J, nb, bias_strength=bias), tb)
            o_gb = m.overlap_batch(greedy_biased_batch(J, nb, bias_strength=bias), tb)
            for k, v in (("qa", o_qa), ("sa", o_sa), ("gu", o_gu), ("gb", o_gb)):
                acc[k].append(v.mean())
            d_sa_qa.append((o_sa - o_qa).mean())
            d_qa_gu.append((o_qa - o_gu).mean())
            d_gb_qa.append((o_gb - o_qa).mean())
        res[key] = dict(n=n, h=str(hspec), f=F, nflip=nflip, n_nets=NET, n_trials=TR,
                        anneal=ci(acc["qa"]), sa=ci(acc["sa"]),
                        greedy_unbiased=ci(acc["gu"]), greedy_biased=ci(acc["gb"]),
                        paired_sa_minus_anneal=ci(d_sa_qa),
                        paired_anneal_minus_greedyunb=ci(d_qa_gu),
                        paired_greedybias_minus_anneal=ci(d_gb_qa))
        np.savez(CKPT, res=json.dumps(res))
        r = res[key]
        lo = r["paired_anneal_minus_greedyunb"][0] - r["paired_anneal_minus_greedyunb"][1]
        star = "  <== anneal > greedy" if lo > 0 else ""
        print(f"n={n:3d} h={str(hspec):>6}:  anneal={r['anneal'][0]:.4f}  SA={r['sa'][0]:.4f}  "
              f"gU={r['greedy_unbiased'][0]:.4f}  gB={r['greedy_biased'][0]:.4f}", flush=True)
        print(f"        SA-anneal={r['paired_sa_minus_anneal'][0]:+.4f}+-{r['paired_sa_minus_anneal'][1]:.4f}   "
              f"anneal-gU={r['paired_anneal_minus_greedyunb'][0]:+.4f}+-{r['paired_anneal_minus_greedyunb'][1]:.4f}{star}   "
              f"gB-anneal={r['paired_greedybias_minus_anneal'][0]:+.4f}+-{r['paired_greedybias_minus_anneal'][1]:.4f}", flush=True)
print("BIAS SWEEP DONE", flush=True)
