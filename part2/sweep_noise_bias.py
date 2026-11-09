#!/usr/bin/env python3
r"""Does the optimal input-bias strength h* depend on the noise level f?

Every earlier h-sweep was run at f=0.1875 only, so all conclusions about h were
conditioned on one corruption level.  Here we sweep (n, f, h) jointly.

Design: the networks and probe batches are drawn once per (n, f) and shared by
every h, and greedy_Omega=0 (which never sees h) is computed once per (n, f),
so the h-dependence is paired and the disorder noise cancels.

Each cell also records the Trotter-resolution diagnostic max|Ediag|*dt: the
annealer's splitting error grows with it, and a separate gate run showed the
integrator stops returning the analytically known answer once this becomes
large (it is fine over the whole grid used here).

Confidence intervals are stored BOTH as 1.96 sigma/sqrt(N) and as a bootstrap
percentile interval over networks, so the two can be cross-checked.
Checkpoint after every (n, f, h) cell.
"""
import numpy as np, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_qhnm as m
from fixes_compute import greedy_biased_batch

DATA = Path(__file__).resolve().parent.parent / "figures" / "data"
DATA.mkdir(parents=True, exist_ok=True)
CKPT = DATA / "fix_noise_bias.npz"
L, OMEGA0, T_ANNEAL, K_ANNEAL = 3, 1.0, 12.0, 120
FVALS = [0.0625, 0.125, 0.1875, 0.25, 0.3125]
HVALS = [0.1, 0.2, 0.3, 0.45, 0.6]
NET, TR = 20, 100
BOOT = 2000

res = {}
if CKPT.exists():
    res = json.loads(str(np.load(CKPT, allow_pickle=True)["res"]))
    print(f"resuming; {len(res)} cells done", flush=True)

def stats(x, rng):
    x = np.asarray(x, float)
    mu = float(x.mean())
    ci = float(1.96*x.std(ddof=1)/np.sqrt(len(x)))
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [mu, ci, float(lo), float(hi)]

for n in (8, 12):
    spins = m.spin_table(n)
    eta, gamma = 1.0/n, 0.01
    dt = T_ANNEAL / K_ANNEAL
    Ecoup = None
    for f in FVALS:
        nflip = int(np.floor(f*n + 0.5))
        # ---- fixture drawn once per (n, f) -----------------------------
        rng = np.random.default_rng(510000 + 1000*n + int(round(f*10000)))
        fixture = []
        for _ in range(NET):
            P = rng.choice([-1, 1], size=(L, n))
            J, _ = m.stream_memristor_couplings(P, eta=eta, gamma=gamma,
                                                dt=1.0, steps_per_pattern=1)
            tb = P[rng.integers(0, L, size=TR)].copy(); nb = tb.copy()
            if nflip:
                fl = np.argsort(rng.random((TR, n)), axis=1)[:, :nflip]
                nb[np.repeat(np.arange(TR), nflip), fl.ravel()] *= -1
            fixture.append((J, tb, nb))
        m.RNG = np.random.default_rng(520000 + 1000*n + int(round(f*10000)))
        o_gu_fixed = [m.overlap_batch(m.retrieve_classical_batch(J, nb), tb)
                      for J, tb, nb in fixture]
        for hi_, h in enumerate(HVALS):
            key = f"{n}_{f}_{h}"
            if key in res:
                continue
            m.RNG = np.random.default_rng(530000 + 1000*n + 97*hi_ + int(round(f*10000)))
            brng = np.random.default_rng(4242)
            acc = {k: [] for k in ("gu", "gb", "sa", "qa")}
            d_sa_qa, d_qa_gu, d_gb_qa, trot = [], [], [], []
            for (J, tb, nb), o_gu in zip(fixture, o_gu_fixed):
                Ec = m.classical_energy(J, np.zeros(n), spins)
                Ed = Ec[:, None] - spins @ (h*nb.astype(float)).T
                trot.append(float(np.abs(Ed).max()*dt))
                o_qa = m.overlap_batch(m.retrieve_quantum_anneal_batch(
                    J, nb, Omega0=OMEGA0, n_steps=K_ANNEAL, T=T_ANNEAL,
                    bias_strength=h, spins=spins), tb)
                o_sa = m.overlap_batch(m.retrieve_classical_sa_batch(J, nb, bias_strength=h), tb)
                o_gb = m.overlap_batch(greedy_biased_batch(J, nb, bias_strength=h), tb)
                for k, v in (("qa", o_qa), ("sa", o_sa), ("gu", o_gu), ("gb", o_gb)):
                    acc[k].append(v.mean())
                d_sa_qa.append((o_sa - o_qa).mean())
                d_qa_gu.append((o_qa - o_gu).mean())
                d_gb_qa.append((o_gb - o_qa).mean())
            res[key] = dict(n=n, f=f, h=h, nflip=nflip, n_nets=NET, n_trials=TR,
                            trotter_max=float(np.max(trot)),
                            anneal=stats(acc["qa"], brng), sa=stats(acc["sa"], brng),
                            greedy_unbiased=stats(acc["gu"], brng),
                            greedy_biased=stats(acc["gb"], brng),
                            paired_sa_minus_anneal=stats(d_sa_qa, brng),
                            paired_anneal_minus_greedyunb=stats(d_qa_gu, brng),
                            paired_greedybias_minus_anneal=stats(d_gb_qa, brng))
            np.savez(CKPT, res=json.dumps(res))
            r = res[key]; a = r["paired_anneal_minus_greedyunb"]
            sig = "SIG" if a[0]-a[1] > 0 else "   "
            print(f"n={n:2d} f={f:.4f} h={h:.2f} (nflip={nflip}, trot={r['trotter_max']:.2f}): "
                  f"QA={r['anneal'][0]:.4f} SA={r['sa'][0]:.4f} gU={r['greedy_unbiased'][0]:.4f} "
                  f"gB={r['greedy_biased'][0]:.4f} | anneal-gU={a[0]:+.4f}+-{a[1]:.4f} {sig}", flush=True)
print("NOISE-BIAS SWEEP DONE", flush=True)
