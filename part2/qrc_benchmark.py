#!/usr/bin/env python3
r"""NARMA-10 benchmark of the six-memristor reservoir, removed from the papers and kept for provenance.
The target is aligned with the standard definition: the state after u[t] is trained on y[t+1], which depends on u[t] and u[t-9].
Data are redrawn for each seed; the device (rate bank and couplings) is fixed at seed 1234 unless stated. Train 200..899, test 900..1198.
Models: nominal reservoir, linear filters over the last 10 and 27 inputs, echo-state network of the same size, and a sweep of the
reservoir over gamma. Every finished job is appended to results/qrc_benchmark.jsonl; a restart skips it.
Usage: python qrc_benchmark.py [workers]. Output: results/qrc_benchmark.jsonl, figures/data/fig_qrc_gamma.npz, printed summary."""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import sim_qrc_narma as q

OUT = HERE.parent/"results"/"qrc_benchmark.jsonl"; NPZ = HERE.parent/"figures"/"data"/"fig_qrc_gamma.npz"
T, NQ, DEV, GAMMA0 = 1200, 6, 1234, 0.03
TRAIN, TEST = slice(200, 900), slice(900, T - 1)
SEEDS = range(30)
GAMMAS = np.geomspace(0.01, 1.2, 16)


def data(seed):
    u, y = q.narma10(T, np.random.default_rng(10_000 + seed))
    return u, np.r_[y[1:], y[-1]]


def reservoir(u, gamma, dev):
    return q.MemristiveQRC(n=NQ, gamma=gamma, eta=0.9, theta0=0.15, coupling=0.35, reset=0.03, gain=0.35,
                           gamma_spread=3.0, seed=dev).run(u)


def job(key):
    kind, seed, g = key
    u, t = data(seed)
    if kind == "res": f = reservoir(u, GAMMA0, DEV)
    elif kind == "res_eta0": f = q.MemristiveQRC(n=NQ, gamma=GAMMA0, eta=0.0, theta0=0.15, coupling=0.35, reset=0.03, gain=0.35, gamma_spread=3.0, seed=DEV).run(u)
    elif kind == "res_dev": f = reservoir(u, GAMMA0, 20_000 + seed)
    elif kind == "sweep": f = reservoir(u, GAMMAS[g], DEV)
    elif kind == "lin10": f = q.linear_floor_features(u, t, mem=10)
    elif kind == "lin27": f = q.linear_floor_features(u, t, mem=27)
    elif kind == "esn": f = q.esn_features(u, 27, np.random.default_rng(30_000 + seed))
    return kind, seed, g, float(q.evaluate(f, t, TRAIN, TEST)[0])


if __name__ == "__main__":
    done = {}
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            r = json.loads(l); done[(r[0], r[1], r[2])] = r[3]
    jobs = [(k, s, -1) for k in ("res", "res_dev", "res_eta0", "lin10", "lin27", "esn") for s in SEEDS]
    jobs += [("sweep", s, g) for s in SEEDS for g in range(len(GAMMAS))]
    todo = [j for j in jobs if j not in done]
    print("jobs", len(jobs), "to do", len(todo), flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p, OUT.open("a") as f:
        for r in p.imap_unordered(job, todo):
            done[r[:3]] = r[3]; f.write(json.dumps(r) + "\n"); f.flush()
    v = lambda k: np.array([done[(k, s, -1)] for s in SEEDS])
    for k in ("res", "res_dev", "res_eta0", "lin10", "lin27", "esn"):
        print(f"{k:8s} {v(k).mean():.3f} +- {v(k).std(ddof=1):.3f}")
    d = v("res") - v("lin10"); print(f"res - lin10 paired {d.mean():.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}, lin10 better in {(d>0).mean():.2f}")
    S = np.array([[done[("sweep", s, g)] for g in range(len(GAMMAS))] for s in SEEDS])
    m = S.mean(0); print("gamma ", np.round(GAMMAS, 3).tolist()); print("mean  ", np.round(m, 3).tolist()); print("sd    ", np.round(S.std(0, ddof=1), 3).tolist())
    b = S.argmin(1); print("best gamma of the mean", round(GAMMAS[m.argmin()], 3), "per-seed quartiles", np.round(np.percentile(GAMMAS[b], [25, 50, 75]), 3).tolist())
    print("seeds with an interior maximum above gamma=0.3 exceeding 1:", int(((S[:, GAMMAS > 0.3] > 1).any(1)).sum()), "of", len(S))
    np.savez(NPZ, gammas=GAMMAS, S=S, lin10=v("lin10"), lin27=v("lin27"), esn=v("esn"), res=v("res"))
