#!/usr/bin/env python3
"""Hebbian capacity of random patterns and of Hadamard codewords by bisection in the number of stored patterns M, for two-body (p = 2) and
three-body (p = 3) couplings. Quoted in Part II, Sec. "Codewords versus higher-order couplings", and described in the Appendix.
Protocol: a stored word corrupted in floor(0.15 n) spins, zero-temperature descent on E = -sum_mu m_mu^p (m_mu = overlap with word mu),
success = the descent returns the intended word exactly, 120 probes per M, threshold 0.9; doubling from M = 2 up to hi_cap (default 16384),
then bisection to a relative tolerance of 1%. Codewords: distinct random rows 1..n-1 of a Sylvester Hadamard matrix, so M <= n - 1. Seed of the
words: 100 + 7 s. A checkpoint is written after every (n, kind, p, seed) cell; a restart skips finished cells. The criterion "nearest" counts a
descent that ends closest to the intended word; the unchanged probe passes it far above these capacities, so it does not measure a memory and
is kept only for that comparison. Summarize the runs in results/capacity_runs/ with capacity_summary.py.
Usage: python capacity_code.py OUT.json n kind p seed_list(comma) [hi_cap] [reltol] [criterion]    (kind: random | code; criterion: exact | nearest)"""
import numpy as np, json, sys, time
from pathlib import Path
from scipy.linalg import hadamard
THR, FF, TRIALS = 0.9, 0.15, 120
CRITERION = "exact"

def descend(S, P, p, sweeps=60, rng=None):
    rng = rng or np.random.default_rng(0); n = S.shape[1]
    m = S @ P.T
    for _ in range(sweeps):
        moved = np.zeros(S.shape[0], dtype=bool)
        for i in rng.permutation(n):
            ds = -2.0 * S[:, i]
            mnew = m + np.outer(ds, P[:, i])
            dE = -(np.sum(mnew**p, axis=1) - np.sum(m**p, axis=1))
            fl = dE < -1e-9
            S[fl, i] *= -1; m[fl] = mnew[fl]; moved |= fl
        if not moved.any(): break
    return S

def success(n, M, kind, p, seed):
    rg = np.random.default_rng(seed)
    if kind == "code":
        if M > n - 1: return 0.0
        H = hadamard(n).astype(float)
        P = H[rg.choice(np.arange(1, n), M, replace=False)].copy()
    else:
        P = rg.choice([-1., 1.], size=(M, n))
    tgt = rg.integers(0, M, TRIALS); S = P[tgt].copy(); k = int(FF * n)
    for i in range(TRIALS): S[i, rg.choice(n, k, replace=False)] *= -1
    out = descend(S, P, p, rng=np.random.default_rng(1))
    if CRITERION == "nearest": return float(np.mean(np.argmax(out @ P.T, axis=1) == tgt))
    return float(np.mean(np.all(out == P[tgt], axis=1)))

def capacity(n, kind, p, seed, hi_cap, reltol, log):
    lo, hi = 1, 2
    hard = n - 1 if kind == "code" else hi_cap
    while hi <= hard:
        s = success(n, hi, kind, p, seed); log.append((hi, s))
        if s >= THR: lo = hi; hi *= 2
        else: break
    hi = min(hi, hard)
    if lo >= hard: return hard
    while hi - lo > max(1, int(lo * reltol)):
        mid = (lo + hi) // 2
        s = success(n, mid, kind, p, seed); log.append((mid, s))
        if s >= THR: lo = mid
        else: hi = mid
    return lo

if __name__ == "__main__":
    out = Path(sys.argv[1]); n = int(sys.argv[2]); kind = sys.argv[3]; p = int(sys.argv[4])
    seeds = [int(x) for x in sys.argv[5].split(",")]
    hi_cap = int(sys.argv[6]) if len(sys.argv) > 6 else 16384
    reltol = float(sys.argv[7]) if len(sys.argv) > 7 else 0.01
    if len(sys.argv) > 8: CRITERION = sys.argv[8]
    res = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    for s in seeds:
        key = f"{n}_{kind}_{p}_s{s}"
        if key in res: print("skip", key, flush=True); continue
        log = []
        c = capacity(n, kind, p, 100 + 7 * s, hi_cap, reltol, log)
        res[key] = {"cap": c, "probes": log, "hi_cap": hi_cap, "reltol": reltol}
        out.write_text(json.dumps(res, indent=1))
        print(f"{key}: cap={c} [{time.time()-t0:.0f}s]", flush=True)
