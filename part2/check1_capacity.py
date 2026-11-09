#!/usr/bin/env python3
"""check1 — R2/R3: czy pojemnosc kodowa naprawde nasyca n-1, i ile wynosi
niecenzurowana pojemnosc p=3 przy n=256 (poprzednio ucieta przez hi_cap=6000)."""
import numpy as np, sys, json
from pathlib import Path
from scipy.linalg import hadamard
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import descend
THR, FF, TRIALS = 0.9, 0.15, 120

def success(n, M, kind, p, seed):
    rg = np.random.default_rng(seed)
    if kind == "code":
        H = hadamard(n).astype(float)
        P = H[rg.choice(np.arange(1, n), M, replace=False)].copy()
    else:
        P = rg.choice([-1., 1.], size=(M, n))
    tgt = rg.integers(0, M, TRIALS); S = P[tgt].copy(); k = int(FF*n)
    for i in range(TRIALS): S[i, rg.choice(n, k, replace=False)] *= -1
    return float(np.mean(np.argmax(descend(S, P, p, rng=np.random.default_rng(1)) @ P.T, axis=1) == tgt))

out = {}
print("R2 — czy M = n-1 przechodzi prog 0.9 dla slow kodowych, p=2?")
for n in (64, 128, 256):
    s = [success(n, n-1, "code", 2, 100+7*k) for k in range(3)]
    out[f"code2_n{n}_M{n-1}"] = s
    print(f"  n={n:4d}  M=n-1={n-1:4d}:  sukces {np.round(s,4).tolist()}  -> {'PASS' if min(s)>=THR else 'FAIL'}", flush=True)
print("\nR3 — pojemnosc p=3, n=256, siatka POWYZEJ dawnego hi_cap=6000")
for M in (5762, 6000, 6500, 7000, 7500, 8000):
    s = [success(256, M, "random", 3, 100+7*k) for k in range(2)]
    out[f"rand3_n256_M{M}"] = s
    print(f"  M={M:5d}:  sukces {np.round(s,4).tolist()}  -> {'PASS' if min(s)>=THR else 'FAIL'}", flush=True)
Path("check1_wynik.json").write_text(json.dumps(out, indent=1))
print("\nzapisano check1_wynik.json")
