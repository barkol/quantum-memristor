"""Summarizes results/capacity_runs/: exact-recovery capacities for n = 64, 128, 256 over 12 seeds, for random words with p = 2, 3 and codewords
with p = 2, with power-law slopes and bootstrap intervals over seeds. Prints the numbers quoted in Part II and writes figures/data/fig_capacity_code.npz."""
import glob, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
d = {}
for f in glob.glob(str(HERE.parent/"results"/"capacity_runs"/"cap_*_s*.json")):
    for k, v in json.load(open(f)).items():
        n, kind, p, s = k.split("_"); d[(int(n), kind, int(p), int(s[1:]))] = v["cap"]
ns, S = [64, 128, 256], range(12); rng = np.random.default_rng(0); out = {"n": np.array(ns)}
for tag, kind, p in (("rand2", "random", 2), ("rand3", "random", 3), ("code2", "code", 2)):
    C = np.array([[d[(n, kind, p, s)] for s in S] for n in ns], float); ln = np.log(ns)
    sl = lambda M: np.polyfit(ln, np.log(M.mean(1)), 1)[0]
    bs = [sl(C[:, rng.integers(0, 12, 12)]) for _ in range(5000)]
    print(tag, " ".join(f"{n}: {C[i].mean():.1f} +- {C[i].std(ddof=1):.1f}" for i, n in enumerate(ns)), f"slope {sl(C):.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
    out["mean_"+tag], out["sd_"+tag] = C.mean(1), C.std(1, ddof=1)
print("rand3 / (n-1):", np.round(out["mean_rand3"]/(np.array(ns)-1), 2), " code2 / rand2:", np.round(out["mean_code2"]/out["mean_rand2"], 2), " rand3 / code2:", np.round(out["mean_rand3"]/out["mean_code2"], 2))
np.savez(HERE.parent/"figures"/"data"/"fig_capacity_code.npz", **out)
