"""Palimpsest forgetting rate of Sec. III B (Fig. 3(d)) from figures/data/fix_panel_d_palimpsest.npz.
Usage: python fit_palimpsest.py [path-to-npz]; default: ../figures/data/fix_panel_d_palimpsest.npz.
ln R(age) linear fit over ages 0..4 (column 0 = newest pattern), percentile bootstrap over the 60 streams."""
import numpy as np, sys, os
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures', 'data', 'fix_panel_d_palimpsest.npz')
p = np.load(sys.argv[1] if len(sys.argv) > 1 else D); X = p['per_fade']; ns = X.shape[0]
mf = X.mean(0); print('R(age 0..5) =', np.round(mf[:6], 4))
fit = lambda M: -np.polyfit(np.arange(5), np.log(M[:5]), 1)[0]
print(f'point estimate rate = {fit(mf):.4f}')
for seed in (0, 1, 2, 20260912):
    rng = np.random.default_rng(seed); b = []; bad = 0
    for _ in range(20000):
        M = X[rng.integers(0, ns, ns)].mean(0)
        if (M[:5] <= 0).any(): bad += 1; continue
        b.append(fit(M))
    b = np.array(b); print(f'seed {seed}: 95% [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]  (dropped {bad} resamples with R=0)')
# sensitivity: weighted fit, fit range 0..3
w = lambda M, k: -np.polyfit(np.arange(k), np.log(M[:k]), 1)[0]
print(f'ages 0..3: {w(mf,4):.3f};  ages 0..5: {w(mf,6) if (mf[:6]>0).all() else float("nan"):.3f}')
