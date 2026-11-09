"""Unbiased greedy control at n=8, L=2, f=0.125 (Sec. III B), independent implementation.
Varies two protocol choices: coupling model (streamed memristor vs exact Hebbian)
and the zero-field tie rule (repo: field>=0 -> +1; strict descent: flip only if dE<0).
Biased descent at h=0.6 and h=0.2 for comparison. Overlap = |m| (sign-blind), as in repo."""
import numpy as np, sys, time

def stream_J(P, eta, gamma=0.01):
    L, n = P.shape
    mu = np.zeros((n, n)); d = np.exp(-gamma)
    for x in P:
        c = np.outer(x, x).astype(float); np.fill_diagonal(c, 0)
        mu = mu * d + (eta * c / gamma) * (1 - d)
    np.fill_diagonal(mu, 0); return mu

def hebb_J(P):
    n = P.shape[1]; J = P.T @ P / n; np.fill_diagonal(J, 0); return J

def descent(J, S0, h, rule, rng, maxsw=200):
    S = S0.astype(float).copy(); n = S.shape[1]
    for _ in range(maxsw):
        ch = False
        for i in rng.permutation(n):
            fld = S @ J[i] + h[:, i]
            if rule == 'repo':          # retrieve_classical_batch / greedy_biased_batch
                new = np.where(fld >= 0, 1.0, -1.0)
            else:                       # strict: keep spin unless flip lowers E
                new = np.where(S[:, i] * fld < -1e-12, -S[:, i], S[:, i])
            if (new != S[:, i]).any(): ch = True
            S[:, i] = new
        if not ch: break
    return S

def run(seed, n=8, L=2, f=0.125, nets=60, tr=100):
    rng = np.random.default_rng(seed); nf = int(np.floor(f * n + .5))
    res = {}
    for _ in range(nets):
        P = rng.choice([-1., 1.], size=(L, n))
        tg = rng.integers(0, L, tr); tb = P[tg]; nb = tb.copy()
        fl = np.argsort(rng.random((tr, n)), axis=1)[:, :nf]
        nb[np.repeat(np.arange(tr), nf), fl.ravel()] *= -1
        for jn, J in (('stream', stream_J(P, 1. / n)), ('hebb', hebb_J(P))):
            for rule in ('repo', 'strict'):
                for hv in (0.0, 0.2, 0.6):
                    S = descent(J, nb, hv * nb, rule, np.random.default_rng(rng.integers(1 << 30)))
                    m = np.abs((S * tb).sum(1) / n).mean()
                    res.setdefault((jn, rule, hv), []).append(m)
    return res

if __name__ == '__main__':
    t0 = time.time()
    for seed in (21, 5, 99, 2026):
        r = run(seed)
        print(f'seed {seed}:', flush=True)
        for k, v in sorted(r.items()):
            v = np.array(v); print(f'   J={k[0]:6s} tie={k[1]:6s} h={k[2]:.1f}  m={v.mean():.4f} +- {1.96*v.std(ddof=1)/np.sqrt(len(v)):.4f}')
        print(f'  [{time.time()-t0:.0f}s]', flush=True)
