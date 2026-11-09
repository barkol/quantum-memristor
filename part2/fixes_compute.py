#!/usr/bin/env python3
r"""Compute layer for the audit fixes (figures/data/*.npz caches only, no plotting).

Blocks (each cached independently, skipped if its .npz already exists, so the
run is resumable after a crash/interrupt):
  A  panel_b_disorder  -- Fig.2(b) recomputed over independent disorder
                          realisations, with a FOURTH method: zero-T greedy
                          descent on the SAME input-biased problem.
  B  panel_a_shotnoise -- Fig.2(a) Hebbian correlation vs the coincidence
                          budget Gamma_c*dt, with the Poisson shot noise of
                          Eq.(6) actually sampled (not replaced by its mean).
  C  panel_d_palimpsest-- Fig.2(d) plus the missing gamma->0 (non-forgetting)
                          control that the "no catastrophic interference"
                          claim needs.
  D  dense_bootstrap   -- per-realisation capacities for Fig.3 so the log-log
                          slopes can carry bootstrap confidence intervals.
"""
import os, sys
from pathlib import Path
import numpy as np
import sim_qhnm as m

DATA = Path(__file__).resolve().parent.parent / "figures" / "data"
DATA.mkdir(parents=True, exist_ok=True)
def done(name): return (DATA / f"{name}.npz").exists()
def put(name, **kw):
    np.savez(DATA / f"{name}.npz", **kw); print(f"  [cached] {name}.npz", flush=True)

# ---------------------------------------------------------------- helpers
def greedy_biased_batch(J, noisy_batch, bias_strength=0.6, max_sweeps=200):
    """The control Fig.2(b) is missing: zero-T greedy descent on the SAME
    input-biased problem the quantum anneal and SA solve."""
    S = np.atleast_2d(noisy_batch).astype(float).copy()
    n = S.shape[1]
    h = bias_strength * np.atleast_2d(noisy_batch).astype(float)
    for _ in range(max_sweeps):
        changed = np.zeros(S.shape[0], dtype=bool)
        for i in m.RNG.permutation(n):
            new = np.where(S @ J[i] + h[:, i] >= 0.0, 1.0, -1.0)
            changed |= new != S[:, i]
            S[:, i] = new
        if not changed.any():
            break
    return S.astype(int)

def corrupt(RNG, true_b, nflip, n):
    nb = true_b.copy()
    if nflip > 0:
        B = true_b.shape[0]
        fl = np.argsort(RNG.random((B, n)), axis=1)[:, :nflip]
        nb[np.repeat(np.arange(B), nflip), fl.ravel()] *= -1
    return nb

# ---------------------------------------------------------------- A
def block_A():
    if done("fix_panel_b_disorder"): print("  [skip] A"); return
    m.RNG = np.random.default_rng(20260910); RNG = m.RNG
    n, L, eta, gamma = 8, 3, 1.0 / 8, 0.01
    spins = m.spin_table(n)
    noise = np.arange(0.0, 0.5001, 1.0 / 32.0)
    NET, TR = 20, 100                     # 2000 trials/point, 20 disorder draws
    K = len(noise)
    per = {k: np.zeros((NET, K)) for k in ("q", "sa", "greedy", "greedy_bias")}
    for net in range(NET):
        P = RNG.choice([-1, 1], size=(L, n))
        J, _ = m.stream_memristor_couplings(P, eta=eta, gamma=gamma, dt=1.0,
                                            steps_per_pattern=1)
        for fi, f in enumerate(noise):
            nflip = int(np.floor(f * n + 0.5))
            tb = P[RNG.integers(0, L, size=TR)].copy()
            nb = corrupt(RNG, tb, nflip, n)
            per["q"][net, fi] = m.overlap_batch(
                m.retrieve_quantum_anneal_batch(J, nb, Omega0=1.0, spins=spins), tb).mean()
            per["sa"][net, fi] = m.overlap_batch(
                m.retrieve_classical_sa_batch(J, nb), tb).mean()
            per["greedy"][net, fi] = m.overlap_batch(
                m.retrieve_classical_batch(J, nb), tb).mean()
            per["greedy_bias"][net, fi] = m.overlap_batch(
                greedy_biased_batch(J, nb), tb).mean()
        print(f"    A net {net+1}/{NET}", flush=True)
    put("fix_panel_b_disorder", noise=noise, n_nets=NET, n_trials=TR,
        **{f"per_{k}": v for k, v in per.items()})

# ---------------------------------------------------------------- B
def block_B():
    if done("fix_panel_a_shotnoise"): print("  [skip] B"); return
    n, L, gamma, dt = 8, 5, 0.01, 1.0
    eta = 1.0 / n
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(20260625)
    P = rng.choice([-1, 1], size=(L, n))
    Jh = (P.T @ P) / n; np.fill_diagonal(Jh, 0.0)
    decay = np.exp(-gamma * dt)
    budgets = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
    REP = 200
    mean_r, std_r = np.zeros(len(budgets)), np.zeros(len(budgets))
    for bi, G in enumerate(budgets):
        rs = np.zeros(REP)
        for rep in range(REP):
            mu = np.zeros((n, n))
            for k in range(L):
                xi = P[k]; corr = np.outer(xi, xi).astype(float)
                Ns = rng.poisson(np.maximum(G * (1.0 + corr) / 2.0, 0))
                No = rng.poisson(np.maximum(G * (1.0 - corr) / 2.0, 0))
                d = (Ns - No).astype(float)
                d = np.triu(d, 1); d = d + d.T; np.fill_diagonal(d, 0.0)
                mu = mu * decay + (eta * (d / G) / gamma) * (1 - decay)
            rs[rep] = np.corrcoef(mu[iu], Jh[iu])[0, 1]
        mean_r[bi], std_r[bi] = rs.mean(), rs.std()
        print(f"    B Gamma_c*dt={G}: r={rs.mean():.4f}+-{rs.std():.4f}", flush=True)
    # the deterministic (mean-field) drive actually used in the published figure
    mu = np.zeros((n, n))
    for k in range(L):
        c = np.outer(P[k], P[k]).astype(float); np.fill_diagonal(c, 0.0)
        mu = mu * decay + (eta * c / gamma) * (1 - decay)
    r_det = np.corrcoef(mu[iu], Jh[iu])[0, 1]
    print(f"    B deterministic-drive r = {r_det:.6f}", flush=True)
    put("fix_panel_a_shotnoise", budgets=budgets, mean_r=mean_r, std_r=std_r,
        r_det=r_det, J_learned_iu=mu[iu], J_hebb_iu=Jh[iu], n_rep=REP)

# ---------------------------------------------------------------- C
def block_C():
    if done("fix_panel_d_palimpsest"): print("  [skip] C"); return
    m.RNG = np.random.default_rng(20260625); RNG = m.RNG
    n, L, dt, f_noise = 8, 20, 1.0, 0.0625
    NS, NP = 60, 12
    spins = m.spin_table(n)
    nflip = int(np.floor(f_noise * n + 0.5))
    out = {}
    for tag, (eta, gamma) in (("fade", (0.30, 0.30)), ("nofade", (0.30, 0.001))):
        per = np.zeros((NS, L))
        for st in range(NS):
            P = RNG.choice([-1, 1], size=(L, n))
            J, _ = m.stream_memristor_couplings(P, eta=eta, gamma=gamma, dt=dt,
                                                steps_per_pattern=1)
            for k in range(L):
                age = (L - 1) - k
                tb = np.tile(P[k], (NP, 1))
                nb = corrupt(RNG, tb, nflip, n)
                R = m.retrieve_quantum_anneal_batch(J, nb, Omega0=1.0, spins=spins)
                per[st, age] = np.mean(m.overlap_batch(R, tb) > 0.999)
        out[tag] = per
        print(f"    C {tag}: newest={per.mean(0)[0]:.3f}", flush=True)
    put("fix_panel_d_palimpsest", per_fade=out["fade"], per_nofade=out["nofade"],
        gamma_fade=0.30, gamma_nofade=0.001, n_streams=NS)

# ---------------------------------------------------------------- D
def block_D():
    if done("fix_dense_bootstrap"): print("  [skip] D"); return
    rng = np.random.default_rng(11)
    def descend(S, P, p, nn, sweeps=80):
        mm = S @ P.T
        for _ in range(sweeps):
            moved = np.zeros(S.shape[0], dtype=bool)
            for i in rng.permutation(nn):
                ds = -2.0 * S[:, i]; mnew = mm + np.outer(ds, P[:, i])
                dE = -(np.sum(mnew ** p, axis=1) - np.sum(mm ** p, axis=1))
                fl = dE < -1e-9; S[fl, i] *= -1; mm[fl] = mnew[fl]; moved |= fl
            if not moved.any(): break
        return S
    def succ(P, p, nn, f=0.08, trials=60):
        Lp = P.shape[0]; nflip = max(1, int(round(f * nn)))
        reps = np.repeat(np.arange(Lp), trials); S = P[reps].astype(float).copy()
        fl = np.argsort(rng.random((S.shape[0], nn)), axis=1)[:, :nflip]
        S[np.repeat(np.arange(S.shape[0]), nflip), fl.ravel()] *= -1
        return np.mean(np.mean(descend(S, P, p, nn) == P[reps], axis=1) >= 0.99)
    def cap_raw(nn, p, thr=0.9, reals=40, trials=60):
        o = []
        for _ in range(reals):
            Lc = 0
            for Lx in range(1, nn * nn):
                if succ(rng.choice([-1, 1], size=(Lx, nn)), p, nn, trials=trials) >= thr:
                    Lc = Lx
                else: break
            o.append(Lc)
        return np.array(o, float)
    n2 = np.array([8, 12, 16, 20, 24, 28, 32]); n3 = np.array([8, 11, 14, 17, 20, 23])
    R2 = np.array([cap_raw(x, 2) for x in n2]); print("    D p2 done", flush=True)
    R3 = np.array([cap_raw(x, 3) for x in n3]); print("    D p3 done", flush=True)
    put("fix_dense_bootstrap", n2=n2, n3=n3, R2=R2, R3=R3)

if __name__ == "__main__":
    print("=" * 64); print("AUDIT-FIX COMPUTE LAYER"); print("=" * 64)
    for fn in (block_B, block_D, block_C, block_A):   # cheapest first
        print(f"-- {fn.__name__}", flush=True); fn()
    print("ALL BLOCKS DONE", flush=True)
