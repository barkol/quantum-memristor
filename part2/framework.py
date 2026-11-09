#!/usr/bin/env python3
r"""Can a memristive Hopfield attractor denoise optical quantum states better
than the qsdl CNN reconstructor?

Common framework, deliberately built on THEIR code so the numbers are
comparable: their state sampler, their noise channel, their Wigner transform,
their fidelity convention (qutip's Tr sqrt(sqrt(rho) sigma sqrt(rho))).

The two families being compared are structurally different and that is the
point:
  * the CNN is a REGRESSION denoiser  -- maps a noisy Wigner image to a clean
    one, can interpolate, but can emit a non-physical output;
  * the Hopfield network is a CODEBOOK/ATTRACTOR denoiser -- snaps the input to
    one of M stored prototypes, so its output is always a legitimate state, but
    its fidelity is capped by how finely the codebook covers state space.

Methods compared
  identity    : do nothing, return the state after the physical channel
  template    : nearest codebook entry by Wigner L2 (classical template match)
  hopfield-2  : two-body Hebbian attractor, zero-T descent
  hopfield-3  : three-body (dense) attractor, zero-T descent
  oracle      : best codebook entry by fidelity -- the ceiling for ANY codebook
                method, not achievable in practice
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import qutip as qt

# The qsdl package (state sampler, Wigner transform, noise channels) is an
# EXTERNAL dependency: https://github.com/Askartan/qsdl  (by F. Grzywa).
# Clone it next to this repository, or point QSDL_PATH at your checkout.
import os
QSDL = Path(os.environ.get(
    "QSDL_PATH", Path(__file__).resolve().parents[2] / "qsdl"))
if not (QSDL / "qsdl").is_dir():
    raise SystemExit(
        f"qsdl not found at {QSDL}. Clone https://github.com/Askartan/qsdl "
        "there, or set QSDL_PATH to your checkout.")
sys.path.insert(0, str(QSDL))
from qsdl.states import sample_state
from qsdl.wigner import apply_wigner
from qsdl.noise import apply_channel_noise, apply_gauss_noise, generate_params
from qsdl.labels import LABELS
from qsdl.config import CUTOFF

# ----------------------------------------------------------------- encoding
def coarse(W, side):
    """Coarse-grain the Wigner image to side x side and normalise by its peak."""
    g = W.shape[0]; b = g // side
    Wc = W[:side*b, :side*b].reshape(side, b, side, b).mean(axis=(1, 3))
    mx = np.abs(Wc).max()
    return Wc / (mx if mx > 0 else 1.0)

THRESH = (-0.15, 0.02, 0.20)

def encode(W, side):
    """Wigner image -> +-1 spins via a thermometer code on the normalised
    coarse image.  A single sign bit is nearly useless here (classical states
    have W >= 0 everywhere), so we use several levels, which keeps both the
    negativity and the coarse amplitude profile.  n = len(THRESH) * side^2."""
    v = coarse(W, side).ravel()
    bits = [np.where(v > th, 1.0, -1.0) for th in THRESH]
    return np.concatenate(bits)

# ----------------------------------------------------------------- retrieval
def descend(S, P, p, sweeps=60, rng=None):
    """Zero-temperature greedy descent on the p-body Hopfield energy
    E = -sum_mu (S . xi^mu)^p, batched over rows of S."""
    rng = rng or np.random.default_rng(0)
    n = S.shape[1]
    m = S @ P.T
    for _ in range(sweeps):
        moved = np.zeros(S.shape[0], dtype=bool)
        for i in rng.permutation(n):
            ds = -2.0 * S[:, i]
            mnew = m + np.outer(ds, P[:, i])
            dE = -(np.sum(mnew**p, axis=1) - np.sum(m**p, axis=1))
            fl = dE < -1e-9
            S[fl, i] *= -1
            m[fl] = mnew[fl]
            moved |= fl
        if not moved.any():
            break
    return S

def to_prototype(S, P):
    """Map each retrieved configuration to the codebook entry it is closest to."""
    return np.argmax(S @ P.T, axis=1)

# ----------------------------------------------------------------- data
def build_codebook(per_class, side, seed, grid, xmax):
    rng = np.random.default_rng(seed)
    rhos, Ws, labs = [], [], []
    for lab in LABELS:
        for _ in range(per_class):
            rho, _ = sample_state(lab, CUTOFF, rng)
            rhos.append(rho)
            Ws.append(apply_wigner(rho, grid, xmax))
            labs.append(lab)
    P = np.array([encode(W, side) for W in Ws])
    C = np.array([coarse(W, side).ravel() for W in Ws])
    return rhos, P, C, labs

def build_testset(per_class, side, seed, grid, xmax):
    rng = np.random.default_rng(seed)
    clean, noisy_rho, Wn, labs = [], [], [], []
    for lab in LABELS:
        for _ in range(per_class):
            rho, _ = sample_state(lab, CUTOFF, rng)
            prm = generate_params(rng)
            rho_n = apply_channel_noise(rho, prm)
            W = apply_wigner(rho_n, grid, xmax)
            W = apply_gauss_noise(W, prm, rng)
            clean.append(rho); noisy_rho.append(rho_n); Wn.append(W); labs.append(lab)
    S = np.array([encode(W, side) for W in Wn])
    Cq = np.array([coarse(W, side).ravel() for W in Wn])
    return clean, noisy_rho, S, Cq, labs

# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=16)       # n = side^2 spins
    ap.add_argument("--codebook-per-class", type=int, default=8)
    ap.add_argument("--test-per-class", type=int, default=6)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--xmax", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str, default="result.json")
    a = ap.parse_args()
    n = len(THRESH) * a.side * a.side
    t0 = time.time()

    rho_cb, P, Ccb, cb_labs = build_codebook(a.codebook_per_class, a.side, a.seed, a.grid, a.xmax)
    M = len(rho_cb)
    print(f"codebook: M={M} prototypes, n={n} spins, 0.14n={0.14*n:.0f}", flush=True)
    clean, noisy, S0, Cq, labs = build_testset(a.test_per_class, a.side, a.seed + 777, a.grid, a.xmax)
    T = len(clean)
    print(f"test set: {T} samples   (setup {time.time()-t0:.1f}s)", flush=True)

    # --- retrievals
    res_idx = {}
    # (a) full-information nearest template: L2 on the real-valued coarse image
    d2 = ((Cq[:, None, :] - Ccb[None, :, :])**2).sum(axis=2)
    res_idx["template-real"] = np.argmin(d2, axis=1)
    # (b) same, but after binarisation -- isolates the cost of the +-1 code
    res_idx["template-bin"] = np.argmax(S0 @ P.T, axis=1)
    for p in (2, 3):
        S = S0.astype(float).copy()
        S = descend(S, P.astype(float), p, rng=np.random.default_rng(a.seed + p))
        res_idx[f"hopfield-{p}"] = to_prototype(S, P.astype(float))
    print(f"retrievals done ({time.time()-t0:.1f}s)", flush=True)

    # --- fidelities
    def fid(r, s):
        return float(qt.fidelity(r, s))
    out = {"config": vars(a), "n_spins": n, "M": M, "T": T, "per_method": {}, "per_class": {}}
    F = {}
    F["identity"] = np.array([fid(clean[i], noisy[i]) for i in range(T)])
    for meth, idx in res_idx.items():
        F[meth] = np.array([fid(clean[i], rho_cb[idx[i]]) for i in range(T)])
    # oracle ceiling for codebook methods
    F["oracle"] = np.array([max(fid(clean[i], rho_cb[m]) for m in range(M)) for i in range(T)])
    print(f"fidelities done ({time.time()-t0:.1f}s)", flush=True)

    for meth, v in F.items():
        out["per_method"][meth] = [float(v.mean()), float(1.96*v.std(ddof=1)/np.sqrt(len(v)))]
    labs = np.array(labs)
    for lab in LABELS:
        msk = labs == lab
        out["per_class"][lab] = {m: float(v[msk].mean()) for m, v in F.items()}
    Path(a.out).write_text(json.dumps(out, indent=1))

    print("\n=== SREDNIA FIDELITY (qutip, ten sam konwent co audyt) ===")
    for m in ("identity", "template-real", "template-bin", "hopfield-2", "hopfield-3", "oracle"):
        mu, ci = out["per_method"][m]
        print(f"  {m:>12}: {mu:.4f} +- {ci:.4f}")
    print(f"\n  [odniesienie z audytu] siec Grzywy 0.9309 | nic-nie-rob 0.8325 | srednia klasy 0.5835")
    print(f"\ncalosc {time.time()-t0:.1f}s -> {a.out}")

if __name__ == "__main__":
    main()
