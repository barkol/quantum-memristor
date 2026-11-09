#!/usr/bin/env python3
r"""Memristive Hopfield attractor as a quantum-state denoiser, compared with the
qsdl CNN reconstructor on the SAME task (their states, their noise channel,
their Wigner transform, qutip fidelity).

Encoding: the coarse Wigner image is passed through a FIXED random projection
and binarised by its sign.  Physically this is a static scattering medium in
front of the detector array, and it is necessary: the natural threshold code
leaves prototype overlaps at 0.56 (random would be 0.04), which collapses the
Hopfield network onto a single attractor.  The projection brings overlaps to
0.20 and lets n be chosen independently of the image size.

The experiment sweeps the codebook size M at fixed n, which puts the two-body
capacity bound 0.14 n directly in play and tests whether the three-body (dense)
network keeps working past it.
Checkpoint after every M.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, qutip as qt

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

def coarse(W, side):
    g = W.shape[0]; b = g // side
    Wc = W[:side*b, :side*b].reshape(side, b, side, b).mean(axis=(1, 3))
    mx = np.abs(Wc).max()
    return (Wc / (mx if mx > 0 else 1.0)).ravel()

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=16)
    ap.add_argument("--nproj", type=int, default=1024)
    ap.add_argument("--cb-max-per-class", type=int, default=40)
    ap.add_argument("--M-list", type=str, default="35,70,140,280")
    ap.add_argument("--test-per-class", type=int, default=10)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--xmax", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str, default="exp.json")
    ap.add_argument("--noisy-codebook", action="store_true",
                    help="store the association noisy-image -> clean state, "
                         "which is what an associative memory must hold when the "
                         "channel moves states off the clean manifold")
    a = ap.parse_args()
    t0 = time.time(); n = a.nproj
    ck = Path(a.out + ".ckpt")

    # ---- codebook (nested: the first M entries of a shuffled pool) ----------
    rng = np.random.default_rng(a.seed)
    pool_rho, pool_C, pool_lab = [], [], []
    for lab in LABELS:
        for _ in range(a.cb_max_per_class):
            rho, _ = sample_state(lab, CUTOFF, rng)
            if a.noisy_codebook:
                prm = generate_params(rng)
                rn = apply_channel_noise(rho, prm)
                Wp = apply_gauss_noise(apply_wigner(rn, a.grid, a.xmax), prm, rng)
            else:
                Wp = apply_wigner(rho, a.grid, a.xmax)
            pool_rho.append(rho); pool_C.append(coarse(Wp, a.side))
            pool_lab.append(lab)
    order = np.random.default_rng(a.seed + 1).permutation(len(pool_rho))
    pool_rho = [pool_rho[i] for i in order]; pool_C = np.array(pool_C)[order]
    print(f"pool: {len(pool_rho)} prototypow  ({time.time()-t0:.0f}s)", flush=True)

    # ---- test set (their pipeline, with noise) -----------------------------
    rng2 = np.random.default_rng(a.seed + 777)
    clean, noisy, Cq, tlab = [], [], [], []
    for lab in LABELS:
        for _ in range(a.test_per_class):
            rho, _ = sample_state(lab, CUTOFF, rng2)
            prm = generate_params(rng2)
            rn = apply_channel_noise(rho, prm)
            W = apply_gauss_noise(apply_wigner(rn, a.grid, a.xmax), prm, rng2)
            clean.append(rho); noisy.append(rn); Cq.append(coarse(W, a.side)); tlab.append(lab)
    Cq = np.array(Cq); T = len(clean)
    print(f"test: {T} probek  ({time.time()-t0:.0f}s)", flush=True)

    R = np.random.default_rng(a.seed + 2).normal(size=(Cq.shape[1], n)) / np.sqrt(Cq.shape[1])
    def bin_(C):
        S = np.sign(C @ R); S[S == 0] = 1.0; return S
    Spool_all = bin_(pool_C); S0 = bin_(Cq)

    fid = lambda r, s: float(qt.fidelity(r, s))
    F_id = np.array([fid(clean[i], noisy[i]) for i in range(T)])
    res = {"config": vars(a), "n": n, "T": T,
           "identity": [float(F_id.mean()), float(1.96*F_id.std(ddof=1)/np.sqrt(T))],
           "by_M": {}}
    if ck.exists(): res = json.loads(ck.read_text())
    print(f"identity = {res['identity'][0]:.4f}", flush=True)

    for M in [int(x) for x in a.M_list.split(",")]:
        if str(M) in res["by_M"]: print(f"  [skip] M={M}"); continue
        P = Spool_all[:M]; rc = pool_rho[:M]
        out = {}
        d2 = ((Cq[:, None, :] - pool_C[:M][None])**2).sum(axis=2)
        idx = {"template-real": np.argmin(d2, axis=1),
               "template-bin": np.argmax(S0 @ P.T, axis=1)}
        for p in (2, 3):
            S = descend(S0.astype(float).copy(), P.astype(float), p,
                        rng=np.random.default_rng(a.seed + p))
            idx[f"hopfield-{p}"] = np.argmax(S @ P.T, axis=1)
        for k, v in idx.items():
            Fv = np.array([fid(clean[i], rc[v[i]]) for i in range(T)])
            out[k] = [float(Fv.mean()), float(1.96*Fv.std(ddof=1)/np.sqrt(T))]
        Fo = np.array([max(fid(clean[i], rc[m]) for m in range(M)) for i in range(T)])
        out["oracle"] = [float(Fo.mean()), float(1.96*Fo.std(ddof=1)/np.sqrt(T))]
        out["capacity_2body_0.14n"] = 0.14*n
        res["by_M"][str(M)] = out
        ck.write_text(json.dumps(res, indent=1))
        print(f"M={M:4d} (0.14n={0.14*n:.0f}): " +
              "  ".join(f"{k}={v[0]:.4f}" for k, v in out.items() if isinstance(v, list)), flush=True)

    Path(a.out).write_text(json.dumps(res, indent=1)); ck.unlink(missing_ok=True)
    print(f"\ngotowe w {time.time()-t0:.0f}s -> {a.out}")

if __name__ == "__main__":
    main()
