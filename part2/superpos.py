#!/usr/bin/env python3
r"""Co robia te metody z KWANTOWA SUPERPOZYCJA stanow, ktorych funkcje Wignera
byly wzorcami przy uczeniu?

W jest liniowa w rho, wiec mieszanina rozklada sie na wzorce, a superpozycja nie:
  |psi> = c_a|a> + c_b|b>  ->  W = |c_a|^2 W_a + |c_b|^2 W_b + 2Re(c_a c_b^* W_ab)
Czlon skrosny W_ab lezy POZA powloka liniowa wzorcow i to on niesie koherencje.

Test dwustronny: ta sama para skladnikow, raz jako superpozycja, raz jako
mieszanina. Metoda widzi koherencje tylko wtedy, gdy zwraca dla nich ROZNE
odpowiedzi. Zwrocenie tej samej odpowiedzi = slepota na koherencje.

Dwa przypadki:
  (A) superpozycja dwoch stanow koherentnych o przeciwnej fazie = stan kota,
      ktory JEST w ich zbiorze klas -> ksiazka kodowa moze sobie poradzic;
  (B) superpozycja dwoch stanow Focka -> klasy takiej u nich NIE MA.
"""
import sys, numpy as np, qutip as qt
from pathlib import Path
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
from experiment import coarse, descend

side, grid, xmax, n = 16, 64, 5.0, 2048
M_PER = 20

# --- codebook (associative: noisy image -> clean state) --------------------
rng = np.random.default_rng(2026)
cb_rho, cb_C = [], []
for lab in LABELS:
    for _ in range(M_PER):
        r, _ = sample_state(lab, CUTOFF, rng); prm = generate_params(rng)
        Wp = apply_gauss_noise(apply_wigner(apply_channel_noise(r, prm), grid, xmax), prm, rng)
        cb_rho.append(r); cb_C.append(coarse(Wp, side))
cb_C = np.array(cb_C); M = len(cb_rho)
R = np.random.default_rng(2028).normal(size=(cb_C.shape[1], n)) / np.sqrt(cb_C.shape[1])
bs = lambda C: np.where(np.sign(C @ R) == 0, 1.0, np.sign(C @ R))
P = bs(cb_C)
print(f"ksiazka kodowa M={M}, n={n} spinow\n")

def pipeline(rho, rng):
    prm = generate_params(rng)
    rn = apply_channel_noise(rho, prm)
    W = apply_gauss_noise(apply_wigner(rn, grid, xmax), prm, rng)
    return rn, coarse(W, side)

def run_methods(Cq):
    S0 = bs(Cq[None])
    out = {}
    out["template"] = int(np.argmin(((Cq[None] - cb_C)**2).sum(axis=1)))
    for p in (2, 3):
        S = descend(S0.astype(float).copy(), P.astype(float), p, rng=np.random.default_rng(5))
        out[f"hopfield-{p}"] = int(np.argmax(S @ P.T))
    return out

F = lambda a, b: float(qt.fidelity(a, b))
CASES = []
al = 2.0
CASES.append(("A: |a>+|-a>  (= stan kota, klasa JEST w zbiorze)",
              qt.coherent(CUTOFF, al), qt.coherent(CUTOFF, -al)))
CASES.append(("B: |2>+|5>   (takiej klasy NIE MA w zbiorze)",
              qt.basis(CUTOFF, 2), qt.basis(CUTOFF, 5)))
CASES.append(("C: |0>+|3>   (takiej klasy NIE MA w zbiorze)",
              qt.basis(CUTOFF, 0), qt.basis(CUTOFF, 3)))

NREP = 12
for name, ka, kb in CASES:
    psi = (ka + kb).unit()
    rho_sup = qt.ket2dm(psi); rho_mix = 0.5*(qt.ket2dm(ka) + qt.ket2dm(kb))
    print("="*78); print(name)
    print(f"  F(superpozycja, mieszanina) = {F(rho_sup, rho_mix):.4f}")
    orc_s = max(F(rho_sup, cb_rho[m]) for m in range(M))
    orc_m = max(F(rho_mix, cb_rho[m]) for m in range(M))
    print(f"  oracle: superpozycja {orc_s:.4f}   mieszanina {orc_m:.4f}"
          f"   (dla rownej superpozycji dwoch ortogonalnych pojedynczy stan daje <= 0.7071)")
    acc = {k: {"sup": [], "mix": []} for k in ("identity","template","hopfield-2","hopfield-3")}
    blind = {k: 0 for k in ("template","hopfield-2","hopfield-3")}
    coh, noi = [], []
    for rep in range(NREP):
        seed = 1000 + 17*rep
        rn_s, Cs = pipeline(rho_sup, np.random.default_rng(seed))
        rn_m, Cm = pipeline(rho_mix, np.random.default_rng(seed))   # ten sam szum
        _,    Cs2 = pipeline(rho_sup, np.random.default_rng(seed+9999))
        coh.append(np.linalg.norm(Cs-Cm)); noi.append(np.linalg.norm(Cs-Cs2))
        acc["identity"]["sup"].append(F(rho_sup, rn_s)); acc["identity"]["mix"].append(F(rho_mix, rn_m))
        i_s = run_methods(Cs); i_m = run_methods(Cm)
        for k in i_s:
            acc[k]["sup"].append(F(rho_sup, cb_rho[i_s[k]]))
            acc[k]["mix"].append(F(rho_mix, cb_rho[i_m[k]]))
            blind[k] += int(i_s[k] == i_m[k])
    print(f"  sygnal koherencji |W_sup-W_mix| = {np.mean(coh):.4f}  wobec samego szumu {np.mean(noi):.4f}"
          f"  -> stosunek {np.mean(coh)/np.mean(noi):.2f}")
    print(f"  {'metoda':>12}  {'F(sup)':>8}  {'F(mix)':>8}   ten sam wzorzec dla sup i mix")
    for k in ("identity","template","hopfield-2","hopfield-3"):
        b = f"{blind[k]}/{NREP}" if k in blind else "-"
        print(f"  {k:>12}  {np.mean(acc[k]['sup']):>8.4f}  {np.mean(acc[k]['mix']):>8.4f}   {b:>8}")
