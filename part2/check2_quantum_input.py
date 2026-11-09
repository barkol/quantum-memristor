#!/usr/bin/env python3
"""check2 — teza: REMIS na wejsciu klasycznym, WYGRANA na wejsciu kwantowym.

Zadanie klasyczne: uszkodzony wzorzec |xi~> jako stan rejestru -> odzyskac |xi>.
Zadanie kwantowe : SUPERPOZYCJA (|xi~_a>+|xi~_b>)/r2 -> odzyskac (|xi_a>+|xi_b>)/r2.

Sufit klasyczny na zadaniu kwantowym jest twardy: dowolna pojedyncza
konfiguracja ma z rowna superpozycja dwoch ortogonalnych stanow fidelity 1/r2.
Odczyt klasyczny (argmax w bazie obliczeniowej) zwraca jedna konfiguracje,
wiec nie moze przekroczyc 1/2. Ewolucja unitarna moze, bo jest liniowa.
"""
import numpy as np, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_qhnm as m

n, L = 8, 3
m.RNG = np.random.default_rng(4242)
P = m.RNG.choice([-1, 1], size=(L, n))
J, _ = m.stream_memristor_couplings(P, eta=1.0/n, gamma=0.01, dt=1.0, steps_per_pattern=1)
spins = m.spin_table(n); dim = 2**n
idx = lambda s: int("".join(map(str, ((1-np.asarray(s))//2).astype(int))), 2)

def evolve(psi0, Om0, T, K):
    """Ewolucja unitarna: H(s) = -(1-s) Om0 sum X + s H_J.  Wejscie = STAN."""
    Ed = m.classical_energy(J, np.zeros(n), spins)
    psi = psi0.astype(complex).copy(); dt = T/K
    for k in range(K):
        s = (k+0.5)/K; Ax = (1-s)*Om0
        psi = m._apply_x_rotations(psi, n, -Ax*dt/2.0)
        psi *= np.exp(-1j*s*Ed*dt)
        psi = m._apply_x_rotations(psi, n, -Ax*dt/2.0)
    return psi

rg = np.random.default_rng(7)
def corrupt(x, k=1):
    y = x.copy(); y[rg.choice(n, k, replace=False)] *= -1; return y
xa, xb = P[0], P[1]
ta, tb = corrupt(xa), corrupt(xb)
ea = np.zeros(dim, complex); ea[idx(ta)] = 1
eb = np.zeros(dim, complex); eb[idx(tb)] = 1
tgt_sup = (np.zeros(dim, complex)); tgt_sup[idx(xa)] += 1; tgt_sup[idx(xb)] += 1
tgt_sup /= np.linalg.norm(tgt_sup)

print("Skan (Om0, T): odzysk klasyczny i fidelity na superpozycji")
print(f"{'Om0':>6} {'T':>6} | {'P(xi_a) z |xi~_a>':>17} | {'F(sup)':>8}  {'sufit klas.':>11}")
best = None
for Om0 in (0.1, 0.2, 0.35, 0.5, 0.8):
    for T in (2, 5, 10, 20):
        K = max(200, int(40*T))
        fa = evolve(ea, Om0, T, K)
        p_a = abs(fa[idx(xa)])**2 + abs(fa[idx(-xa)])**2
        fs = evolve((ea+eb)/np.sqrt(2), Om0, T, K)
        F = abs(np.vdot(tgt_sup, fs))**2
        if best is None or F > best[0]: best = (F, Om0, T, p_a)
        print(f"{Om0:>6.2f} {T:>6} | {p_a:>17.4f} | {F:>8.4f}  {0.5:>11.4f}")
print(f"\nnajlepsze: F(sup)={best[0]:.4f} przy Om0={best[1]}, T={best[2]} (P(xi_a)={best[3]:.4f})")
print(f"sufit dowolnej pojedynczej konfiguracji na superpozycji: |<sup|xi>|^2 = 0.5000")
Path("check2_wynik.json").write_text(json.dumps({"best_F":best[0],"Om0":best[1],"T":best[2]}, indent=1))
