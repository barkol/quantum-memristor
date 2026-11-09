#!/usr/bin/env python3
r"""check4 — struktura KWANTOWEJ sieci Hopfielda: kod liniowy + dekoder syndromowy.

Trzy rzeczy do rozstrzygniecia przed napisaniem sekcji:
 (1) czy dla kodu LINIOWEGO syndrom naprawde zalezy TYLKO od bledu (twierdzenie,
     nie statystyka) - sprawdzamy Hx dla RM(1,m) na wielu parach slow i bledach;
 (2) jaka jest pojemnosc i tolerancja bledow takiej pamieci przy n = 16..64;
 (3) koszt zasobowy: ile bitow syndromu i ile CNOT-ow na ekstrakcje.
"""
import numpy as np, json, itertools
from pathlib import Path

def rm1_generator(m):
    """RM(1,m): generator (m+1) x 2^m nad GF(2). Wiersze: 1 oraz m zmiennych."""
    n = 2**m
    rows = [np.ones(n, dtype=np.uint8)]
    for i in range(m):
        rows.append(np.array([(x >> i) & 1 for x in range(n)], dtype=np.uint8))
    return np.array(rows)            # (m+1, n)

def code_words(G):
    k, n = G.shape
    C = []
    for a in itertools.product([0,1], repeat=k):
        C.append(np.mod(np.array(a) @ G, 2).astype(np.uint8))
    return np.array(C)               # (2^k, n)

def parity_check(G):
    """H = baza jadra G^T nad GF(2), metoda eliminacji Gaussa."""
    k, n = G.shape
    A = G.copy() % 2
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, k) if A[i, c]), None)
        if p is None: continue
        A[[r, p]] = A[[p, r]]
        for i in range(k):
            if i != r and A[i, c]: A[i] ^= A[r]
        piv.append(c); r += 1
        if r == k: break
    free = [c for c in range(n) if c not in piv]
    H = np.zeros((len(free), n), dtype=np.uint8)
    for j, f in enumerate(free):
        H[j, f] = 1
        for i, c in enumerate(piv): H[j, c] = A[i, f]
    return H

out = {}
print(f"{'n':>5} {'k':>3} {'slow':>6} {'d_min':>6} {'t':>4} {'bity syndromu':>14} {'CNOT-ow':>9} {'syndrom=f(blad)?':>17}")
for m in (4, 5, 6):
    n = 2**m
    G = rm1_generator(m); C = code_words(G); H = parity_check(G)
    # d_min
    w = C[1:].sum(axis=1)
    d = int(w.min())
    t = (d-1)//2
    # (1) czy Hx zalezy tylko od bledu? sprawdzamy dla losowych par slow i bledow
    rg = np.random.default_rng(0); ok = True; tested = 0
    for _ in range(400):
        i, j = rg.integers(0, len(C), 2)
        e = np.zeros(n, dtype=np.uint8)
        e[rg.choice(n, rg.integers(1, max(2, t+1)), replace=False)] = 1
        s1 = (H @ ((C[i] ^ e) % 2)) % 2
        s2 = (H @ ((C[j] ^ e) % 2)) % 2
        se = (H @ e) % 2
        if not (np.array_equal(s1, s2) and np.array_equal(s1, se)): ok = False
        tested += 1
    cnots = int(H.sum())              # kazda jedynka w H = jeden CNOT na ancille
    out[f"n{n}"] = dict(n=n, k=int(G.shape[0]), words=int(len(C)), d=d, t=int(t),
                        synd_bits=int(H.shape[0]), cnots=cnots, syndrome_data_independent=ok)
    print(f"{n:>5} {G.shape[0]:>3} {len(C):>6} {d:>6} {t:>4} {H.shape[0]:>14} {cnots:>9} "
          f"{'TAK ('+str(tested)+' prob)' if ok else 'NIE':>17}")

print("\nPojemnosc: liczba slow kodowych wobec granicy Hebba 0.14n i sufitu ortogonalnosci n-1")
for m in (4,5,6):
    n=2**m; r=out[f"n{n}"]
    print(f"  n={n:3d}: slow kodowych {r['words']:5d}  |  0.14n = {0.14*n:5.1f}  |  n-1 = {n-1:4d}"
          f"  |  tolerancja {r['t']}/{n} = {r['t']/n:.0%} dlugosci slowa")
Path("check4_wynik.json").write_text(json.dumps(out, indent=1))
print("\nzapisano check4_wynik.json")
