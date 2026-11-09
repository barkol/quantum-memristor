#!/usr/bin/env python3
r"""Coherent decoder on RM(1,4), quoted in Part II, Sec. "Coherent recall": all 32 words of an n = 16 register, a random pair of words, a
random relative phase, and a bit-flip operator of weight 1 to 3 acting on both branches. The recall |x>|0> -> |D(x)>|x xor D(x)> is
applied on the two-branch state, the ancilla is traced out, and the fidelity with the target superposition is recorded. Controls: two
different errors, and a single phase flip on a coordinate where the two words differ. Also prints the RM(1,6) syndrome count.
Deterministic (seed 20261101). Output: printed; a copy is stored in results/check5_rm_decoder.txt."""
import itertools
import numpy as np


def rm1(m):
    n = 2**m
    G = np.array([[1]*n] + [[(x >> i) & 1 for x in range(n)] for i in range(m)], dtype=np.uint8)
    C = np.array([np.mod(np.array(a) @ G, 2) for a in itertools.product([0, 1], repeat=m+1)], dtype=np.uint8)
    return G, C


def nullspace(G):
    """A basis of the parity checks in the sparse form of the Hadamard (extended Hamming dual) construction."""
    A = G.copy() % 2; r, piv = 0, []
    for c in range(A.shape[1]):
        p = [i for i in range(r, A.shape[0]) if A[i, c]]
        if not p: continue
        A[[r, p[0]]] = A[[p[0], r]]
        for i in range(A.shape[0]):
            if i != r and A[i, c]: A[i] ^= A[r]
        piv.append(c); r += 1
    free = [c for c in range(A.shape[1]) if c not in piv]
    H = np.zeros((len(free), A.shape[1]), dtype=np.uint8)
    for k, c in enumerate(free):
        H[k, c] = 1
        for i, q in enumerate(piv): H[k, q] = A[i, c]
    return H


def decode(C, x):
    return C[np.argmin((C ^ x).sum(1))]


def recall_fidelity(C, c1, c2, phi, e1, e2, sign2=1.0):
    """F = sum_anc |<t|v_anc>|^2 for the two-branch input (|c1+e1> + sign2 e^{i phi}|c2+e2>)/sqrt 2 and t = (|c1> + e^{i phi}|c2>)/sqrt 2."""
    out = {}
    for x, a in (((c1 ^ e1), 1/np.sqrt(2)), ((c2 ^ e2), sign2*np.exp(1j*phi)/np.sqrt(2))):
        d = decode(C, x); k = (d.tobytes(), (x ^ d).tobytes()); out[k] = out.get(k, 0) + a
    F = {}
    for (d, anc), a in out.items():
        w = 1/np.sqrt(2) if d == c1.tobytes() else (np.exp(-1j*phi)/np.sqrt(2) if d == c2.tobytes() else 0.0)
        F[anc] = F.get(anc, 0) + w*a
    return float(sum(abs(v)**2 for v in F.values()))


def main():
    rng = np.random.default_rng(20261101); G, C = rm1(4); H = nullspace(G); n = C.shape[1]; N = 500
    print("RM(1,4): words", len(C), "n", n, "min distance", min(int((a ^ b).sum()) for a, b in itertools.combinations(C, 2)), "H", H.shape)
    def err(w):
        e = np.zeros(n, dtype=np.uint8); e[rng.choice(n, w, replace=False)] = 1; return e
    Fs, Fz, Fd, ok_b, ok_s = [], [], [], 0, 0
    for _ in range(N):
        i, j = rng.choice(len(C), 2, replace=False); c1, c2 = C[i], C[j]; phi = rng.uniform(0, 2*np.pi)
        e = err(rng.integers(1, 4)); Fs.append(recall_fidelity(C, c1, c2, phi, e, e))
        ok_b += (decode(C, c1 ^ e) == c1).all() and (decode(C, c2 ^ e) == c2).all()
        ok_s += ((H @ (c1 ^ e)) % 2 == (H @ (c2 ^ e)) % 2).all()
        e2 = err(rng.integers(1, 4))
        while (e2 == e).all(): e2 = err(rng.integers(1, 4))
        Fd.append(recall_fidelity(C, c1, c2, phi, e, e2))
    print(f"same bit-flip error, {N} draws: branches restored {ok_b/N:.3f}, syndrome equal {ok_s/N:.3f}, fidelity min {min(Fs):.6f} mean {np.mean(Fs):.6f}")
    print(f"two different errors, {N} draws: fidelity mean {np.mean(Fd):.4f}, min {min(Fd):.6f}, max {max(Fd):.6f}")
    i, j = 0, 1; d = np.where(C[i] != C[j])[0]; z = np.zeros(n, dtype=np.uint8)
    print("one phase flip where the words differ: fidelity", round(recall_fidelity(C, C[i], C[j], 0.3, z, z, sign2=-1.0), 6))
    G6, _ = rm1(6); H6 = nullspace(G6)
    print(f"RM(1,6): n 64, words 128, distance 32, t 15, ancillas {H6.shape[0]}, nonzero entries of H {int(H6.sum())}")


if __name__ == "__main__":
    main()
