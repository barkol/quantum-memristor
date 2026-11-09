"""Sensitivity of the unipolar response rate at jump 0.05 to protocol; X and Y RMS against horizon. Pool(2)."""
import numpy as np, multiprocessing as mp
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from own_extra import signed_rec, mf_basis, chi, OM
def unip(a):
    D, settle, off, seed, gdt, crn = a
    A, gam, th0 = 50, .01, np.pi/16; G = A*gam/D; dt = gdt/G; N = 100000
    thmf = brentq(lambda t: th0+A*np.cos(2*t)**2-t, th0, np.pi/4)
    rng = np.random.default_rng([seed, int(settle), int(off*1000), crn]); th = np.full(N, thmf); rel = np.exp(-gam*dt)
    for _ in range(int(settle/dt)):
        hit = rng.random(N) < G*dt; th = th0+(th+hit*(rng.random(N) < np.cos(2*th)**2)*D-th0)*rel
    x = th.copy(); y = th+off; R = []; T = []
    for s in range(int(10/dt)):
        hit = rng.random(N) < G*dt; u = rng.random(N)
        hit2 = hit if crn else rng.random(N) < G*dt; v = u if crn else rng.random(N)
        x = th0+(x+hit*(u < np.cos(2*x)**2)*D-th0)*rel; y = th0+(y+hit2*(v < np.cos(2*y)**2)*D-th0)*rel
        R.append((y-x).mean()); T.append((s+1)*dt)
    R = np.array(R); T = np.array(T); o = {}
    for w in (2., 5., 10.):
        m = (T <= w) & (R > 0); o[w] = round(float(-np.polyfit(T[m], np.log(R[m]), 1)[0]), 3)
    return dict(settle=settle, off=off, seed=seed, crn=crn, rates=o, out=float((th > np.pi/4).mean()))
def xy(a):
    b, D, nc = a
    t, m, s, l, lf, O = signed_rec(D, N=20000 if D > .01 else 6000, T=2*np.pi/OM*nc, basis=b, th_start=0.)
    if b == "X": return dict(b=b, D=D, nc=nc, rms=float(np.sqrt(np.mean((m-mf_basis(t, "X", 0.))**2))))
    return dict(b=b, D=D, nc=nc, rms=float(np.sqrt(np.mean((O-np.cos(2*chi(t)))**2))))
if __name__ == "__main__":
    with mp.Pool(2) as P:
        for r in P.imap(unip, [(.05, 30., .005, s, .1, 1) for s in (1, 2)]+[(.05, 30., .02, s, .1, 1) for s in (1, 2)]+[(.05, 100., .02, 1, .1, 1), (.05, 30., .05, 1, .1, 1), (.05, 100., .1, 1, .1, 0), (.05, 30., .1, 2, .1, 0)]):
            print(r, flush=True)
        for r in P.imap(xy, [("X", .025, 5), ("Y", .025, 5), ("X", .025, 1), ("Y", .025, 1), ("Y", .0025, 5)]):
            print(r, flush=True)
