"""Last own checks. Lobe area for the figure-eight, same-basis index, A_X and A_Z, Euler-RK45 difference, RMS against N and jump, exact cells at low frequency."""
import numpy as np, multiprocessing as mp, json
from scipy.integrate import solve_ivp
from own_exact import exact, mf
ETA, GAM, OM = .5, .01, .12
chi = lambda t: .5*np.sin(OM*t)
def lobes(eta, gam, om, th0=np.pi/8, nc=10):
    per = 2*np.pi/om
    s = solve_ivp(lambda t, y: eta*np.sin(om*t)-gam*(y-th0), [0, nc*per], [th0], rtol=1e-9, atol=1e-11, dense_output=True)
    t = np.linspace((nc-1)*per, nc*per, 20001); th = s.sol(t)[0]; v = np.sin(om*t); i = np.sin(2*th)**2*v
    return float(np.sum(np.abs(.5*(i[1:]+i[:-1])*np.diff(v)))*0+abs(np.trapezoid(i[:10001], v[:10001]))+abs(np.trapezoid(i[10000:], v[10000:]))), float(th.max()-th.min())
def bas(sig, read):
    T = 5*2*np.pi/OM
    f = {"Z": lambda t, y: ETA*np.cos(4*y-2*chi(t))-GAM*y, "X": lambda t, y: ETA*np.sin(4*y-2*chi(t))-GAM*y}[sig]
    s = solve_ivp(f, [0, T], [0.], rtol=1e-8, atol=1e-10, dense_output=True)
    t = np.linspace(T-2*np.pi/OM, T, 20001)[:-1]; th = s.sol(t)[0]; ph = 4*th-2*chi(t)
    x = np.sin(2*chi(t)); y = {"Z": np.cos(ph), "X": np.sin(ph)}[read]
    return float(abs(.5*np.sum(x*np.roll(y, -1)-y*np.roll(x, -1))))
def clock(a):
    D, N, seed = a; rng = np.random.default_rng([seed, int(D*1e4), N]); dt = D/ETA; T = 3*2*np.pi/OM; n = int(T/dt)
    th = np.zeros(N); e = 0.; res = np.zeros(n+1); eu = np.zeros(n+1)
    for i in range(n):
        ph = 4*th-2*chi(i*dt)
        th = th+np.where(rng.random(N) < (1+np.cos(ph))/2, 1., -1.)*ETA*dt-GAM*th*dt
        e = e+dt*(ETA*np.cos(4*e-2*chi(i*dt))-GAM*e)
        res[i+1] = th.mean(); eu[i+1] = e
    t = np.arange(n+1)*dt
    rk = solve_ivp(lambda t, y: ETA*np.cos(4*y-2*chi(t))-GAM*y, [0, t[-1]], [0.], t_eval=t, rtol=1e-10, atol=1e-12).y[0]
    return dict(D=D, N=N, rms=float(np.sqrt(np.mean((res-rk)**2))), euler_rk=float(np.sqrt(np.mean((eu-rk)**2))), noise_vs_euler=float(np.sqrt(np.mean((res-eu)**2))))
if __name__ == "__main__":
    print("omega lobes", {o: lobes(.15, .1, o) for o in (.1, .15, .2, .25, .3, .35, .4, .5, 1., 3.)})
    print("eta lobes", {e: lobes(e, .1, .3) for e in (.05, .1, .12, .15, .18, .2, .3)})
    print("gamma lobes", {g: lobes(.15, g, .3) for g in (.03, .1, .3, 1.)})
    a = {(s, r): bas(s, r) for s, r in (("Z", "Z"), ("X", "Z"), ("X", "X"))}; print("areas", a, "M_XZ", abs(a["X", "Z"]-a["Z", "Z"])/(a["X", "Z"]+a["Z", "Z"]), "M same-basis", abs(a["X", "X"]-a["Z", "Z"])/(a["X", "X"]+a["Z", "Z"]), "r", np.pi*GAM/(4*ETA), "(1-r)/(1+r)", (1-np.pi*GAM/(4*ETA))/(1+np.pi*GAM/(4*ETA)))
    with mp.Pool(4) as P:
        for x in P.map(clock, [(.025, 200000, 1), (.025, 20000, 1), (.0125, 20000, 1), (.00625, 20000, 1)]): print(x, flush=True)
        for om, A, D in [(.12, .3, .025), (.12, .3, .1), (2., .3, .25), (2.01, .3, .1)]:
            r = np.array(P.map(exact, [(om, A, D, s, 20000) for s in range(4)])); m = mf(om, A); print("exact", om, A, D, r.mean()/m, r.std(ddof=1)/2/m, flush=True)
