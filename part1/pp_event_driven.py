"""Event-driven exact simulation of the signed point process: no time step in the dynamics.
Between Poisson events theta relaxes exactly; outcome probability is evaluated at the event time; sampled on a grid.
Also repeats the repo's discretized scheme at G*dt=0.1 and finer, for a step-size convergence series. Pool(4). Checkpointed."""
import numpy as np, json, os, sys, multiprocessing as mp
from scipy.integrate import solve_ivp
ETA, GAM = .5, .01
def shoe(x, y): return abs(.5*np.sum(x*np.roll(y, -1)-y*np.roll(x, -1)))
def mf(om, A):
    per = 2*np.pi/om; c = lambda t: .5*A*np.sin(om*t); T = 40+15*per
    s = solve_ivp(lambda t, y: ETA*np.cos(4*y-2*c(t))-GAM*y, [0, T+per], [np.pi/8], rtol=1e-10, atol=1e-12, dense_output=True, method="DOP853")
    t = np.linspace(T, T+per, 8001)[:-1]; th = s.sol(t)[0]
    return float(shoe(np.sin(2*c(t)), np.cos(4*th-2*c(t))))
def exact(a):
    om, A, D, seed, N = a
    rng = np.random.default_rng([seed, int(om*1000), int(A*100), int(D*1000)])
    G = ETA/D; per = 2*np.pi/om; grid = 400; dg = per/grid
    burn = int(np.ceil(10/per)); ncyc = int(np.ceil(30/per))
    c = lambda t: .5*A*np.sin(om*t)
    th = np.full(N, np.pi/8); last = np.zeros(N); nxt = rng.exponential(1/G, N)
    acc = np.zeros(grid)
    for k in range((burn+ncyc)*grid):
        tg = k*dg
        m = nxt < tg
        while m.any():
            i = np.where(m)[0]; te = nxt[i]
            x = th[i]*np.exp(-GAM*(te-last[i]))
            p = (1+np.cos(4*x-2*c(te)))/2
            th[i] = x+np.where(rng.random(i.size) < p, 1., -1.)*D
            last[i] = te; nxt[i] = te+rng.exponential(1/G, i.size)
            m = nxt < tg
        if k >= burn*grid:
            acc[k % grid] += np.cos(4*th*np.exp(-GAM*(tg-last))-2*c(tg)).mean()
    acc /= ncyc; tt = np.arange(grid)*dg
    return float(shoe(np.sin(2*c(tt)), acc))
def euler(a):
    """The repo's scheme with arrival probability Gdt per step."""
    om, A, D, seed, N, Gdt = a
    rng = np.random.default_rng([seed, int(om*1000), int(A*100), int(D*1000), int(Gdt*1e4)])
    G = ETA/D; per = 2*np.pi/om; steps = max(200, int(round(per/(Gdt/G)))); dt = per/steps
    burn = int(np.ceil(10/per)); ncyc = int(np.ceil(30/per)); c = lambda t: .5*A*np.sin(om*t)
    th = np.full(N, np.pi/8); acc = np.zeros(steps); rel = np.exp(-GAM*dt)
    for cy in range(burn+ncyc):
        for s in range(steps):
            k = c(s*dt); ph = 4*th-2*k
            if cy >= burn: acc[s] += np.cos(ph).mean()
            arr = rng.random(N) < G*dt
            th = (th+arr*np.where(rng.random(N) < (1+np.cos(ph))/2, 1., -1.)*D)*rel
    acc /= ncyc; tt = np.arange(steps)*dt
    return float(shoe(np.sin(2*c(tt)), acc))
def euler_det(om, A, D, Gdt):
    """Noise-free iteration of the repo's update: isolates the step-size lag from the noise."""
    G = ETA/D; per = 2*np.pi/om; steps = max(200, int(round(per/(Gdt/G)))); dt = per/steps; rel = np.exp(-GAM*dt)
    c = lambda t: .5*A*np.sin(om*t); th = np.pi/8; O = np.zeros(steps)
    for cy in range(40):
        for s in range(steps):
            ph = 4*th-2*c(s*dt)
            if cy == 39: O[s] = np.cos(ph)
            th = (th+ETA*dt*np.cos(ph))*rel
    return float(shoe(np.sin(2*c(np.arange(steps)*dt)), O))
if __name__ == "__main__":
    out = sys.argv[1]; res = json.load(open(out)) if os.path.exists(out) else {}
    def save(): json.dump(res, open(out+".tmp", "w")); os.replace(out+".tmp", out)
    with mp.Pool(4) as P:
        for om, A, D in [(2., 1., .1), (2., .3, .1), (2., 1., .025), (2., .3, .025), (1.5, 1., .1), (1.25, 1., .1), (1., 1., .1), (1.75, 1., .1), (1.75, 1., .025), (1.5, 1., .025), (2.25, 1., .025), (.12, 1., .1), (2., 1., .25), (2., .3, .25)]:
            k = f"exact_{om}_{A}_{D}"
            if k in res: continue
            r = np.array(P.map(exact, [(om, A, D, s, 20000) for s in range(4)])); m = mf(om, A)
            res[k] = dict(area=r.mean(), sem=r.std(ddof=1)/2, mf=m, ratio=r.mean()/m, rsem=r.std(ddof=1)/2/m); print(k, res[k], flush=True); save()
        for D, gd in [(.1, .1), (.1, .05), (.1, .02), (.025, .1), (.025, .05)]:
            k = f"euler_{D}_{gd}"
            if k in res: continue
            r = np.array(P.map(euler, [(2., 1., D, s, 20000, gd) for s in range(4)])); m = mf(2., 1.)
            res[k] = dict(area=r.mean(), sem=r.std(ddof=1)/2, ratio=r.mean()/m, rsem=r.std(ddof=1)/2/m, det=euler_det(2., 1., D, gd), det_ratio=euler_det(2., 1., D, gd)/m); print(k, res[k], flush=True); save()
