"""Checks of register numbers quoted in Part I. Pool(2). Checkpointed per task to validate_extra.json."""
import numpy as np, json, os, sys, multiprocessing as mp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
ETA, GAM, OM, AA = .5, .01, .12, 1.
chi = lambda t, A=AA, om=OM: .5*A*np.sin(om*t)
def shoe(x, y): return abs(.5*np.sum(x*np.roll(y, -1)-y*np.roll(x, -1)))
def signed_rec(D, N=20000, T=262., seed=1, basis="Z", th_start=0., drive=True, Gdt=.05):
    """Poisson arrivals with signed jump D; returns t, mean theta, final std, fraction leaving the well, output mean."""
    rng = np.random.default_rng([seed, int(D*1e4), ord(basis)])
    G = ETA/D; dt = Gdt/G; n = int(T/dt); th = np.full(N, th_start); rel = np.exp(-GAM*dt)
    mean = np.zeros(n+1); O = np.zeros(n+1); mean[0] = th.mean(); O[0] = np.cos(4*th-(2*chi(0) if drive else 0)).mean()
    left = np.zeros(N, bool); c0 = th_start
    for i in range(n):
        t = i*dt; c = chi(t) if drive else 0.
        ph = 4*th-2*c
        p = {"Z": (1+np.cos(ph))/2, "X": (1+np.sin(ph))/2, "Y": np.full(N, .5)}[basis]
        hit = rng.random(N) < G*dt
        th = (th+hit*np.where(rng.random(N) < p, 1., -1.)*D)*rel
        left |= np.abs(th-c0) > np.pi/4
        mean[i+1] = th.mean(); O[i+1] = np.cos(4*th-2*chi(t+dt)).mean()
    return np.arange(n+1)*dt, mean, th.std(), left.mean(), (np.abs(th-c0) > np.pi/4).mean(), O
def mf_basis(ts, basis="Z", th_start=0.):
    f = {"Z": lambda t, y: ETA*np.cos(4*y-2*chi(t))-GAM*y, "X": lambda t, y: ETA*np.sin(4*y-2*chi(t))-GAM*y, "Y": lambda t, y: -GAM*y}[basis]
    return solve_ivp(f, [0, ts[-1]], [th_start], t_eval=ts, rtol=1e-10, atol=1e-12, method="DOP853").y[0]
def task_rms(D):
    t, m, s, l, lf, O = signed_rec(D, N=20000, T=2*np.pi/OM*5, th_start=0.)
    mf = mf_basis(t)
    m3 = t <= 3*2*np.pi/OM
    return dict(task="rms", D=D, rms_5cyc=float(np.sqrt(np.mean((m-mf)**2))), rms_3cyc=float(np.sqrt(np.mean((m-mf)[m3]**2))), spread_T262=float(s))
def task_well(D):
    # no drive, start at locked point pi/8
    t, m, s, l, lf, O = signed_rec(D, N=20000, T=262., th_start=np.pi/8, drive=False)
    return dict(task="well", D=D, ever_left=float(l), out_at_end=float(lf), std=float(s))
def task_x(args):
    D, st = args
    t, m, s, l, lf, O = signed_rec(D, N=20000, T=2*np.pi/OM*3, basis="X", th_start=st)
    mf = mf_basis(t, "X", st)
    return dict(task="X", D=D, start=st, rms=float(np.sqrt(np.mean((m-mf)**2))))
def task_y(D):
    G = ETA/D
    t, m, s, l, lf, O = signed_rec(D, N=20000 if D > .01 else 6000, T=2*np.pi/OM*3, basis="Y", th_start=0.)
    return dict(task="Y", D=D, rms_output=float(np.sqrt(np.mean((O-np.cos(2*chi(t)))**2))))
def task_area05(D=.5):
    om = OM; per = 2*np.pi/om; A = 1.; G = ETA/D; dt = per/int(np.ceil(per/(.02/G))); N = 40000
    rng = np.random.default_rng(5); steps = int(round(per/dt))
    th = np.full(N, np.pi/8); acc = np.zeros(steps); rel = np.exp(-GAM*dt)
    for c in range(1+6):
        for s in range(steps):
            k = chi(s*dt); ph = 4*th-2*k
            if c >= 1: acc[s] += np.cos(ph).mean()
            hit = rng.random(N) < G*dt
            th = (th+hit*np.where(rng.random(N) < (1+np.cos(ph))/2, 1., -1.)*D)*rel
    acc /= 6; tt = np.arange(steps)*dt
    t = np.linspace(5*per, 6*per, 4001)[:-1]
    so = solve_ivp(lambda t, y: ETA*np.cos(4*y-2*chi(t))-GAM*y, [0, 6*per], [np.pi/8], rtol=1e-10, atol=1e-12, dense_output=True)
    th2 = so.sol(t)[0]; mf = shoe(np.sin(2*chi(t)), np.cos(4*th2-2*chi(t)))
    return dict(task="area05", D=D, pp=float(shoe(np.sin(2*chi(tt)), acc)), mf=float(mf), ratio=float(shoe(np.sin(2*chi(tt)), acc)/mf))
def task_unip(D):
    a, gam, th0 = 50, .01, np.pi/16; G = a*gam/D; dt = .1/G; N = 200000
    f = lambda t: th0+a*np.cos(2*t)**2-t; thmf = brentq(f, th0, np.pi/4)
    ge = gam+2*a*gam*np.sin(4*thmf)
    rng = np.random.default_rng(11); th = np.full(N, thmf); rel = np.exp(-gam*dt)
    for _ in range(int(30/dt)):
        hit = rng.random(N) < G*dt; th = th0+(th+hit*(rng.random(N) < np.cos(2*th)**2)*D-th0)*rel
    x = th.copy(); y = th+.005; R = []; T = []
    for s in range(int(12/dt)):
        hit = rng.random(N) < G*dt; u = rng.random(N)
        x = th0+(x+hit*(u < np.cos(2*x)**2)*D-th0)*rel; y = th0+(y+hit*(u < np.cos(2*y)**2)*D-th0)*rel
        R.append((y-x).mean()); T.append((s+1)*dt)
    R = np.array(R); T = np.array(T); o = {}
    for w in (3., 6., 12.):
        m = (T <= w) & (R > 0); o[w] = float(-np.polyfit(T[m], np.log(R[m]), 1)[0])
    return dict(task="unip", D=D, thmf=thmf, Pplus=float(np.cos(2*thmf)**2), gam_eff=float(ge), rates=o)
def task_kernel(D=.5):
    """Eq. theta_formal at D=0.5 unipolar: mean theta against the kernel integral of the sampled <P+>."""
    a = 50; gam, th0 = .01, np.pi/16; G = ETA/D; dt = .05/G; N = 200000; rng = np.random.default_rng(13)
    th = np.full(N, th0); rel = np.exp(-gam*dt); T = 20.; n = int(T/dt); integ = 0.
    for i in range(n):
        p = np.cos(2*th)**2
        hit = rng.random(N) < G*dt; c = rng.random(N) < p
        th = th0+(th+hit*c*D-th0)*rel
        integ = (integ+ETA*p.mean()*dt)*rel
    return dict(task="kernel", D=D, mean=float(th.mean()-th0), kernel=float(integ), sem=float(th.std()/np.sqrt(N)))
def task_debye():
    o = []
    for eta in (.05, .5):
        ge = GAM+4*eta
        for A in (.1, 1.):
            for fac in (.1, .3, 1, 3, 10):
                om = fac*ge; per = 2*np.pi/om; T = max(20/ge, 2*per)*2
                c = lambda t: .5*A*np.sin(om*t)
                # start on the locked branch; theta0=0
                s = solve_ivp(lambda t, y: eta*np.cos(4*y-2*c(t))-GAM*y, [0, T+per], [np.pi/8], rtol=1e-10, atol=1e-12, dense_output=True, method="DOP853")
                t = np.linspace(T, T+per, 8001)[:-1]; th = s.sol(t)[0]
                num = shoe(np.sin(2*c(t)), np.cos(4*th-2*c(t)))
                pr = 4*np.pi*eta*A**2*om/(ge**2+om**2)
                o.append(dict(eta=eta, A=A, fac=fac, num=float(num), debye=float(pr), ratio=float(num/pr)))
    return dict(task="debye", rows=o)
def area_open(eta, gam, om, th0=np.pi/8):
    per = 2*np.pi/om
    s = solve_ivp(lambda t, y: eta*np.sin(om*t)-gam*(y-th0), [0, 10*per], [th0], rtol=1e-9, atol=1e-11, dense_output=True)
    t = np.linspace(9*per, 10*per, 8001)[:-1]; th = s.sol(t)[0]; v = np.sin(om*t)
    return shoe(v, np.sin(2*th)**2*v), float(th.max()-th.min())
def task_fp():
    return dict(task="fingerprint",
      omega={o: area_open(.15, .1, o) for o in (.05, .1, .15, .2, .25, .3, .35, .4, .5, .7, 1., 3.)},
      eta={e: area_open(e, .1, .3) for e in (.05, .08, .1, .12, .15, .18, .2, .25, .3)},
      gamma={g: area_open(.15, g, .3) for g in (.01, .03, .1, .3, 1.)})
def task_honest():
    from scipy.optimize import minimize
    X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1]).astype(complex)
    ket = lambda t, p: np.array([np.cos(t/2), np.exp(1j*p)*np.sin(t/2)])
    rng = np.random.default_rng(3); o = {}
    for phi in (0, .5, .7495, np.pi/2, np.pi):
        def F(v):
            psi = np.diag([1, 1, 1, np.exp(1j*phi)])@np.kron(ket(v[0], v[1]), ket(v[2], v[3]))
            return -sum((psi.conj()@np.kron(A, B)@psi).real for A, B in ((X, Z), (Z, X), (Y, Y)))
        o[round(phi, 4)] = (float(-min(minimize(F, rng.uniform(0, 6.3, 4)).fun for _ in range(30))), float(1+2*np.sin(phi/2)))
    # Explicit three-photon product pulse |+x>|+y>|+z>, unanimous outcomes only, per-setting conditional correlators
    pX = np.array([1, 1], complex)/2**.5; pY = np.array([1, 1j])/2**.5; pZ = np.array([1, 0], complex)
    psi = np.kron(np.kron(pX, pY), pZ); s = []
    for P in (X, Y, Z):
        w, v = np.linalg.eigh(P); a = v[:, 1]; b = v[:, 0]
        A = np.kron(np.kron(a, a), a); B = np.kron(np.kron(b, b), b)
        pa, pb = abs(A.conj()@psi)**2, abs(B.conj()@psi)**2; s.append((float(pa), float(pb)))
    return dict(task="honest", honest=o, three_photon_probs=s, S3=sum((a-b)/(a+b) for a, b in s))
T = [(task_debye, ()), (task_fp, ()), (task_honest, ()), (task_rms, (.025,)), (task_rms, (.1,)), (task_rms, (.25,)), (task_rms, (.5,)),
     (task_well, (.25,)), (task_well, (.5,)), (task_x, ((.025, 0.),)), (task_x, ((.025, np.pi/8),)), (task_y, (.025,)), (task_y, (.0025,)),
     (task_area05, ()), (task_unip, (.05,)), (task_unip, (.025,)), (task_kernel, ())]
def run(i):
    f, a = T[i]; return i, f(*a)
if __name__ == "__main__":
    out = sys.argv[1]; res = json.load(open(out)) if os.path.exists(out) else {}
    todo = [i for i in range(len(T)) if str(i) not in res]
    with mp.Pool(2) as P:
        for i, r in P.imap_unordered(run, todo):
            res[str(i)] = r; json.dump(res, open(out+".tmp", "w"), default=float); os.replace(out+".tmp", out); print(i, json.dumps(r, default=float)[:1800], flush=True)
