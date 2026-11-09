# unipolar closure and weak-feedback rate; NARMA eta=0 check (repo code on copy)
import numpy as np, multiprocessing as mp
def unip(args):
    D,a,gam,th0,seed=args; rng=np.random.default_rng(seed); N=20000; G=a*gam/D
    from scipy.optimize import brentq
    # MF locked point below first dark zero
    f=lambda t:th0+a*np.cos(2*t)**2-t
    thmf=brentq(f,th0,np.pi/4) if f(np.pi/4)<0 else None
    th=np.full(N,thmf); dt=.1/G; t=0; out={}
    T=[100,2000] if gam==.01 else [100]
    for Tmax in T:
        while t<Tmax:
            arr=rng.random(N)<G*dt
            th=th0+(th+arr*(rng.random(N)<np.cos(2*th)**2)*D-th0)*np.exp(-gam*dt); t+=dt
        out[Tmax]=(float((th>np.pi/4).mean()),float(th.mean()))
    return dict(D=D,a=a,thmf=thmf,out=out)
def weak(args):
    D,a,th0,seed=args; gam=.1; rng=np.random.default_rng(seed); N=200000; G=a*gam/D; dt=min(.1/G,.05)
    th=np.full(N,th0)
    for s in range(int(60/dt)):
        arr=rng.random(N)<G*dt
        th=th0+(th+arr*(rng.random(N)<np.cos(2*th)**2)*D-th0)*np.exp(-gam*dt)
    A=th.copy();B=th+.05; r=[];ts=[]
    for s in range(int(15/dt)):
        u=rng.random(N);arr=rng.random(N)<G*dt
        A=th0+(A+arr*(u<np.cos(2*A)**2)*D-th0)*np.exp(-gam*dt)
        B=th0+(B+arr*(u<np.cos(2*B)**2)*D-th0)*np.exp(-gam*dt)
        r.append((B-A).mean());ts.append((s+1)*dt)
    r=np.array(r);ts=np.array(ts);m=(ts<8)&(r>0); rate=-np.polyfit(ts[m],np.log(r[m]),1)[0]
    # MF
    from scipy.integrate import solve_ivp
    ss=solve_ivp(lambda t,x:a*gam*np.cos(2*x)**2-gam*(x-th0),[0,500],[th0],rtol=1e-9).y[0,-1]
    g=gam+2*a*gam*np.sin(2*(2*ss))
    return dict(D=D,a=a,th0=round(th0,3),rate=round(rate,4),gam_eff=round(g,4),ratio=round(rate/g,3))
if __name__=='__main__':
    with mp.Pool(6) as P:
        print(P.map(unip,[(D,50,.01,np.pi/16,1) for D in (.025,.05,.1)]),flush=True)
        for x in P.map(weak,[(.1,1,t,2) for t in (np.pi/8,0,-np.pi/8,3*np.pi/8)]+[(.01,.1,np.pi/16,3)]): print(x,flush=True)
