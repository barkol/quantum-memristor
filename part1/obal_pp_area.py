# The area ratios of this script carry the lag of the update at G*dt = 0.1; the manuscript quotes pp_event_driven.py.
# minimal own point-process checks (c,d,e); run on ultron
import numpy as np, multiprocessing as mp, json, sys
from scipy.integrate import solve_ivp
gam=.01; eta=.5
def area(x,y):
    return abs(.5*np.sum(x*np.roll(y,-1)-y*np.roll(x,-1)))
def mf_area(om,Achi):
    chi=lambda t:(Achi/2)*np.sin(om*t)
    per=2*np.pi/om; T0=max(60,20*per)
    f=lambda t,th:eta*np.cos(4*th-2*chi(t))-gam*th
    th0=np.pi/8
    s=solve_ivp(f,[0,T0+per],[th0],rtol=1e-9,atol=1e-11,dense_output=True,method='LSODA')
    t=np.linspace(T0,T0+per,4001)[:-1]; th=s.sol(t)[0]
    return area(np.sin(2*chi(t)),np.cos(4*th-2*chi(t)))
def pp_area(args):
    om,Achi,D,seed=args; rng=np.random.default_rng(seed); N=30000
    G=eta/D; dt=min(.1/G,.02); per=2*np.pi/om; steps=max(200,int(round(per/dt))); dt=per/steps
    ncyc=max(1,int(np.ceil(30/per))); burn=max(1,int(np.ceil(10/per)))
    th=np.full(N,np.pi/8); 
    chi0=lambda t:(Achi/2)*np.sin(om*t)
    th=np.pi/8+chi0(0)/2+0*th
    acc=np.zeros(steps); relax=np.exp(-gam*dt)
    t=0.
    for c in range(burn+ncyc):
        for s in range(steps):
            chi=chi0(t); ph=4*th-2*chi
            if c>=burn: acc[s]+=np.mean(np.cos(ph))
            arr=rng.random(N)<G*dt
            p=(1+np.cos(ph))/2
            sgn=np.where(rng.random(N)<p,1.,-1.)
            th=th+arr*sgn*D
            th=th*relax
            t+=dt
    acc/=ncyc
    tt=(np.arange(steps))*dt
    return dict(om=om,Achi=Achi,D=D,app=area(np.sin(2*chi0(tt)),acc))
def kappa(args):
    D,seed=args; rng=np.random.default_rng(seed); N=100000; G=eta/D; dt=.1/G
    th=np.full(N,np.pi/8)
    for s in range(int(6/dt)):
        ph=4*th; arr=rng.random(N)<G*dt
        th=(th+arr*np.where(rng.random(N)<(1+np.cos(ph))/2,1.,-1.)*D)*np.exp(-gam*dt)
    z=np.exp(1j*4*th); return dict(D=D,kappa=abs(z.mean()),meansin=np.sin(4*th).mean())
def rate(args):
    D,seed=args; rng=np.random.default_rng(seed); N=100000; G=eta/D; dt=.1/G
    th=np.full(N,np.pi/8)
    for s in range(int(8/dt)):
        ph=4*th; arr=rng.random(N)<G*dt
        th=(th+arr*np.where(rng.random(N)<(1+np.cos(ph))/2,1.,-1.)*D)*np.exp(-gam*dt)
    # settled independent copies: offset mean, CRN
    a=th.copy(); b=th+.02; res=[];ts=[]
    n=int(2/dt)
    for s in range(n):
        u=rng.random(N); arr=rng.random(N)<G*dt
        a=(a+arr*np.where(u<(1+np.cos(4*a))/2,1.,-1.)*D)*np.exp(-gam*dt)
        b=(b+arr*np.where(u<(1+np.cos(4*b))/2,1.,-1.)*D)*np.exp(-gam*dt)
        res.append((b-a).mean()); ts.append((s+1)*dt)
    res=np.array(res);ts=np.array(ts); out={}
    for w in (.25,.5,1,2):
        m=(ts<=w)&(res>0); out[w]=-np.polyfit(ts[m],np.log(res[m]),1)[0]
    return dict(D=D,rates=out)
def spread(args):
    D,mode,seed=args; rng=np.random.default_rng(seed); N=20000; G=eta/D; om=.12;A=1
    th=np.full(N,0.); t=0; T=262.
    # start theta=0 is in the well? use default
    if mode=='clock':
        dt=1/G
        while t<T:
            ph=4*th-2*(A/2)*np.sin(om*t); th=(th+np.where(rng.random(N)<(1+np.cos(ph))/2,1.,-1.)*D)*np.exp(-gam*dt); t+=dt
    elif mode=='pulse1':
        dt=1/G
        while t<T:
            ph=4*th-2*(A/2)*np.sin(om*t); p=(1+np.cos(ph))/2; n=rng.poisson(1,N); k=rng.binomial(n,p)
            th=(th+(2*k-n)*D)*np.exp(-gam*dt); t+=dt
    else:
        dt=.1/G
        while t<T:
            ph=4*th-2*(A/2)*np.sin(om*t); arr=rng.random(N)<G*dt
            th=(th+arr*np.where(rng.random(N)<(1+np.cos(ph))/2,1.,-1.)*D)*np.exp(-gam*dt); t+=dt
    return dict(D=D,mode=mode,std=th.std())
def fold(args):
    chi,a,m,w=args; g=1.;
    f=lambda t,th:g*a*(1+m*np.sin(w*t))*np.cos(2*th-chi)**2-g*th
    per=2*np.pi/w
    s=solve_ivp(f,[0,6*per],[0.],method='LSODA',rtol=1e-8,atol=1e-10,dense_output=True)
    t=np.linspace(5*per,6*per,20001)[:-1]; th=s.sol(t)[0]
    G=1+m*np.sin(w*t); R=G*np.cos(2*th-chi)**2
    return dict(chi=chi,a=a,m=m,w=w,A=area(G,R),P=[float(np.min(np.cos(2*th-chi)**2)),float(np.max(np.cos(2*th-chi)**2))])
if __name__=='__main__':
    with mp.Pool(6) as P:
        args=[(om,A,D,10) for om in (.12,1,2,4,8) for A in (.3,1.) for D in (.025,.1,.25)]
        r=P.map(pp_area,args)
    for x in r:
        mf=mf_area(x['om'],x['Achi']); print(x['om'],x['Achi'],x['D'],round(x['app'],4),round(mf,4),round(x['app']/mf,3),flush=True)
