import os
NZ = os.environ.get("NZ_DIR", "results/part1/nz")
# Grid: Delta x A x omega x gamma. Each point: find settling time from hierarchy, then MC batches until
# rel. SEM of pair area <= target, or event budget exhausted. One npz per point => checkpoint & resume.
import numpy as np,time,sys,os,json,itertools,multiprocessing as mp
from nzlib import *
OUT=f'{NZ}/grid'
ETA=.5;TH0=0.;TH_I=np.pi/8;M=256
DS=[.025,.05,.1,.25,.5];AS=[.2,1.,2.];OS=[.1,.5,2.,6.];GS=[.001,.01,.1,.5]
TARGET=0.001;BUDGET=float(os.environ.get('BUDGET','2.5e9'));NB=24
def nrule(D): return int(np.ceil(np.pi/(2*D)))-1
def settle(D,A,om,g,nmax):
    Tp=2*np.pi/om
    f=hier_rhs_factory(D,ETA,g,TH0,A,om,nmax)
    y=cn_init(TH_I,TH0,nmax);t=0.;prev=None
    Tmin=max(2*Tp,6.0);Tcap=max(4*Tp,min(4/g,3000.))
    while True:
        ts=t+np.arange(M+1)*Tp/M
        s=solve_ivp(f,(t,ts[-1]),y,t_eval=ts,method='DOP853',rtol=1e-8,atol=1e-10)
        y=s.y[:,-1];t=ts[-1]
        chi=(A/2)*np.sin(om*ts[:-1]);ar=shoelace(np.sin(2*chi),s.y[1,:-1])
        if prev is not None and t>=Tmin and abs(ar-prev)<=2e-4*abs(ar)+1e-7:
            return t,True,ar
        if t>=Tcap: return t,False,ar
        prev=ar
def point(args):
    D,A,om,g=args
    fn=f'{OUT}/D{D}_A{A}_w{om}_g{g}.npz'
    if os.path.exists(fn): return fn,'skip'
    t0=time.time();nmax=nrule(D)
    Ts,settled,arH=settle(D,A,om,g,nmax)
    Tp=2*np.pi/om
    ts=Ts+np.arange(M)*Tp/M
    chi=(A/2)*np.sin(om*ts);x=np.sin(2*chi)
    ss=np.random.SeedSequence([int(D*1000),int(A*10),int(om*10),int(g*1000)])
    seeds=iter(ss.generate_state(4096,np.uint64))
    # pilot
    def batch(N):
        o,h,e=mc(N,D,ETA,g,TH0,A,om,False,TH_I,ts,int(next(seeds)));return o,h,e
    Np=1000;bs=[];nev=0
    for b in range(8):
        o,h,e=batch(Np);bs.append((Np,o,h));nev+=e
    def areas(o): return shoelace(x,o[:,0,0]),shoelace(-x,o[:,1,0])
    a=np.array([np.mean(areas(o)) for _,o,_ in bs])
    rel=a.std(ddof=1)*np.sqrt(Np)/max(abs(a.mean()),1e-12)
    need=(rel/TARGET)**2*1.2
    evp=nev/(8*Np)+M
    Ntot=int(min(need,BUDGET/evp));Ntot=max(Ntot,24*2000)
    N=max(Ntot//NB,2000)
    bs=[]; nev=0
    for b in range(NB):
        o,h,e=batch(N);bs.append((N,o,h));nev+=e
    O=np.array([b[1] for b in bs]);hist=np.mean([b[2] for b in bs],axis=0)
    apair=np.array([np.mean(areas(o)) for o in O]);aplus=np.array([areas(o)[0] for o in O])
    np.savez_compressed(fn+'.tmp.npz',D=D,A=A,om=om,g=g,Ts=Ts,settled=settled,N=N,nb=NB,nev=nev,ts=ts,x=x,mean=O.mean(0),sem=O.std(0,ddof=1)/np.sqrt(NB),apair=apair,aplus=aplus,hist=hist,arH=arH,elapsed=time.time()-t0)
    os.replace(fn+'.tmp.npz',fn)
    return fn,f'{time.time()-t0:.0f}s N={N*NB} relsem={apair.std(ddof=1)/np.sqrt(NB)/abs(apair.mean()):.2e} Ts={Ts:.0f} settled={settled}'
if __name__=='__main__':
    os.makedirs(OUT,exist_ok=True)
    pts=list(itertools.product(DS,AS,OS,GS))
    np.random.default_rng(0).shuffle(pts)
    with mp.Pool(int(sys.argv[1]) if len(sys.argv)>1 else 11) as p:
        for i,(fn,msg) in enumerate(p.imap_unordered(point,[tuple(map(float,q)) for q in pts])):
            print(i,os.path.basename(fn),msg,flush=True)
