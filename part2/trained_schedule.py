import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from retrieval_core import *
# variational schedule: piecewise-linear s(u) and A(u), 6 knots each, plus Omega0 and optional h; (mu,lambda) ES
# train: base 5200; test: base 5300 (held out).
n=8;Z=spins_table(8).astype(float)
U=np.linspace(0,1,6)
def ev(a):
    th,base,net,L,f,tr,T,hfix=a
    s_k=np.clip(th[:6],0,1);a_k=np.clip(th[6:12],0,None);hv=hfix if hfix else float(np.clip(th[12],.02,1.5))
    pats,mu,xi,xit,k=make_probes(base,8,L,net,f,tr)
    J=hebbian(pats);h=hv*xit.astype(float);EJ=energy_J_table(J,Z)
    P=anneal(J,h,Z,EJ,T=T,K=int(T*10),Omega0=1.0,sched=lambda u:np.interp(u,U,s_k),omega_sched=lambda u:np.interp(u,U,a_k))
    q=Z[P.argmax(1)]
    return float((np.abs((xi*q).sum(1))/8).mean())
def fit(pool,L,f,T,hfix,seed,gens=25,lam=32,mu=8,Ntr=40,tr=25):
    rng=np.random.default_rng(seed)
    x=np.r_[U,1-U,0.6];sig=.15
    best=(-1,None);hist=[]
    for g in range(gens):
        cand=[x+sig*rng.normal(size=13) for _ in range(lam)]
        if g==0: cand[0]=x.copy()
        args=[(c,5200,net,L,f,tr,T,hfix) for c in cand for net in range(Ntr)]
        v=np.array(pool.map(ev,args,chunksize=16)).reshape(lam,Ntr).mean(1)
        o=np.argsort(-v)[:mu]
        x=np.mean([cand[i] for i in o],axis=0);sig*=.97
        if v[o[0]]>best[0]: best=(v[o[0]],cand[o[0]].copy())
        hist.append(float(v[o[0]]))
    return best,hist
def test(pool,th,L,f,T,hfix,base=5300,N=60,tr=100):
    v=np.array(pool.map(ev,[(th,base,net,L,f,tr,T,hfix) for net in range(N)],chunksize=4))
    return v
if __name__=='__main__':
    path='../results/trained_schedule.jsonl';st=Store(path)
    with Pool(8) as pool:
        for L,f,T,hfix in [(3,.125,12,.6),(3,.1875,12,.6),(3,.125,50,.0),(3,.1875,50,.0),(3,.125,12,.0),(3,.1875,12,.0),(5,.125,50,.0),(5,.1875,50,.0)]:
            key=f'{L}|{f}|{T}|{hfix}'
            if key in st.done: continue
            b,hist=fit(pool,L,f,T,hfix,seed=[11,L,int(f*1e4),T,int(hfix*10)])
            t=test(pool,b[1],L,f,T,hfix)
            hv=hfix if hfix else float(np.clip(b[1][12],.02,1.5))
            # reference on the same test networks at same h and on grid of h: biased descent
            ref={}
            for hh in sorted(set([hv,.1,.2,.3,.4,.6])):
                gv=[]
                for net in range(60):
                    pats,mu,xi,xit,k=make_probes(5300,8,L,net,f,100)
                    J=hebbian(pats);s=greedy(J,hh*xit.astype(float),xit);gv.append((np.abs((xi*s).sum(1))/8).mean())
                ref[str(round(hh,4))]=gv
            st.add(key,{'train':b[0],'hist':hist,'th':b[1].tolist(),'test':t.tolist(),'h':hv,'gb_ref':ref})
            print(key,b[0],t.mean(),flush=True)
    print('DONE')
