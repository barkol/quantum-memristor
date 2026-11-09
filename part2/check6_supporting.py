#!/usr/bin/env python3
r"""Controls behind numbers quoted in Part II: J = 0 scan and its mean, signed overlap, L = 1, 2, n up to 14, random start, bare CCZ and no-field capacity, ESN sizes, gamma below the sweep. Every job is appended to results/check6_supporting.jsonl; a restart skips finished jobs."""
# K audit: independent reruns; every job appended to k2.jsonl, restart skips finished jobs.
import os,sys,json
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"))
import numpy as np
from multiprocessing import Pool
from scipy.linalg import hadamard
OUT="check6_supporting.jsonl"
FF,TR,THR=0.15,120,0.9

def capfn(kind,n,seed,M):
    rg=np.random.default_rng(seed)
    P=rg.choice([-1.,1.],size=(M,n))
    tgt=rg.integers(0,M,TR);S=P[tgt].copy();k=int(FF*n)
    for i in range(TR):S[i,rg.choice(n,k,replace=False)]*=-1
    rng=np.random.default_rng(1)
    if kind in('full','nofield'):
        f=(lambda m:m**3) if kind=='full' else (lambda m:(m**3-(3*n-2)*m)/6.0)
        m=S@P.T
        for _ in range(60):
            mv=np.zeros(TR,bool)
            for i in rng.permutation(n):
                ds=-2.0*S[:,i];mn=m+np.outer(ds,P[:,i])
                dE=-(f(mn).sum(1)-f(m).sum(1));fl=dE<-1e-9
                S[fl,i]*=-1;m[fl]=mn[fl];mv|=fl
            if not mv.any():break
    else: # bare CCZ: H = sum_{i<j<k} J_ijk(-sss + sum ss - sum s), J_ijk=sum_mu xi xi xi
        T=np.einsum('mi,mj,mk->ijk',P,P,P,optimize=True)
        idx=np.arange(n)
        T[idx,idx,:]=0;T[idx,:,idx]=0;T[:,idx,idx]=0
        G=T.sum(2)            # G_ab=sum_{c} J_abc (c != a,b automatically)
        h=T.sum((1,2))/2.0    # h_a=sum_{b<c} J_abc
        for _ in range(60):
            mv=np.zeros(TR,bool)
            for i in rng.permutation(n):
                # dE of flipping i: 2 s_i [ sum_{b<c} J_ibc s_b s_c - sum_b G_ib s_b + h_i ]
                q=0.5*np.einsum('bc,tb,tc->t',T[i],S,S,optimize=True)
                dE=2*S[:,i]*(q-S@G[i]+h[i])
                fl=dE<-1e-9;S[fl,i]*=-1;mv|=fl
            if not mv.any():break
    return float(np.mean(np.all(S==P[tgt],axis=1)))

def capacity(kind,n,s):
    seed=100+7*s;lo,hi=1,2
    while hi<=16384:
        if capfn(kind,n,seed,hi)>=THR:lo=hi;hi*=2
        else:break
    while hi-lo>max(1,int(lo*0.01)):
        mid=(lo+hi)//2
        if capfn(kind,n,seed,mid)>=THR:lo=mid
        else:hi=mid
    return lo

def hadam(n,s):
    rg=np.random.default_rng(100+7*s);H=hadamard(n).astype(float)
    P=H[1:].copy();tgt=rg.integers(0,n-1,TR);S=P[tgt].copy();k=int(FF*n)
    for i in range(TR):S[i,rg.choice(n,k,replace=False)]*=-1
    m=S@P.T;rng=np.random.default_rng(1)
    for _ in range(60):
        mv=np.zeros(TR,bool)
        for i in rng.permutation(n):
            ds=-2.0*S[:,i];mn=m+np.outer(ds,P[:,i])
            dE=-((mn**2).sum(1)-(m**2).sum(1));fl=dE<-1e-9
            S[fl,i]*=-1;m[fl]=mn[fl];mv|=fl
        if not mv.any():break
    return float(np.mean(np.all(S==P[tgt],axis=1)))

def res_job(g,seed):
    import qrc_benchmark as b
    u,t=b.data(seed);f=b.reservoir(u,g,b.DEV)
    return float(b.q.evaluate(f,t,b.TRAIN,b.TEST)[0])
def esn_job(nf,seed):
    import qrc_benchmark as b
    u,t=b.data(seed);f=b.q.esn_features(u,nf,np.random.default_rng(30000+seed))
    return float(b.q.evaluate(f,t,b.TRAIN,b.TEST)[0])

def net_job(seed,n,L,f,nets,tr):
    # biased greedy, SA, QA at h=0.6 plus unbiased and random-start biased; sign-blind and signed
    import sim_qhnm as m
    from fixes_compute import greedy_biased_batch,corrupt
    rng=np.random.default_rng(seed);m.RNG=np.random.default_rng(seed+1)
    spins=m.spin_table(n);nf=int(np.floor(f*n+.5));r={k:[] for k in['gb','sa','q','gu','gr','gb_s','sa_s','q_s','gu_s','gr_s','exact_gb','exact_q','same']}
    for _ in range(nets):
        P=rng.choice([-1.,1.],size=(L,n))
        J,_x=m.stream_memristor_couplings(P,eta=1./n,gamma=.01,dt=1.,steps_per_pattern=1)
        tg=rng.integers(0,L,tr);tb=P[tg];nb=corrupt(rng,tb,nf,n)
        G=greedy_biased_batch(J,nb,.6);A=m.retrieve_classical_sa_batch(J,nb,.6)
        Q=m.retrieve_quantum_anneal_batch(J,nb,Omega0=1.,n_steps=120,T=12.,bias_strength=.6,spins=spins)
        U=m.retrieve_classical_batch(J,nb)
        # random start, same biased energy: single-spin-flip descent
        S=rng.choice([-1.,1.],size=nb.shape);h=.6*nb
        for _s in range(200):
            mv=False
            for i in range(n):
                fld=S@J[i]-S[:,i]*J[i,i]+h[:,i];b=np.sign(fld);b[b==0]=S[:,i][b==0]
                if (b!=S[:,i]).any():mv=True
                S[:,i]=b
            if not mv:break
        for nm,R in(('gb',G),('sa',A),('q',Q),('gr',S))+((('gu',U),) if U is not None else ()):
            r[nm].append(float(m.overlap_batch(R,tb).mean()));r[nm+'_s'].append(float(((R*tb).sum(1)/n).mean()))
        r['exact_gb'].append(float((m.overlap_batch(G,tb)>.999).mean()));r['exact_q'].append(float((m.overlap_batch(Q,tb)>.999).mean()))
        r['same'].append(float((G==A).all(1).mean()))
    return {k:(np.mean(v),np.std(v,ddof=1)/np.sqrt(len(v))) if len(v)>1 else None for k,v in r.items()}

def cap_small(seed,n,L,nets=60,pr=8):
    import sim_qhnm as m
    m.RNG=np.random.default_rng(seed);rng=np.random.default_rng(seed+7);sp=m.spin_table(n)
    nf=int(np.floor(.125*n+.5));hits=[]
    from fixes_compute import corrupt
    for _ in range(nets):
        P=rng.choice([-1.,1.],size=(L,n))
        J,_x=m.stream_memristor_couplings(P,eta=1./n,gamma=.01,dt=1.,steps_per_pattern=1)
        tg=rng.integers(0,L,pr);tb=P[tg];nb=corrupt(rng,tb,nf,n)
        R=m.retrieve_quantum_anneal_batch(J,nb,Omega0=1.,spins=sp)
        hits.append(float((m.overlap_batch(R,tb)>.999).mean()))
    return float(np.mean(hits)),float(1.96*np.std(hits,ddof=1)/np.sqrt(nets))

def scan(_):
    # J=0 control of check2_quantum_input.py; exact same seeds; J switch
    import sim_qhnm as m
    n,L=8,3;m.RNG=np.random.default_rng(4242);P=m.RNG.choice([-1,1],size=(L,n))
    J0,_x=m.stream_memristor_couplings(P,eta=1./n,gamma=.01,dt=1.,steps_per_pattern=1)
    spins=m.spin_table(n);dim=2**n;idx=lambda s:int("".join(map(str,((1-np.asarray(s))//2).astype(int))),2)
    rg=np.random.default_rng(7)
    def cor(x):
        y=x.copy();y[rg.choice(n,1,replace=False)]*=-1;return y
    xa,xb=P[0],P[1];ta,tb=cor(xa),cor(xb)
    ea=np.zeros(dim,complex);ea[idx(ta)]=1;eb=np.zeros(dim,complex);eb[idx(tb)]=1
    tg=np.zeros(dim,complex);tg[idx(xa)]+=1;tg[idx(xb)]+=1;tg/=np.linalg.norm(tg)
    same=(np.where(ta!=xa)[0].tolist(),np.where(tb!=xb)[0].tolist(),int((xa!=xb).sum()))
    out={'flips_and_dist':same}
    for lab,J in(('J',J0),('J0',0*J0)):
        Ed=m.classical_energy(J,np.zeros(n),spins);Fs=[]
        for Om in(.1,.2,.35,.5,.8):
            for T in(2,5,10,20):
                K=max(200,int(40*T));psi=((ea+eb)/np.sqrt(2)).copy();dt=T/K
                for k in range(K):
                    s=(k+.5)/K;Ax=(1-s)*Om
                    psi=m._apply_x_rotations(psi,n,-Ax*dt/2.);psi=psi*np.exp(-1j*s*Ed*dt);psi=m._apply_x_rotations(psi,n,-Ax*dt/2.)
                Fs.append(float(abs(np.vdot(tg,psi))**2))
        out[lab]=dict(best=max(Fs),mean=float(np.mean(Fs)),n=len(Fs))
    return out

def run(k):
    t=k[0]
    if t=='cap':return capacity(k[1],k[2],k[3])
    if t=='had':return hadam(k[1],k[2])
    if t=='res':return res_job(k[1],k[2])
    if t=='esn':return esn_job(k[1],k[2])
    if t=='net':return net_job(*k[1:])
    if t=='small':return cap_small(*k[1:])
    if t=='scan':return scan(0)
def wrap(k):
    try:return k,run(tuple(k))
    except Exception as e:return k,{'err':repr(e)}
if __name__=='__main__':
    jobs=[['scan',0]]
    jobs+=[['had',n,s] for n in(64,128,256) for s in range(3)]
    jobs+=[['net',s,8,3,f,60,100] for s in(11,12,13,14) for f in(.125,.1875,.28125,.3125)]
    jobs+=[['net',21,8,1,.125,60,100],['net',21,8,2,.125,60,100]]
    jobs+=[['small',s,n,L] for s in(1,2,3) for n in(6,8,10,12,14) for L in(1,2)]
    jobs+=[['esn',nf,s] for nf in(6,27) for s in range(30)]
    jobs+=[['res',g,s] for g in(.002,.004,.007) for s in range(30)]
    jobs+=[['cap','nofield',64,s] for s in range(12)]+[['cap','full',64,s] for s in range(3)]
    jobs+=[['cap','bare',64,s] for s in range(12)]
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT):done.add(json.dumps(json.loads(l)[0]))
    todo=[j for j in jobs if json.dumps(j) not in done]
    print(len(jobs),len(todo),flush=True)
    with Pool(8) as p,open(OUT,'a') as f:
        for r in p.imap_unordered(wrap,todo):
            f.write(json.dumps(r)+"\n");f.flush();print(r[0],flush=True)
    print('DONE',flush=True)
