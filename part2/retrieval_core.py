"""R2 independent implementation from manuscript equations only (Eq. hebbian, overlap, A1, Appendix)."""
import numpy as np, json, os, sys, time, math
from multiprocessing import Pool

def hebbian(pats):
    n=pats.shape[1]
    J=pats.T@pats/n
    np.fill_diagonal(J,0.0)
    return J

def nflip(f,n,mode='halfup'):
    x=f*n
    if mode=='halfup': return int(math.floor(x+0.5+1e-12))
    return int(np.round(x))   # banker's

def spins_table(n):
    idx=np.arange(2**n)
    return 1-2*((idx[:,None]>>np.arange(n))&1).astype(np.int8)  # bit q -> sigma_z on qubit q

def strings_to_idx(s):
    n=s.shape[-1]
    return (((1-s)//2).astype(np.int64)<<np.arange(n)).sum(-1)

def energy_J_table(J,Z):
    return -0.5*np.einsum('ai,ij,aj->a',Z,J,Z,optimize=True)

def energies(J,h,s):
    """E(s) = -sum_{i<j} J s s - sum h s; s (B,n); h (B,n)"""
    return -0.5*np.einsum('bi,ij,bj->b',s,J,s,optimize=True)-(h*s).sum(1)

def anneal(J,hB,Z,EJ,Omega0=1.0,T=12.0,K=120,init='plus',xi_init=None,sched=None,order='ZXZ',omega_sched=None):
    """Exact 2^n Trotter of Eq. (A1). hB: (B,n). returns probs (B,2^n).
    sched(u) -> s at u in (0,1); default s=u. omega_sched optional independent A(u)."""
    n=J.shape[0];D=2**n;B=hB.shape[0]
    Ed=EJ[None,:]-(hB@Z.T)          # (B,D) problem diag
    if init=='plus':
        psi=np.full((B,D),1/np.sqrt(D),dtype=np.complex128)
    else:
        psi=np.zeros((B,D),dtype=np.complex128)
        psi[np.arange(B),strings_to_idx(xi_init)]=1.0
    dt=T/K
    for k in range(K):
        u=(k+0.5)/K
        s=u if sched is None else sched(u)
        A=(1-s)*Omega0 if omega_sched is None else omega_sched(u)*Omega0
        if order=='ZXZ':
            ph=np.exp(-0.5j*dt*s*Ed)
            psi*=ph
            rx(psi,n,A*dt)
            psi*=ph
        else:
            rx(psi,n,A*dt/2)
            psi*=np.exp(-1j*dt*s*Ed)
            rx(psi,n,A*dt/2)
    return (psi.real**2+psi.imag**2)

def rx(psi,n,a):
    """apply prod_q exp(+i a sigma_x) in place"""
    c=np.cos(a);s=1j*np.sin(a)
    B,D=psi.shape
    for q in range(n):
        st=1<<q
        v=psi.reshape(B,D>>(q+1),2,st)
        a0=v[:,:,0,:].copy()
        v[:,:,0,:]=c*a0+s*v[:,:,1,:]
        v[:,:,1,:]=s*a0+c*v[:,:,1,:]

def greedy(J,h,s0,order='seq',rng=None):
    """Descent: flip spin i iff the flip strictly lowers E. Batch. order: seq|rand|steep"""
    s=s0.copy().astype(np.float64)
    B,n=s.shape
    if order=='steep':
        while True:
            lf=s@J+h
            d=2*s*lf            # Delta E
            i=d.argmin(1)
            mask=d[np.arange(B),i]<-1e-10
            if not mask.any(): break
            s[np.where(mask)[0],i[mask]]*=-1
        return s.astype(np.int8)
    while True:
        ch=False
        perm=np.arange(n) if order=='seq' else rng.permutation(n)
        for i in perm:
            lf=s@J[i]+h[:,i]
            m=s[:,i]*lf<-1e-10
            if m.any():
                s[m,i]*=-1; ch=True
        if not ch: break
    return s.astype(np.int8)

def sa(J,h,xit,rng,start='random',sweeps=300,restarts=3,hbar_mode='input'):
    """Metropolis, geometric T0=1.5 hbar -> 0.02 hbar, best of restarts, T=0 settle."""
    B,n=xit.shape
    best=None;bestE=None
    lf0=xit@J+h
    if hbar_mode=='input':
        hb=np.median(np.abs(lf0),axis=1)
    elif hbar_mode=='h+J':
        hb=np.median(np.abs(J).sum(1))+np.median(np.abs(h),axis=1)
    hb=np.maximum(hb,1e-9)
    for r in range(restarts):
        s=(rng.choice([-1.0,1.0],size=(B,n)) if start=='random' else xit.astype(np.float64).copy())
        Ts=np.geomspace(1.5,0.02,sweeps)
        for T_ in Ts:
            for i in rng.permutation(n):
                lf=s@J[i]+h[:,i]
                dE=2*s[:,i]*lf
                acc=(dE<=0)|(rng.random(B)<np.exp(-np.clip(dE/(T_*hb),0,500)))
                s[acc,i]*=-1
        s=greedy(J,h,s.astype(np.int8)).astype(np.float64)
        E=energies(J,h,s)
        if best is None: best=s.copy();bestE=E
        else:
            m=E<bestE-1e-12; best[m]=s[m];bestE[m]=E[m]
    return best.astype(np.int8)

def overlaps(xi,s):
    n=xi.shape[1]
    d=(xi*s).sum(1)/n
    return np.abs(d),d

def make_net(base,n,L,net):
    rng=np.random.default_rng([base,n,L,net])
    return rng.choice([-1,1],size=(L,n)).astype(np.int8)

def make_probes(base,n,L,net,f,trials,round_mode='halfup',fkey=None):
    pats=make_net(base,n,L,net)
    rng=np.random.default_rng([base,n,L,net,1000+int(round(f*1e6))])
    mu=rng.integers(0,L,size=trials)
    k=nflip(f,n,round_mode)
    xi=pats[mu]
    xit=xi.copy()
    for t in range(trials):
        fl=rng.choice(n,size=k,replace=False)
        xit[t,fl]*=-1
    return pats,mu,xi,xit,k

def cell(base,n,L,net,f,hval,trials=100,round_mode='halfup',qa=True,qa_kw=None,sa_kw=None,want_extra=True,Z_cache={}):
    """Run all rules on one network at one (f,h). Returns dict of per-network means."""
    pats,mu,xi,xit,k=make_probes(base,n,L,net,f,trials,round_mode)
    J=hebbian(pats)
    h=hval*xit.astype(np.float64)
    rng=np.random.default_rng([base,n,L,net,2000+int(round(f*1e6)),int(round(hval*1e4))])
    if n not in Z_cache: Z_cache[n]=spins_table(n).astype(np.float64)
    Z=Z_cache[n];EJ=energy_J_table(J,Z)
    out={}
    gu=greedy(J,np.zeros_like(h),xit)
    gb=greedy(J,h,xit)
    kw=dict(sa_kw or {})
    sv=sa(J,h,xit,rng,**kw)
    Ed=EJ[None,:]-h@Z.T
    gsidx=Ed.argmin(1);Emin=Ed.min(1)
    gs=Z[gsidx].astype(np.int8)
    rules={'gu':gu,'gb':gb,'sa':sv,'gs':gs,'in':xit}
    P=None
    if qa:
        P=anneal(J,h,Z,EJ,**(qa_kw or {}))
        q=Z[P.argmax(1)].astype(np.int8)
        rules['qa']=q
    for name,s in rules.items():
        a,d=overlaps(xi,s)
        out['m_'+name]=float(a.mean())
        out['ms_'+name]=float(d.mean())
        out['x_'+name]=float((a>0.999).mean())
        out['xs_'+name]=float((d>0.999).mean())
        Es=energies(J,h,s.astype(np.float64))
        out['pgs_'+name]=float((Es<=Emin+1e-9).mean())
        out['Eres_'+name]=float((Es-Emin).mean())
    for nm in ('sa','qa'):
        if nm in rules: out['agree_gb_'+nm]=float((rules['gb']==rules[nm]).all(1).mean())
    if qa:
        m_all=np.abs(xi@Z.T)/n; ms_all=(xi@Z.T)/n
        out['m_qa_samp']=float((P*m_all).sum(1).mean())
        out['ms_qa_samp']=float((P*ms_all).sum(1).mean())
        out['Eres_qa_samp']=float(((P*Ed).sum(1)-Emin).mean())
        gsmask=Ed<=Emin[:,None]+1e-9
        out['pgs_qa_prob']=float((P*gsmask).sum(1).mean())
    out['k']=k
    return out

class Store:
    def __init__(self,path):
        self.path=path;self.done={}
        if os.path.exists(path):
            for line in open(path):
                try: r=json.loads(line); self.done[r['key']]=r
                except Exception: pass
        self.fh=open(path,'a')
    def add(self,key,r):
        r=dict(r);r['key']=key;self.done[key]=r
        self.fh.write(json.dumps(r)+'\n');self.fh.flush()

def _job(a):
    key,kw=a
    return key,cell(**kw)

def run(path,jobs,workers=8,progress=500):
    st=Store(path)
    todo=[(k,v) for k,v in jobs if k not in st.done]
    print(path,'total',len(jobs),'todo',len(todo),flush=True)
    t0=time.time()
    if todo:
        with Pool(workers) as p:
            for i,(k,r) in enumerate(p.imap_unordered(_job,todo,chunksize=1)):
                st.add(k,r)
                if i%progress==0: print(i,len(todo),round(time.time()-t0),flush=True)
    return st
