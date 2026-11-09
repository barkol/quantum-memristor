# Own SDP for the no-click budget. Pair state CZ(pi)|+>|+>, untrusted k, trusted k+1.
# Robustness: R=min Tr sum rho_l -1 st sum_l D(a|x,l) rho_l >= sigma_a|x (generalized robustness of Cavalcanti-Skrzypczyk).
import numpy as np, cvxpy as cp, itertools, json, sys
from math import log10, sqrt
I2=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],dtype=complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1.,-1]).astype(complex)
plus=np.array([1,1],dtype=complex)/sqrt(2)
CZ=np.diag([1,1,1,-1]).astype(complex)
psi=CZ@np.kron(plus,plus); rho0=np.outer(psi,psi.conj())
SET={3:[X,Y,Z],2:[X,Z]}
def proj(P,a): return (I2+a*P)/2
def cond(rho,P,a):
    # k first factor. returns unnormalized state of k+1
    M=np.kron(proj(P,a),I2); r=M@rho@M
    return r.reshape(2,2,2,2).trace(axis1=0,axis2=2)
def robust(sig):  # sig[x][a] list; a index 0..A-1
    nx=len(sig); na=len(sig[0])
    lams=list(itertools.product(range(na),repeat=nx))
    rho=[cp.Variable((2,2),hermitian=True) for _ in lams]
    cons=[r>>0 for r in rho]
    for x in range(nx):
        for a in range(na):
            S=sum(rho[i] for i,l in enumerate(lams) if l[x]==a)
            cons.append(S-sig[x][a]>>0)
    obj=cp.real(sum(cp.trace(r) for r in rho))-1
    p=cp.Problem(cp.Minimize(obj),cons); p.solve(solver=cp.MOSEK)
    return p.value
def assem(N,V,eps_pre,eps_post,mode,nostate='biased'):
    """mode 'third': outcomes +,-,0; 'random': no-click -> random +-.
    k present at gate w.p. eps_pre; then detected w.p. eps_post. Gate/white visibility V on the present branch.
    absent branch: k+1 stays |+> (biased) or white ('white')."""
    rho=V*rho0+(1-V)*np.eye(4)/4
    rhoB=rho.reshape(2,2,2,2).trace(axis1=0,axis2=2)
    pl=np.outer(plus,plus.conj()) if nostate=='biased' else I2/2
    pl=V*pl+(1-V)*I2/2
    null=eps_pre*(1-eps_post)*rhoB+(1-eps_pre)*pl
    out=[]
    for P in SET[N]:
        sp=eps_pre*eps_post*cond(rho,P,1); sm=eps_pre*eps_post*cond(rho,P,-1)
        if mode=='third': out.append([sp,sm,null])
        else: out.append([sp+null/2,sm+null/2])
    return out
def bis(f,a,b,tol=2e-4):
    # f(eps) >0 steerable, increasing
    fa,fb=f(a),f(b)
    if fb<=1e-7: return None
    while b-a>tol:
        m=(a+b)/2
        if f(m)>1e-7: b=m
        else: a=m
    return (a+b)/2
def closed_third(N,V):
    # W<=concave envelope of sqrt(m) at m=N eps; honest W=N eps V
    best=None
    pts=[(0,0)]+[(m,sqrt(m)) for m in range(1,N+1)]
    for (m0,w0),(m1,w1) in zip(pts,pts[1:]):
        s=(w1-w0)/(m1-m0); d=V-s
        if d<=0: continue
        me=(w0-s*m0)/d
        if m0-1e-12<=me<=m1+1e-12: return me/N
    return None
res={}
print('control R: N=3',robust(assem(3,1,1,1,'random')),'N=2',robust(assem(2,1,1,1,'random')),' expected',2-sqrt(3),3-2*sqrt(2))
for N in (3,2):
  for Vg in (1,.95,.9):
    V=Vg**2
    # A: pure white no-click branch, post-gate loss only
    t3=bis(lambda e:robust(assem(N,V,1,e,'third')),.1,1)
    tr=bis(lambda e:robust(assem(N,V,1,e,'random')),.1,1)
    cl=closed_third(N,V)
    # B: all loss pre-gate: biased no-click state
    t3b=bis(lambda e:robust(assem(N,V,e,1,'third')),.05,1)
    trb=bis(lambda e:robust(assem(N,V,e,1,'random')),.05,1)
    # C: equal split, as text protocol: eps=eta^2, eta_pre=eta_post=eta
    t3c=bis(lambda e:robust(assem(N,V,e**.5,e**.5,'third')),.05,1)
    trc=bis(lambda e:robust(assem(N,V,e**.5,e**.5,'random')),.05,1)
    print(f'N={N} Vg={Vg}: post-only third {t3:.4f} closed {cl:.4f} | random {tr:.4f} (1/(V sqrtN)={1/(V*sqrt(N)):.4f}) | pre-only third {t3b:.4f} random {trb:.4f} | split third {t3c:.4f} random {trc:.4f}')
    sys.stdout.flush()
    res[f'{N}_{Vg}']=dict(third=t3,closed=cl,random=tr,pre_third=t3b,pre_random=trb,split_third=t3c,split_random=trc)
json.dump(res,open('sdp_budget.json','w'),indent=1)
