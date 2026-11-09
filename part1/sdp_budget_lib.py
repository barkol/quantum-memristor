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
