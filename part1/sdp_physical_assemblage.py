# Generalized steering robustness of the PHYSICAL random-outcome assemblage vs white assemblage on the same loss axis.
import numpy as np, itertools, cvxpy as cp
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]).astype(complex)
plus=np.array([[.5,.5],[.5,.5]],complex)
def RT(sig,N):
    st=list(itertools.product((0,1),repeat=N))
    v={l:cp.Variable((2,2),hermitian=True) for l in st}
    cons=[v[l]>>0 for l in st]
    for x in range(N):
        for a in (0,1):
            cons.append(sum(v[l] for l in st if l[x]==a)-sig[x][a]>>0)
    p=cp.Problem(cp.Minimize(cp.real(cp.trace(sum(v.values())))-1),cons);p.solve(solver=cp.MOSEK);return max(p.value,0)
def asm(eta,N,phys=True,Vg=1.0):
    # photon k: pre-gate loss prob 1-eta -> k+1 stays |+>; post-gate loss eta(1-eta) -> k+1 white; both surviving eta^2 pure.
    # trusted bases steered: k: X->Z, Z->X, Y->Y; N=3 uses all; N=2 uses X,Z.
    maps=[Z,X,Y][:N]
    out=[]
    for B in maps:
        row=[]
        for s in (1,-1):
            pi=(I+s*B)/2
            if phys:
                rho=eta**2*(Vg**2*pi+(1-Vg**2)*I/2)+(1-eta)*plus+eta*(1-eta)*I/2
            else:
                V=Vg**2*eta**2; rho=V*pi+(1-V)*I/2
            row.append(rho/2)
        out.append(row)
    return out
print('L dB  white_N3 phys_N3 white_N2 phys_N2')
for L in (0,0.5,1.0,1.193,1.5,2.0,2.386,3.0,4.0,6.0,8.0):
    e=10**(-L/10)
    print(f'{L:5.3f}',*[f'{RT(asm(e,N,ph),N):.4f}' for N in (3,) for ph in (False,True)],*[f'{RT(asm(e,N,ph),N):.4f}' for N in (2,) for ph in (False,True)])
# signaling check of phys assemblage: sum_a sigma_a|x independent of x
a=asm(.8,3);print('marginals equal',[np.round(a[x][0]+a[x][1],4).tolist() for x in range(3)][0])
lo,hi=1.193,1.6
for _ in range(12):
    mid=(lo+hi)/2
    if RT(asm(10**(-mid/10),3,True),3)>1e-6: lo=mid
    else: hi=mid
print('phys N=3 random-outcome assemblage: R_T zero at L approx',round(lo,3))
