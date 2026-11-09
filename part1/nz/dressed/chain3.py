# Test uwarunkowania: double vs mpmath 60 cyfr dla tych samych lancuchow
import numpy as np, mpmath as mp
from math import pi
exec(open('chain.py').read().split("oms=np.array")[0])
mp.mp.dps=60
def resp_mp(p,q,a,om):
    PI=mp.pi; delta=2*PI*p/q; Delta=delta/4; Gam=mp.mpf(eta)/Delta
    psi=[a+2*PI*j/q for j in range(q)]
    L=mp.zeros(q,q); L1=mp.zeros(q,q)
    for j in range(q):
        jp=(j+p)%q; jm=(j-p)%q; c=mp.cos(psi[j]); s=mp.sin(psi[j])
        L[jp,j]+=Gam*(1+c)/2; L[jm,j]+=Gam*(1-c)/2; L[j,j]-=Gam
        L1[jp,j]+=Gam*s/2; L1[jm,j]-=Gam*s/2
    M=L.copy()
    for j in range(q): M[0,j]=1
    b=mp.zeros(q,1); b[0]=1
    P=mp.lu_solve(M,b)
    rc=sum(P[j]*mp.cos(psi[j]) for j in range(q)); rs=sum(P[j]*mp.sin(psi[j]) for j in range(q))
    src=L1*P
    A=mp.mpc(0,om)*mp.eye(q)-L
    x=mp.lu_solve(A,src)
    return Delta,rc,rs,rs+sum(mp.cos(psi[j])*x[j] for j in range(q))
for p,q in [(1,6),(7,44),(15,94),(29,182),(11,70),(1,8),(13,102)]:
  for om in (0.5,2.0):
    a=pi/2+0.1
    R=resp(p,q,a,np.array([om]))
    D,rc,rs,G=resp_mp(p,q,mp.mpf(a),om)
    print(f"p/q={p}/{q} D={float(D):.5f} om={om} r_mp={float(rs):.9f} r_dbl={R['rs']:.9f} ImG_mp={float(G.imag):.9f} ImG_dbl={R['G'][0].imag:.9f} Q={float(G.imag)/(kappa*om/(om**2+kappa**2)):.6f}",flush=True)
