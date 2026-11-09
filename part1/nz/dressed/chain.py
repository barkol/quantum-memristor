#!/usr/bin/env python
# Krok 2: dokladna odpowiedz liniowa A->0 lancucha Markowa na okregu, gamma=0.
# Bez hierarchii i bez symulacji: algebra liniowa na siatce psi_j = a + 2 pi j/q, skok +-p.
import numpy as np, sys, json
from fractions import Fraction
from math import gcd, pi
eta=0.5; kappa=4*eta

def build(p,q,a):
    delta=2*pi*p/q; Delta=delta/4; Gam=eta/Delta
    psi=a+2*pi*np.arange(q)/q
    Wp=Gam*(1+np.cos(psi))/2; Wm=Gam*(1-np.cos(psi))/2
    L=np.zeros((q,q)); L1=np.zeros((q,q))
    for j in range(q):
        jp=(j+p)%q; jm=(j-p)%q
        L[jp,j]+=Wp[j]; L[jm,j]+=Wm[j]; L[j,j]-=Wp[j]+Wm[j]
        dW=Gam*np.sin(psi[j])/2       # dW+/d(2chi) = +, dW-/d(2chi) = -
        L1[jp,j]+=dW; L1[jm,j]-=dW    # calkowity wyplyw bez zmian
    return psi,L,L1,Delta,delta,Gam

def stationary(L):
    q=L.shape[0]; M=L.copy(); M[0,:]=1; b=np.zeros(q); b[0]=1
    return np.linalg.solve(M,b)

def resp(p,q,a,oms):
    psi,L,L1,Delta,delta,Gam=build(p,q,a)
    pi_=stationary(L); r_c=pi_@np.cos(psi); r_s=pi_@np.sin(psi)
    src=L1@pi_; G=[]
    for om in oms:
        x=np.linalg.solve(1j*om*np.eye(q)-L,src)
        G.append(r_s+np.cos(psi)@x)
    return dict(Delta=Delta,delta=delta,Gam=Gam,rc=r_c,rs=r_s,G=np.array(G),psi=psi,L=L,L1=L1,pi=pi_)

def poles(res):
    w,V=np.linalg.eig(res['L']); U=np.linalg.inv(V)
    src=res['L1']@res['pi']; c=np.cos(res['psi'])
    rows=[]
    for n in range(len(w)):
        if abs(w[n])<1e-9: continue
        resid=(c@V[:,n])*(U[n,:]@src)
        # G = r - sum W_n kappa l_n/(i om + kappa l_n), residuum przy 1/(i om - lam) to resid = -W_n kappa l_n
        l=-w[n]/kappa; W=-resid/(kappa*l)
        rows.append((l,W))
    return rows

def best(Dt,qmax=600):
    # p nieparzyste, q parzyste: rownowaga szczegolowa
    bestv=None
    for q in range(2,qmax+1,2):
        p=int(round(4*Dt*q/(2*pi)))
        for pp in (p-1,p,p+1):
            if pp<1 or pp%2==0 or gcd(pp,q)!=1: continue
            e=abs(2*pi*pp/q/4-Dt)
            if bestv is None or e<bestv[0]-1e-15: bestv=(e,pp,q)
    return bestv[1],bestv[2]

def I1I0(x):
    from scipy.special import ive
    return ive(1,x)/ive(0,x)

oms=np.array([0.05,0.2,0.5,1,1.5,2,3,4,6,10,20,50,200])
out={'oms':oms.tolist(),'rows':[]}
print("# (a) r, reguly sum, bieguny. a=pi/2+0.37 ogolny offset")
cases=[('N',1,64),('N',1,32),('N',1,16),('N',1,12),('N',1,10),('N',1,6)]
for Dt in (0.025,0.05,0.1,0.15,0.25):
    cases.append(('Dt',)+best(Dt))
for kind,p,q in cases:
    a=pi/2+0.37*2*pi/q
    R=resp(p,q,a,oms); D=R['Delta']; d=R['delta']; r=np.hypot(R['rc'],R['rs'])
    ser=1-D-D**2/2-D**3/3
    rho2=(np.sin(d)-(1-np.cos(d))*r)/d
    bMF=kappa*oms/(oms**2+kappa**2)
    Qx=R['G'].imag/bMF
    P=poles(R)
    # sortuj po |W|
    P=sorted(P,key=lambda t:-abs(t[1]))
    S0=sum(W for l,W in P); S1=sum(W/l for l,W in P); S2=sum(W*l for l,W in P)
    print(f"\n{kind} p/q={p}/{q} Delta={D:.6f} delta={d:.5f} Gam={R['Gam']:.3f}")
    print(f"  r={r:.10f} <cos psi>={R['rc']:.2e} seria(W2)={ser:.8f} d={r-ser:+.2e} (D^4={D**4:.2e}) vonMises={I1I0(2/d):.8f}")
    print(f"  S0-r={S0.real-r:+.2e} S1-1={S1.real-1:+.2e} S2-rho2={S2.real-rho2:+.2e} rho2={rho2:.8f} r^2={r*r:.8f}")
    print("  bieguny l_n, W_n:",' '.join(f"({l.real:.5f}{'' if abs(l.imag)<1e-9 else f'{l.imag:+.3f}i'},{W.real:+.3e})" for l,W in P[:5]))
    # przewidywania W3
    i1=np.argmin([abs(l.real-r) for l,W in P])
    l1,W1=P[i1]
    print(f"  l1-r={l1.real-r:+.3e} (D^2/3={D*D/3:.3e}); W1-r={W1.real-r:+.3e} (D^2/2={D*D/2:.3e})")
    # bieguny kolo 3
    near3=[(l.real,W.real) for l,W in P if abs(l.real-3)<1.0 and abs(W)>1e-9]
    print("  mod ~3:",near3[:3],f"(prog W3=-D^2/2={-D*D/2:.3e})")
    f1=r*r*(oms**2+kappa**2)/(oms**2+(kappa*r)**2)
    f2=rho2*(oms**2+kappa**2)/(oms**2+kappa**2*rho2)
    print("  om:     "+' '.join(f"{o:8.2f}" for o in oms))
    print("  Q exact "+' '.join(f"{x:8.5f}" for x in Qx))
    print("  Q/W1 -1 "+' '.join(f"{100*(x/y-1):+8.3f}" for x,y in zip(Qx,f1))+"  [%]")
    print("  Q/rho-1 "+' '.join(f"{100*(x/y-1):+8.3f}" for x,y in zip(Qx,f2))+"  [%]")
    out['rows'].append(dict(kind=kind,p=p,q=q,Delta=D,r=r,Q=Qx.tolist(),rho2=rho2,poles=[(float(l.real),float(W.real)) for l,W in P[:8]]))
json.dump(out,open('chain.json','w'),indent=0)

print("\n# (b) zaleznosc od offsetu a siatki i parzystosc q, Delta~0.25 i 0.1, om=2")
for p,q in [best(0.25),best(0.1),(1,6),(1,8),(1,16),(1,7),(1,15),(1,31)]:
    for a in (pi/2,pi/2+pi/q*0.5,pi/2+pi/q):
        R=resp(p,q,a,np.array([2.0])); d=R['delta']
        print(f" p/q={p}/{q} Delta={R['Delta']:.4f} a-pi/2={a-pi/2:.4f} r={np.hypot(R['rc'],R['rs']):.10f} <cos>={R['rc']:+.2e} ImG={R['G'][0].imag:.10f}")
