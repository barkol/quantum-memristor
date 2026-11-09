# Bieguny z residuow liczonych calka konturowa (odporne na niemal zdegenerowane pary i nienormalnosc);
# gesta siatka Delta; bledy wzorow; test W3. gamma=0.
import numpy as np
from math import pi,gcd
exec(open('chain.py').read().split("oms=np.array")[0])
def G_s(res,s):
    src=res['L1']@res['pi']; q=len(src)
    return res['rs']+np.cos(res['psi'])@np.linalg.solve(s*np.eye(q)-res['L'],src)
def clusters(res):
    w=np.linalg.eigvals(res['L'])
    l=np.sort((-w/kappa).real); l=l[l>1e-7]
    c=[]
    for x in l:
        if not c or x-c[-1][-1]>2e-3: c.append([x])
        else: c[-1].append(x)
    return [np.mean(x) for x in c]
def residues(res,cl,nmax=6,npts=48):
    out=[]
    for i,c in enumerate(cl[:nmax]):
        gaps=[abs(c-cl[j]) for j in range(len(cl)) if j!=i]
        rad=0.4*min(gaps)*kappa
        tot=0
        for t in 2*pi*(np.arange(npts)+.5)/npts:
            s=-c*kappa+rad*np.exp(1j*t)
            tot+=G_s(res,s)*rad*np.exp(1j*t)/npts   # (1/2 pi i) oint G ds
        out.append((c,(-tot/(kappa*c)).real))   # W_n = -resid/(kappa l)
    return out
print("# Delta l1 W1 | r r^2 rho^2 | l1-r  (D^2/3) W1-r (D^2/2) | l3 W3 (-D^2/2) | reszta S0: r-W1-W3 ; S1: W1/l1+W3/l3-1 ; S2 W1l1+W3l3-rho2")
for N in (64,48,32,24,20,16,14,12,10,8,6):
    p,q=1,N
    a=pi/2+0.37*2*pi/q
    R=resp(p,q,a,np.array([1.0])); D=R['Delta']; d=R['delta']; r=np.hypot(R['rc'],R['rs'])
    rho2=(np.sin(d)-(1-np.cos(d))*r)/d
    cl=clusters(R); rs=residues(R,cl)
    (l1,W1),(l3,W3)=rs[0],rs[1]
    print(f"N={N:3d} D={D:.5f} l1={l1:.6f} W1={W1:.6f} r={r:.6f} r2={r*r:.6f} rho2={rho2:.6f} | l1-r={l1-r:+.3e} ({D*D/3:.3e}) W1-r={W1-r:+.3e} ({D*D/2:.3e}) | l3={l3:.4f} W3={W3:+.3e} ({-D*D/2:.3e}) | rest {[f'{x:.3f}:{y:+.1e}' for x,y in rs[2:4]]}")
print("\n# maks bledy przy gamma=0 po om in 0.01..1000 (log), dla kilku wzorow")
oms=np.logspace(-2,3,61); bMF=kappa*oms/(oms**2+kappa**2)
print("Delta  r   max|W1-err|% (om)  max|rho|% (om) low-om-Q   Qinf/r^2-1%")
for Dt in (0.0125,0.025,0.05,0.075,0.09,0.1,0.11,0.12,0.13,0.15,0.17,0.19,0.2,0.21,0.22,0.23,0.25,0.3):
    p,q=best(Dt,qmax=400)
    R=resp(p,q,pi/2+0.1,oms); D=R['Delta']; d=R['delta']; r=np.hypot(R['rc'],R['rs'])
    rho2=(np.sin(d)-(1-np.cos(d))*r)/d
    Q=R['G'].imag/bMF
    f1=r*r*(oms**2+kappa**2)/(oms**2+(kappa*r)**2)
    f2=rho2*(oms**2+kappa**2)/(oms**2+kappa**2*rho2)
    e1=100*(Q/f1-1); e2=100*(Q/f2-1)
    print(f"{D:.4f} {r:.6f} {np.abs(e1).max():6.3f} ({oms[np.abs(e1).argmax()]:8.2f}) peak-area {e1[np.abs(oms-2).argmin()]:+6.3f} | {np.abs(e2).max():6.3f} ({oms[np.abs(e2).argmax()]:7.2f}) | Q(.01)={Q[0]:.5f} Qinf={Q[-1]:.5f} ")
