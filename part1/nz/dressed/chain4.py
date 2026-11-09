import numpy as np, mpmath as mp
from math import pi
exec(open('chain.py').read().split("oms=np.array")[0])
from scipy.special import ive
# (a) r z ulamka lancuchowego (zamkniete przez tan(k pi/N)=inf przy k=N/2), kontrola z lancuchem; wspolczynnik Delta^4
mp.mp.dps=80
def r_cf(N):
    x=mp.mpf(0)
    for k in range(N//2-1,0,-1):
        x=1/(2*mp.tan(k*mp.pi/N)+x)
    return x
print("# r(N) z CF vs lancuch; (r-(1-D-D^2/2-D^3/3))/D^4")
for N in (4,6,8,16,32,64,128,256,512,1024,2048):
    D=mp.pi/(2*N); rr=r_cf(N)
    extra=''
    if N<=64:
        R=resp(1,N,pi/2+0.3*2*pi/N,np.array([1.0])); extra=f" chain={np.hypot(R['rc'],R['rs']):.12f}"
    print(N,float(D),mp.nstr(rr,14),mp.nstr((rr-(1-D-D**2/2-D**3/3))/D**4,8),extra)
print("\n# zamkniete przyblizenia r: blad wzgledem CF")
for N in (64,32,16,12,10,8,6,4):
    D=pi/(2*N); rr=float(r_cf(N)); d=4*D
    vm=ive(1,2/d)/ive(0,2/d); vm2=ive(1,1/np.sin(d/2))/ive(0,1/np.sin(d/2))
    print(f"D={D:.5f} r={rr:.8f} 1-D={1-D-rr:+.2e} 1-D-D2/2={1-D-D*D/2-rr:+.2e} ser3={1-D-D*D/2-D**3/3-rr:+.2e} 1+ln(1-D)={1+np.log(1-D)-rr:+.2e} vM(2/d)={vm-rr:+.2e} vM(1/sin(d/2))={vm2-rr:+.2e}")
# (b) gesty grid Q, bledy W1 przy om=.5,2,6,20 i maks
print("\n# gamma=0, q<=200: Delta, r, Q(om)/W1-1 [%] dla om=0.05,.5,1,2,3,6,20,1000; max po om>=0.5; max po om in[0.05,1000]")
oms=np.array([0.05,.2,.5,1,1.5,2,3,4,6,10,20,50,1000.]); bMF=kappa*oms/(oms**2+kappa**2); iw=[0,2,3,5,6,8,10,12]
for Dt in (0.0125,0.025,0.05,0.075,0.09,0.1,0.11,0.12,0.13,0.15,0.165,0.18,0.19,0.2,0.21,0.22,0.23,0.25,0.3):
    p,q=best(Dt,qmax=200)
    R=resp(p,q,pi/2+0.1,oms); De=R['Delta']; d=R['delta']; r=np.hypot(R['rc'],R['rs'])
    Q=R['G'].imag/bMF
    f1=r*r*(oms**2+kappa**2)/(oms**2+(kappa*r)**2); e=100*(Q/f1-1)
    rho2=(np.sin(d)-(1-np.cos(d))*r)/d; f2=rho2*(oms**2+kappa**2)/(oms**2+kappa**2*rho2); e2=100*(Q/f2-1)
    print(f"{De:.4f} q={q:3d} r={r:.6f} "+' '.join(f"{e[i]:+6.3f}" for i in iw)+f" | max>=.5 {np.abs(e[2:]).max():.3f} all {np.abs(e).max():.3f} | rho: {np.abs(e2[2:]).max():.3f}")
# (c) Q(0): S1
print("\n# Q(om->0) przy pierscieniu N")
for N in (6,8,10,12,14,16,20):
    R=resp(1,N,pi/2+.1,np.array([1e-4,1e-3])); Q=R['G'].imag/(kappa*np.array([1e-4,1e-3])/(np.array([1e-4,1e-3])**2+kappa**2)); print(N,R['Delta'],Q, 'ln(1-Q)*D',np.log(1-Q[0])*R['Delta'])
