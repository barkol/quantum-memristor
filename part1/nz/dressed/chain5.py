import numpy as np, mpmath as mp
from math import pi,gcd
exec(open('chain.py').read().split("oms=np.array")[0])
# skan gladkosci: wszystkie p/q (q parzyste<=100, p nieparzyste) dla Delta w [0.18,0.32]
oms=np.array([.05,.5,2.0,20.]); bMF=kappa*oms/(oms**2+kappa**2)
rows=[]
for q in range(2,101,2):
    for p in range(1,q,2):
        if gcd(p,q)!=1: continue
        De=2*pi*p/q/4
        if 0.18<=De<=0.32:
            R=resp(p,q,pi/2+0.1,oms); r=np.hypot(R['rc'],R['rs']); Q=R['G'].imag/bMF
            f1=r*r*(oms**2+kappa**2)/(oms**2+(kappa*r)**2)
            rows.append((De,p,q,r,*(100*(Q/f1-1)),R['rc']))
rows.sort()
print("Delta p/q r  err%(.05,.5,2,20) <cos>")
for x in rows: print(f"{x[0]:.4f} {x[1]:3d}/{x[2]:3d} {x[3]:.5f} {x[4]:+7.3f} {x[5]:+7.3f} {x[6]:+7.3f} {x[7]:+7.3f} {x[8]:+.1e}")
