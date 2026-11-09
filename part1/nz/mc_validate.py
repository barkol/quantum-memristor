import numpy as np,time,sys
from nzlib import *
# 1. gamma=0, driven: hierarchy is claimed exact. 2. CRN pair unbiased. 
def run(Delta,A,om,gam,N=100000,per=3,M=64,nb=16,nmax=None,uni=False,th_i=np.pi/8,th0=0.0,eta=.5):
    Tp=2*np.pi/om
    ts=(np.arange(per*M)+1)*Tp/M
    ress=[];t0=time.time();nev=0
    for b in range(nb):
        o,h,e=mc(N,Delta,eta,gam,th0,A,om,uni,th_i,ts,1000+b);ress.append(o);nev+=e
    ress=np.array(ress) # nb,ns,2,NC
    print(f'  Delta={Delta} A={A} om={om} gam={gam}: {nev/(time.time()-t0)/1e6:.1f} M pair events/s')
    if nmax is None: nmax=int(np.ceil(np.pi/(2*Delta)))-1
    H=hier(Delta,eta,gam,th0,A,om,nmax,th_i,ts,uni=uni)
    m=ress.mean(0);se=ress.std(0,ddof=1)/np.sqrt(nb)
    z=(m[:,0,0]-H['c'].real)/se[:,0,0]; z2=(m[:,0,1]-H['c'].imag)/se[:,0,1]; zt=(m[:,0,4]-H['th'])/se[:,0,4]
    print(f'  nmax={nmax}: y:rms z {np.sqrt(np.mean(z**2)):.2f} max {np.abs(z).max():.2f}, ry: {np.sqrt(np.mean(z2**2)):.2f},{np.abs(z2).max():.2f}; theta: {np.sqrt(np.mean(zt**2)):.2f},{np.abs(zt).max():.2f}; se y {se[:,0,0].mean():.2e}; max|y diff| {np.abs(m[:,0,0]-H["c"].real).max():.2e}')
    return m,H
print('gamma=0 exactness (expect z~1):')
for D,A,om in [(0.025,1,2),(0.1,1,2),(0.1,2,6),(0.25,1,0.5),(0.5,1,2)]:
    run(D,A,om,0.0)
print('with large nmax at Delta=0.25, 0.5 (expect z~1 if exact & converged):')
for D,A,om,nm in [(0.25,1,0.5,40),(0.5,1,2,40),(0.5,1,2,12)]:
    run(D,A,om,0.0,nmax=nm)
print('unipolar gamma=0')
for D,A,om in [(0.1,1,2),(0.25,1,2)]:
    run(D,A,om,0.0,uni=True,th_i=0.3)
