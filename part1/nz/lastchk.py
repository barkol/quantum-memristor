import os
NZ = os.environ.get("NZ_DIR", "results/part1/nz")
import numpy as np,glob,time,json
from nzlib import *
from variants import run_variant
d=[]
for f in glob.glob(f'{NZ}/grid/*.npz'):
    Z=np.load(f)
    if not bool(Z['settled']):continue
    a=Z['apair'];b=Z['aplus'];nb=len(a);se=np.sqrt((b.var(ddof=1)+a.var(ddof=1))/nb)
    d.append(((b.mean()-a.mean())/se, b.std(ddof=1)/max(a.std(ddof=1),1e-15)))
d=np.array(d);print('pair vs + only: n',len(d),'z rms',np.sqrt(np.mean(d[:,0]**2)),'max|z|',np.abs(d[:,0]).max(),'median noise ratio sd(+)/sd(pair)',np.median(d[:,1]))
ts=np.linspace(0,251,1001)
t0=time.time()
import variants
for args in [(.05,1.,.1,.5),(.1,1.,.1,.5)]:
    D,A,om,g=args;nm=int(np.ceil(np.pi/(2*D)))-1
    from scipy.integrate import solve_ivp
    for T in (20,50,100,251):
        tt=np.linspace(0,T,50);t0=time.time();P=run_variant('H2',D,.5,g,0.,A,om,nm,np.pi/8,tt,tmax=60)
        print(args,'T',T,'H2 ok' if P is not None else 'H2 FAIL',round(time.time()-t0,1),'s',None if P is None else (float(np.abs(P[0]).max()),float(P[1][-1])),flush=True)
        if P is None:break
