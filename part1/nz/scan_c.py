# Random search: truncated hierarchy at nmax=rule and rule+1..; max |c_1|, max|c_n|; pure or mixed start; also mode growth in time at fixed params.
import numpy as np,time
from nzlib import *
rng=np.random.default_rng(7);N=0;t0=time.time()
best={}
hits=[]
while time.time()-t0<240:
    D=float(np.exp(rng.uniform(np.log(.01),np.log(1.2))));A=float(rng.choice([0,.2,1,2,4]));om=float(np.exp(rng.uniform(np.log(.05),np.log(10))));g=float(np.exp(rng.uniform(np.log(1e-3),np.log(2))))
    eta=float(np.exp(rng.uniform(np.log(.01),np.log(1.))));uni=bool(rng.integers(2))
    rule=max(int(np.ceil(np.pi/(2*D)))-1,1);rule=min(rule,60);th=float(rng.uniform(0,np.pi));tr0=float(rng.choice([0,rng.uniform(0,1)]))
    ts=np.linspace(0,min(100,4*2*np.pi/om+20),400)
    H=hier(D,eta,g,tr0,A,om,max(rule,2),th,ts,uni=uni,rtol=1e-8,atol=1e-10,tmax=6.)
    if H is None: continue
    if 'fail' in H: hits.append(('fail',D,A,om,g,eta,uni,H));continue
    N+=1;v=H['cmaxabs'][1:].max()
    if v>1+1e-5: hits.append((float(v),round(D,4),A,round(om,3),round(g,4),round(eta,4),uni,rule))
hits_num=[h for h in hits if h[0]!='fail']
print('runs',N,'|c_n|>1 hits',len(hits_num),'fails',len(hits)-len(hits_num))
print(sorted(hits_num,key=lambda h:-h[0])[:10]);print([h for h in hits if h[0]=='fail'][:6])
print('time domain: rule+1 blow-up time at constant chi, gamma=.01, start pure phi=pi/2')
for D,nm in [(.025,63),(.05,32),(.1,16),(.25,7),(.05,63),(.04,120),(.2,24)]:
    H=hier(D,.5,.01,0.,0.,1.,nm,np.pi/8,np.linspace(0,600,301),rtol=1e-9,atol=1e-11,tmax=40.,blow=1.0001e0*2)
    if 'fail' in H: print(D,nm,H)
    else: print(D,nm,'no blow to t=600, max|c_n|',H['cmaxabs'].max())
