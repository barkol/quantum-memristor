# Random search: truncated hierarchy at nmax=rule and rule+1..; max |c_1|, max|c_n|; pure or mixed start; also mode growth in time at fixed params.
import numpy as np,time
from nzlib import *
rng=np.random.default_rng(7);N=0;t0=time.time()
best={}
hits=[];C=[]
while time.time()-t0<150:
    D=float(np.exp(rng.uniform(np.log(.01),np.log(1.2))));A=float(rng.choice([0,.2,1,2,4]));om=float(np.exp(rng.uniform(np.log(.05),np.log(10))));g=float(np.exp(rng.uniform(np.log(1e-3),np.log(2))))
    eta=float(np.exp(rng.uniform(np.log(.01),np.log(1.))));uni=bool(rng.integers(2))
    rule=max(int(np.ceil(np.pi/(2*D)))-1,1);rule=min(rule,60);th=float(rng.uniform(0,np.pi));tr0=float(rng.choice([0,rng.uniform(0,1)]))
    ts=np.linspace(0,min(100,4*2*np.pi/om+20),400)
    H=hier(D,eta,g,tr0,A,om,max(rule,2),th,ts,uni=uni,rtol=1e-8,atol=1e-10,tmax=6.)
    if H is None: continue
    if 'fail' in H: hits.append(('fail',D,A,om,g,eta,uni,H));continue
    N+=1;v=H['cmaxabs'][1:].max();v1=np.abs(H['c']).max();C.append(v1)
    if v>1+1e-5: hits.append((float(v),float(v1),round(D,4),A,round(om,3),round(g,4),round(eta,4),uni,rule))
hits_num=[h for h in hits if h[0]!='fail']
print('runs',N,'|c_n|>1 hits',len(hits_num),'fails',len(hits)-len(hits_num))
print(sorted(hits_num,key=lambda h:-h[0])[:10]);print([h for h in hits if h[0]=='fail'][:6])
print('max|c1| overall',max(C),' fraction >1+1e-5:',np.mean(np.array(C)>1+1e-5))
