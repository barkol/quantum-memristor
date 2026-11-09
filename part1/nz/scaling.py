# Deterministic: |A_H-A_H2|/A_H2 on locked orbit as noise-free proxy (H2 validated separately) => order in gamma, Delta
import numpy as np
from nzlib import *
from variants import run_variant
for A,om in [(.2,2.),(1.,2.),(1.,.5)]:
  print('A,om',A,om)
  Tb={}
  for D in [.0125,.025,.05,.1]:
    row=[]
    for g in [.0025,.005,.01,.02,.04,.08,.16]:
        Tp=2*np.pi/om;T=max(10*Tp,40.);ts=np.concatenate(([0.],T+np.arange(256)*Tp/256));x=np.sin(A*np.sin(om*ts[1:]))
        nm=min(int(np.ceil(np.pi/(2*D)))-1,40)
        H=hier(D,.5,g,0.,A,om,nm,np.pi/8,ts,rtol=1e-10,atol=1e-12)
        P=run_variant('H2',D,.5,g,0.,A,om,nm,np.pi/8,ts,rtol=1e-10,atol=1e-12)
        a=shoelace(x,H['c'].real[1:]);b=shoelace(x,P[0].real[1:])
        row.append((a-b)/abs(b)*100)
    Tb[D]=row;print(f' D={D}',' '.join(f'{v:+9.4f}' for v in row),flush=True)
  arr=np.array([Tb[D] for D in Tb]);gs=np.log([.0025,.005,.01,.02,.04,.08,.16]);ds=np.log(list(Tb))
  print(' local order in gamma (D rows):',np.round(np.diff(np.log(np.abs(arr)),axis=1)/np.diff(gs),2).tolist())
  print(' local order in Delta (g cols):',np.round(np.diff(np.log(np.abs(arr)),axis=0)/np.diff(ds)[:,None],2).tolist())
