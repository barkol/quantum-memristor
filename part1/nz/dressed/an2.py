import json,numpy as np,pickle,os
from gcl import area_gcl
from mf import area_mf
J=json.load(open('analiza.json'))
t8=round(np.pi/8,4)
C=pickle.load(open('gcl.pkl','rb')) if os.path.exists('gcl.pkl') else {}
def G(d,A,g,w):
    k=(d,A,g,w)
    if k not in C: C[k]=area_gcl(4*d,A,g,w,n=256)/area_mf(.5,A,g,w,n=256)[0]
    return C[k]
out=[]
print("# D A g om | Q se | W1 err% z | W1*[G(A)/G(0.02)] err% z")
for x in J['rows']:
    if abs(x['t0']-t8)>1e-3: continue
    f=G(x['D'],x['A'],x['g'],x['w'])/G(x['D'],0.02,x['g'],x['w'])
    Qc=x['Q1']*f; x['Qc']=Qc; x['zc']=(x['Q']-Qc)/x['sQ']; out.append(x)
    print(f"{x['D']:.3f} {x['A']:.1f} {x['g']:.2f} {x['w']:5.1f} | {x['Q']:.5f} {x['sQ']:.5f} | {100*(x['Q']/x['Q1']-1):+7.3f} z{x['z1']:+7.1f} | {100*(x['Q']/Qc-1):+7.3f} z{x['zc']:+7.1f}")
pickle.dump(C,open('gcl.pkl','wb')); json.dump(out,open('an2.json','w'))
