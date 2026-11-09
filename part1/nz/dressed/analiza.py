import numpy as np, glob, sys, json, os, pickle
sys.path.insert(0,'.')
from analiza_io import read
from mf import area_mf
eta=.5;kappa=2.
cache='mfcache.pkl'; C=pickle.load(open(cache,'rb')) if os.path.exists(cache) else {}
def M(e,A,g,w,t0=np.pi/8):
    k=(round(e,10),A,g,w,round(t0,6))
    if k not in C: C[k]=area_mf(e,A,g,w,theta0=t0,n=512)[0]
    return C[k]
R={}
for f in glob.glob('res/*.bin'):
    d=read(f)
    if d['nb']<1024: continue
    R[(d['Delta'],d['A'],d['gamma'],d['omega'],round(d['theta0'],4))]=d
def stat(x):
    n=len(x); m=x.mean(); s=x.std(ddof=1)/np.sqrt(n)
    ac=np.corrcoef(x[:-1],x[1:])[0,1] if s>0 else 0
    return m,s,ac
rmap={}
t8=round(np.pi/8,4)
for k,d in R.items():
    if k[1]==0.0:
        rS=stat(d['S']); eS=stat(d['E'])
        rmap[(k[0],k[2])]=(rS[0],rS[1],eS[0],eS[1],stat(d['C2'])[0])
print("# r(Delta,gamma) z A=0: Delta gamma r se r_1rz=1-D*q  <eps sin eps> se")
for (De,g),v in sorted(rmap.items()):
    q=kappa/(kappa+g); print(f"{De:.3f} {g:.3f} {v[0]:.6f} {v[1]:.1e} 1-Dq={1-De*q:.5f} 1-Dq-(Dq)^2/2={1-De*q-(De*q)**2/2:.6f} {v[2]:.5f} {v[3]:.1e}")
rows=[]
print("\n# konfiguracje: D A g om | Az/Az_MF +- se | W1 err% z | W4 err% z | rho err% z | lag1")
for k in sorted(R):
    De,A,g,w,t0=k
    if A==0: continue
    d=R[k]; m,s,ac=stat(d['I'])
    if (De,g) not in rmap: continue
    r,rse,e,ese,_=rmap[(De,g)]
    tt=np.pi/8 if t0==t8 else d['theta0']
    amf=M(eta,A,g,w,tt); Q=m/amf; sQ=s/amf
    K=kappa+g
    Q1=r*r*(w*w+K*K)/(w*w+(g+kappa*r)**2)
    Q4=r*M(eta*r,A,g,w,tt)/amf
    dl=4*De; Gam=eta/De
    m2=(Gam*np.sin(dl)-Gam*(1-np.cos(dl))*r+g*e)/kappa
    Qr=m2*(w*w+K*K)/(w*w+(g+kappa*np.sqrt(m2))**2)
    Qr4=np.sqrt(m2)*np.sqrt(m2)/(r if False else 1)  # placeholder bez uzycia
    z1=(Q-Q1)/sQ; z4=(Q-Q4)/sQ; zr=(Q-Qr)/sQ
    rows.append(dict(D=De,A=A,g=g,w=w,t0=t0,Q=Q,sQ=sQ,Q1=Q1,Q4=Q4,Qr=Qr,z1=z1,z4=z4,zr=zr,ac=ac,r=r,area=m,sarea=s,amf=amf))
    if t0==t8: print(f"{De:.3f} {A:.1f} {g:.2f} {w:5.1f} | {Q:.5f} {sQ:.5f} | {100*(Q/Q1-1):+7.3f} z{z1:+7.1f} | {100*(Q/Q4-1):+7.3f} z{z4:+7.1f} | {100*(Q/Qr-1):+7.3f} z{zr:+7.1f} | {ac:+.2f}")
    else: print(f"THETA0={t0} {De:.3f} {A:.1f} {g:.2f} {w:5.1f} | area {m:.5f} {s:.5f} amf(theta0) {amf:.5f} amf(pi/8) {M(eta,A,g,w):.5f} Q {Q:.5f} {sQ:.5f}")
print("\n# ekstrapolacja A->0: Q(A)=Q0+c A^2 (3 punkty, chi2) wzgledem W1(A->0)")
ex=[]
for De in sorted({r_['D'] for r_ in rows}):
  for g in sorted({r_['g'] for r_ in rows}):
    for w in sorted({r_['w'] for r_ in rows}):
      S=[x for x in rows if x['D']==De and x['g']==g and x['w']==w and x['t0']==t8]
      if len(S)<3: continue
      a=np.array([x['A'] for x in S]);Q=np.array([x['Q'] for x in S]);s=np.array([x['sQ'] for x in S])
      Xm=np.c_[np.ones_like(a),a**2]; W=np.diag(1/s**2); cov=np.linalg.inv(Xm.T@W@Xm); bt=cov@Xm.T@W@Q
      chi=((Q-Xm@bt)**2/s**2).sum(); Q1=S[0]['Q1']; sb=np.sqrt(cov[0,0])
      ex.append(dict(D=De,g=g,w=w,Q0=bt[0],sQ0=sb,c=bt[1],sc=np.sqrt(cov[1,1]),Q1=Q1,chi=chi))
      print(f"{De:.3f} {g:.2f} {w:5.1f} Q0={bt[0]:.5f}+-{sb:.5f} c={bt[1]:+.4f}+-{np.sqrt(cov[1,1]):.4f} chi2(1)={chi:5.1f} W1={Q1:.5f} err={100*(bt[0]/Q1-1):+.3f}% z={(bt[0]-Q1)/sb:+.1f}  Qr={S[0]['Qr']:.5f} err={100*(bt[0]/S[0]['Qr']-1):+.3f}%")
pickle.dump(C,open(cache,'wb')); json.dump(dict(rows=[{k:(float(v)) for k,v in x.items()} for x in rows],ex=[{k:float(v) for k,v in x.items()} for x in ex]),open('analiza.json','w'))
