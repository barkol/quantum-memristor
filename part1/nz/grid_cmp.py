import os
NZ = os.environ.get("NZ_DIR", "results/part1/nz")
# Deterministic models vs stored MC at identical times. One json per point (resume).
import numpy as np,glob,os,json,sys,time,multiprocessing as mp
from nzlib import *
from variants import run_variant
ETA=.5;TH0=0.;TH_I=np.pi/8
OUT=f'{NZ}/cmp'
def nrule(D): return int(np.ceil(np.pi/(2*D)))-1
def area_pair(x,yp,ym): return .5*(shoelace(x,yp)+shoelace(-x,ym))
def work(fn):
    out=f"{OUT}/{os.path.basename(fn)[:-4]}.json"
    if os.path.exists(out): return out,'skip'
    t0=time.time();Z=np.load(fn)
    D,A,om,g=[float(Z[k]) for k in('D','A','om','g')];Ts=float(Z['Ts']);ts=Z['ts'];x=Z['x']
    tt=np.concatenate(([0.],ts)) # dense output from t=0
    mean=Z['mean'];sem=Z['sem'];ap=Z['apair']
    Amc=ap.mean();Ase=ap.std(ddof=1)/np.sqrt(len(ap))
    r=dict(D=D,A=A,om=om,g=g,Ts=Ts,settled=bool(Z['settled']),N=int(Z['N'])*int(Z['nb']),Amc=Amc,Ase=Ase,Aplus=float(Z['aplus'].mean()),
           kout=float(1-Z['hist'][0,20]),gr=g/(g+4*ETA))
    rule=nrule(D)
    def cmpm(name,cp,cm,thp=None):
        Am=area_pair(x,cp.real,cm.real)
        dyp=cp.real-mean[:,0,0];drx=cp.imag-mean[:,0,1]
        r[name]=dict(A=Am,err=(Am-Amc)/abs(Amc)*100,z=(Am-Amc)/Ase,
            rms=float(np.sqrt(np.mean(dyp**2+drx**2))),zmax=float(np.max(np.abs(dyp/(sem[:,0,0]+1e-300)))),
            dth=None if thp is None else float(np.sqrt(np.mean((thp-mean[:,0,4])**2))))
    for name,nm,cl in [('H',rule,'H'),('Hbig',max(2*rule+1,12),'H'),('MF',1,'MF'),('GC',1,'GC')]:
        Hp=hier(D,ETA,g,TH0,A,om,nm,TH_I,tt,closure=cl,rtol=1e-8,atol=1e-10,tmax=(60. if name=='Hbig' else 300.))
        if Hp is None or 'fail' in Hp: r[name]=dict(fail=Hp);continue
        Hm=hier(D,ETA,g,TH0,A,om,nm,TH_I,tt,closure=cl,sgn=-1.,rtol=1e-8,atol=1e-10,tmax=(60. if name=='Hbig' else 300.))
        if Hm is None or 'fail' in Hm: r[name]=dict(fail=Hm);continue
        cmpm(name,Hp['c'][1:],Hm['c'][1:],Hp['th'][1:]);r[name]['cmax']=float(max(Hp['cmaxabs'].max(),Hm['cmaxabs'].max()))
    for name in ['Hstat','Hdyn','H2']:
        P=run_variant(name,D,ETA,g,TH0,A,om,rule,TH_I,tt);Mn=run_variant(name,D,ETA,g,TH0,A,om,rule,TH_I,tt,sgn=-1.)
        if P is None or Mn is None: r[name]=None;continue
        cmpm(name,P[0][1:],Mn[0][1:],P[1][1:])
    # measured closure residual: cov(theta,e^{i phi}) from MC 
    cov=(mean[:,0,6]+1j*mean[:,0,7])-mean[:,0,4]*(mean[:,0,0]+1j*mean[:,0,1])
    r['covmax']=float(np.abs(cov).max());r['var']=float(np.mean(mean[:,0,5]-mean[:,0,4]**2));r['cabsmean']=float(np.mean(np.hypot(mean[:,0,0],mean[:,0,1])))
    json.dump(r,open(out+'.tmp','w'));os.replace(out+'.tmp',out)
    return out,f'{time.time()-t0:.0f}s'
if __name__=='__main__':
    os.makedirs(OUT,exist_ok=True)
    fs=sorted(glob.glob(f'{NZ}/grid/*.npz'))
    with mp.Pool(int(sys.argv[1])) as p:
        for i,(o,m) in enumerate(p.imap_unordered(work,fs)): print(i,os.path.basename(o),m,flush=True)
