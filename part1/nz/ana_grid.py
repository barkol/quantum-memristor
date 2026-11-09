import os
NZ = os.environ.get("NZ_DIR", "results/part1/nz")
import json,glob,numpy as np,sys
R=[json.load(open(f)) for f in glob.glob(f'{NZ}/cmp/*.json')]
print(len(R),'points')
DS=[.025,.05,.1,.25,.5];GS=[.001,.01,.1,.5]
def e(r,m):
    v=r.get(m)
    if not v or 'err' not in v: return np.nan
    return v['err']
def lock(r): return r['A']*r['om']<1.9
print('rel. SEM of MC area: median %.2e, 90%% %.2e, max %.2e'%tuple(np.percentile([r['Ase']/abs(r['Amc']) for r in R],[50,90,100])))
print('settled fraction',np.mean([r['settled'] for r in R]))
def table(m,sel,lab):
    print(f'--- {lab} model {m}: max|err%| (rows D, cols g={GS}) ')
    for D in DS:
        row=[]
        for g in GS:
            v=[abs(e(r,m)) for r in R if r['D']==D and r['g']==g and sel(r)]
            row.append(np.nanmax(v) if len(v) and np.any(~np.isnan(v)) else np.nan)
        print(f'  D={D:<6}',' '.join(f'{x:8.2f}' for x in row))
for m in ['H','H2','Hdyn','Hstat','GC','MF','Hbig']:
    table(m,lock,'LOCKED Aw<1.9')
    table(m,lambda r:not lock(r),'UNLOCKED Aw>=1.9')
print('--- worst per Delta for H, listing')
for D in DS:
    rr=sorted([r for r in R if r['D']==D],key=lambda r:-abs(e(r,'H')))[:5]
    for r in rr: print(f"  D={D} A={r['A']} w={r['om']} g={r['g']}: Amc={r['Amc']:.5f}+-{r['Ase']:.5f} H {e(r,'H'):+.2f}% z={r['H']['z']:+.1f} H2 {e(r,'H2'):+.2f}% Hbig {e(r,'Hbig'):+.2f}% MF {e(r,'MF'):+.2f}% GC {e(r,'GC'):+.2f}% kout={r['kout']:.3f} rmsc={r['H']['rms']:.2e} settled={r['settled']}")
print('--- significance: g<=0.01, D<=0.1, |z|>3 for H')
sig=[r for r in R if r['g']<=.01 and r['D']<=.1 and abs(r['H']['z'])>3]
print(len(sig),'of',len([r for r in R if r['g']<=.01 and r['D']<=.1]))
for r in sorted(sig,key=lambda r:-abs(r['H']['z']))[:14]:
    print(f"  D={r['D']} A={r['A']} w={r['om']} g={r['g']}: {e(r,'H'):+.3f}% z={r['H']['z']:+.1f} se={r['Ase']/abs(r['Amc'])*100:.3f}% cov={r['covmax']:.2e}")
print('--- scaling: locked, |z|>4: log|err| = a + p log(gr)+ q log D')
sel=[r for r in R if lock(r) and abs(r['H']['z'])>4 and r['D']<=.25]
X=np.array([[1,np.log(r['gr']),np.log(r['D'])] for r in sel]);y=np.log([abs(e(r,'H')) for r in sel])
if len(sel)>4:
    b,res,_,_=np.linalg.lstsq(X,y,rcond=None);print(' n',len(sel),'coef',b,'resid rms',np.std(y-X@b))
print('--- slices (A=1,w=2) err H per (D,g); and H2')
for m in ['H','H2','Hdyn']:
  for A,om in [(1.,2.),(.2,2.),(1.,.5),(2.,.5)]:
    print(m,A,om,[ (D,[round(e(r,m),2) for g in GS for r in R if r['D']==D and r['g']==g and r['A']==A and r['om']==om]) for D in DS])
print('--- rms c error relative to |c| ; H')
for D in DS: print(D,[ (g,round(max(r['H']['rms'] for r in R if r['D']==D and r['g']==g and lock(r)),4),round(max(r['H']['rms'] for r in R if r['D']==D and r['g']==g and not lock(r)),4)) for g in GS])
print('--- Hbig failures',sum(1 for r in R if r.get('Hbig') and 'fail' in r['Hbig']),'; cmax H max',max(r['H'].get('cmax',0) for r in R),'; Hbig max',max(r['Hbig'].get('cmax',0) for r in R if r.get('Hbig')))
fails=[(r['D'],r['A'],r['om'],r['g'],r['Hbig']['fail']) for r in R if r.get('Hbig') and 'fail' in r['Hbig']];print(fails[:20])
for m in ['H2','Hdyn','Hstat','H']:
    print(m,'None',sum(1 for r in R if not r.get(m) or 'err' not in r[m]),[ (r['D'],r['A'],r['om'],r['g']) for r in R if not r.get(m) or 'err' not in r[m]][:8])
print('--- H cmax>1 points');print([(r['D'],r['A'],r['om'],r['g'],round(r['H']['cmax'],4)) for r in R if r['H'].get('cmax',0)>1.0001][:20])
print('--- kout>0.1 locked');print([(r['D'],r['A'],r['om'],r['g'],round(r['kout'],2)) for r in R if lock(r) and r['kout']>.1][:40])
