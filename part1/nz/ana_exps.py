import os
NZ = os.environ.get("NZ_DIR", "results/part1/nz")
import json,glob,numpy as np,os,re
R={os.path.basename(f)[:-5]:json.load(open(f)) for f in glob.glob(f'{NZ}/exps/*.json')}
print(len(R),'done')
def st(n):
    r=R.get(n)
    if not r:return None
    return r
print('=== E1 long time const chi: <theta> MC vs H, at t index; branch out frac')
for n in sorted(k for k in R if k.startswith('E1_')):
    r=R[n];t=np.array(r['ts']);H=r['H'];H2=r.get('H2') or {}
    idx=[np.argmin(abs(t-x)) for x in (10,100,1000,3000) if x<=t[-1]*1.01]
    hist=np.array(r['hist']);
    print(n,'kout=%.3f'%(1-hist[20]),' '.join(f"t={t[i]:.0f}: th {r['th_mc'][i]:.3f}/{H['th'][i] if 'th' in H else np.nan:.3f}/{(H2.get('th') or [np.nan]*len(t))[i]:.3f} |c| {abs(complex(r['c_mc_re'][i],r['c_mc_im'][i])):.3f}/{abs(complex(H['c_re'][i],H['c_im'][i])) if 'c_re' in H else np.nan:.3f}" for i in idx),'maxdc %.3f'%H.get('max_c',np.nan),'maxdth %.3f'%H.get('max_th',np.nan),'H2 maxdc %.3f'%H2.get('max_c',np.nan))
print('=== E2 initial states: max|dc|, max|dth| H (H2) ; final values')
for n in sorted(k for k in R if k.startswith('E2')):
    r=R[n];H=r['H'];H2=r.get('H2') or {}
    hist=np.array(r['hist']);i=-1
    print(n,f"H maxdc {H.get('max_c',np.nan):.3f} maxdth {H.get('max_th',np.nan):.3f} | H2 maxdc {H2.get('max_c',np.nan):.3f} maxdth {H2.get('max_th',np.nan):.3f} | end c MC {r['c_mc_re'][i]:+.3f}{r['c_mc_im'][i]:+.3f}i H {H.get('c_re',[np.nan])[i]:+.3f}{H.get('c_im',[np.nan])[i]:+.3f}i th {r['th_mc'][i]:.3f}/{H.get('th',[np.nan])[i]:.3f} sem {r['sem_y']:.4f}")
print('=== E3 unlocked low eta/gamma')
for n in sorted(k for k in R if k.startswith('E3')):
    r=R[n]
    f=lambda m:(r[m].get('err',np.nan) if r.get(m) else np.nan)
    print(n,f"Amc={r['Amc']:.3e}+-{r['Ase']:.1e} H {f('H'):+.1f}% H2 {f('H2'):+.1f}% MF {f('MF'):+.1f}% GC {f('GC'):+.1f}% cabs {r['cabs']:.3f}")
print('=== E4 unipolar under drive: area errors')
rows=[]
for n in sorted(k for k in R if k.startswith('E4')):
    r=R[n];f=lambda m:(r[m].get('err',np.nan) if r.get(m) else np.nan)
    z=r['H'].get('z',np.nan) if r.get('H') else np.nan
    print(n[3:],f"Amc={r['Amc']:.4f}+-{r['Ase']:.4f} H {f('H'):+.1f}% (z={z:+.0f}) H2 {f('H2'):+.1f}% pH={r['pH']:.3f} var={r['var']:.3f} rmsc={r['H'].get('rms_c',np.nan) if r.get('H') else np.nan:.3f}")
print('=== E5 unipolar kick response d<theta>(tau)/dk: MC (se) vs H at tau ~ .5,2,5,10,30,100; local rate over first e-fold; ')
def rate(t,y):
    y=np.array(y);t=np.array(t);y=y/y[0]
    m=np.where(y<np.exp(-1))[0]
    if len(m)==0:return np.nan
    i=m[0];return 1/t[i]
for n in sorted(k for k in R if k.startswith('E5')):
    r=R[n];t=np.array(r['taus']);ii=[np.argmin(abs(t-x)) for x in (.5,5,30,100)]
    print(n[7:],f"pH mc/H {r['pH_mc']:.4f}/{r['pH_H']:.4f} th {r['th_ss_mc']:.3f}/{r['th_ss_H']:.3f} rate1/e mc {rate(t,r['dth_mc']):.3f} H {rate(t,r['dth_H']):.3f} | ",' '.join(f"{r['dth_mc'][i]/r['dk']:.3f}({r['dth_se'][i]/r['dk']:.3f})/{r['dth_H'][i]/r['dk']:.3f}" for i in ii))
print('=== E6 signed kick');
for n in sorted(k for k in R if k.startswith('E6')):
    r=R[n];t=np.array(r['taus']);ii=[np.argmin(abs(t-x)) for x in (.1,.5,1,2,4)]
    print(n,f"rate1/e mc {rate(t,r['dth_mc']):.3f} H {rate(t,r['dth_H']):.3f} |",' '.join(f"{r['dth_mc'][i]/r['dk']:.4f}({r['dth_se'][i]/r['dk']:.4f})/{r['dth_H'][i]/r['dk']:.4f}" for i in ii))
print('=== E7 stationary signed')
for n in sorted(k for k in R if k.startswith('E7')):
    r=R[n]
    for i in (0,-1):
        c=complex(r['c_mc_re'][i],r['c_mc_im'][i]);H=r['H'];hc=complex(H['c_re'][i],H['c_im'][i]);H2=r['H2'];h2=complex(H2['c_re'][i],H2['c_im'][i])
        print(n,f"t={r['ts'][i]:.0f} |c| MC {abs(c):.5f}(se {r['sem_y']:.5f}) H {abs(hc):.5f} H2 {abs(h2):.5f}; rz MC {c.real:+.5f} H {hc.real:+.5f} H2 {h2.real:+.5f}; th {r['th_mc'][i]:.4f}/{H['th'][i]:.4f}/{H2['th'][i]:.4f} var {r['var_mc'][i]:.4f} kout {1-r['hist'][20]:.3f}")
print('=== E8 high precision');
for n in sorted(k for k in R if k.startswith('E8')):
    r=R[n];print(n,f"Amc={r['Amc']:.5f}+-{r['Ase']:.5f} ({r['Ase']/abs(r['Amc'])*100:.3f}%)",' '.join(f"{m} {r[m]['err']:+.3f}% z={r[m]['z']:+.1f}" for m in('H','H2','Hstat') if r.get(m)))
print('=== E9 long time rerun');
for n in sorted(k for k in R if k.startswith('E9')):
    r=R[n];f=lambda m:(r[m].get('err',np.nan) if r.get(m) else np.nan)
    print(n[3:],f"Ts={r['Ts']:.0f} Amc={r['Amc']:.4f}+-{r['Ase']:.4f} H {f('H'):+.2f}% H2 {f('H2'):+.2f}% th {r['th_mc']:.3f} rmsc {r['H'].get('rms_c',np.nan):.3f}")
