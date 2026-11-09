import os
NZ = os.environ.get("NZ_DIR", "results/part1/nz")
# Break-the-equation experiments. Each job writes json in exps/ (resume); MC error from 16 batches.
import numpy as np,os,sys,json,time,multiprocessing as mp
from nzlib import *
from variants import run_variant
OUT=f'{NZ}/exps'
def nrule(D): return max(int(np.ceil(np.pi/(2*D)))-1,2)
def mcb(N,D,eta,g,th0,A,om,uni,ti,ts,sd,nb=16,phi_ref=np.pi/2):
    R=[];H=[]
    for b in range(nb):
        o,h,_=mc(N,D,eta,g,th0,A,om,uni,ti,ts,sd+b,phi_ref);R.append(o);H.append(h)
    R=np.array(R);return R.mean(0),R.std(0,ddof=1)/np.sqrt(nb),np.mean(H,0)
def cmpstate(name,D,eta,g,th0,A,om,uni,ti,ts,N,sd,models=('H','H2')):
    m,s,h=mcb(N,D,eta,g,th0,A,om,uni,ti,ts,sd)
    cM=m[:,0,0]+1j*m[:,0,1];thM=m[:,0,4]
    res=dict(name=name,D=D,eta=eta,g=g,th0=th0,A=A,om=om,uni=uni,ts=list(ts),c_mc_re=list(cM.real),c_mc_im=list(cM.imag),th_mc=list(thM),var_mc=list(m[:,0,5]-thM**2),sem_y=float(s[:,0,0].mean()),sem_th=float(s[:,0,4].mean()),hist=list(h[0]))
    nm=nrule(D);tt=np.concatenate(([0.],ts))
    for mdl in models:
        if mdl=='H':
            H=hier(D,eta,g,th0,A,om,nm,ti,tt,uni=uni,rtol=1e-8,atol=1e-10)
            if H is None or 'fail' in H: res[mdl]=dict(fail=H);continue
            c=H['c'][1:];th=H['th'][1:];res['cmax_H']=float(H['cmaxabs'].max())
        else:
            P=run_variant(mdl,D,eta,g,th0,A,om,nm,ti,tt,uni=uni,tmax=500)
            if P is None: res[mdl]=None;continue
            c=P[0][1:];th=P[1][1:]
        res[mdl]=dict(c_re=list(c.real),c_im=list(c.imag),th=list(th),rms_c=float(np.sqrt(np.mean(np.abs(c-cM)**2))),max_c=float(np.abs(c-cM).max()),rms_th=float(np.sqrt(np.mean((th-thM)**2))),max_th=float(np.abs(th-thM).max()),cmax=float(np.abs(c).max()))
    return res
def job(spec):
    name=spec['name'];fn=f'{OUT}/{name}.json'
    if os.path.exists(fn):return name,'skip'
    t0=time.time();k=spec['kind']
    if k=='state':
        r=cmpstate(name,spec['D'],spec.get('eta',.5),spec['g'],spec.get('th0',0.),spec.get('A',0.),spec.get('om',1.),spec.get('uni',False),np.array(spec['ti']),np.array(spec['ts']),spec['N'],spec['sd'])
    elif k=='area':
        D,eta,g,th0,A,om,uni=[spec.get(q,dflt) for q,dflt in (('D',0),('eta',.5),('g',.01),('th0',0.),('A',1.),('om',2.),('uni',False))]
        ti=np.array(spec['ti']);Tp=2*np.pi/om;M=256;Ts=spec['Ts'];ts=Ts+np.arange(M)*Tp/M;x=np.sin(A*np.sin(om*ts))
        nb=16;AP=[];O=[]
        for b in range(nb):
            o,h,_=mc(spec['N'],D,eta,g,th0,A,om,uni,ti,ts,spec['sd']+b);O.append(o);AP.append(.5*(shoelace(x,o[:,0,0])+shoelace(-x,o[:,1,0])))
        O=np.array(O);AP=np.array(AP);r=dict(name=name,D=D,eta=eta,g=g,th0=th0,A=A,om=om,uni=uni,Ts=Ts,Amc=AP.mean(),Ase=AP.std(ddof=1)/4,cabs=float(np.mean(np.hypot(O.mean(0)[:,0,0],O.mean(0)[:,0,1]))),th_mc=float(O.mean(0)[:,0,4].mean()),var=float((O.mean(0)[:,0,5]-O.mean(0)[:,0,4]**2).mean()),pH=float((1+O.mean(0)[:,0,0]).mean()/2))
        tt=np.concatenate(([0.],ts));nm=nrule(D)
        for mdl in spec.get('models',['H','H2','MF','GC']):
            if mdl in('H','MF','GC'):
                nmm=nm if mdl=='H' else 1
                if mdl!='H' and uni: pass
                P=hier(D,eta,g,th0,A,om,nmm,ti,tt,uni=uni,closure=mdl,rtol=1e-8,atol=1e-10);Mm=hier(D,eta,g,th0,A,om,nmm,ti,tt,uni=uni,closure=mdl,sgn=-1.,rtol=1e-8,atol=1e-10)
                if P is None or Mm is None or 'fail' in P or 'fail' in Mm:r[mdl]=dict(fail=[P if P is None or 'fail' in P else 'ok',Mm if Mm is None or 'fail' in Mm else 'ok']);continue
                cp=P['c'][1:];cm=Mm['c'][1:];cmax=float(P['cmaxabs'].max())
            else:
                P=run_variant(mdl,D,eta,g,th0,A,om,nm,ti,tt,uni=uni,tmax=500);Mm=run_variant(mdl,D,eta,g,th0,A,om,nm,ti,tt,uni=uni,sgn=-1.,tmax=500)
                if P is None or Mm is None:r[mdl]=None;continue
                cp=P[0][1:];cm=Mm[0][1:];cmax=float(np.abs(cp).max())
            a=.5*(shoelace(x,cp.real)+shoelace(-x,cm.real))
            r[mdl]=dict(A=a,err=(a-r['Amc'])/abs(r['Amc'])*100,z=(a-r['Amc'])/r['Ase'],rms_c=float(np.sqrt(np.mean(np.abs(cp-(O.mean(0)[:,0,0]+1j*O.mean(0)[:,0,1]))**2))),cmax=cmax)
    elif k=='kick':
        D,eta,g,th0,uni,tk,dk=[spec[q] for q in('D','eta','g','th0','uni','tk','dk')]
        ti=spec['ti'];ts=tk+np.array(spec['taus']);R=[]
        for b in range(16): 
            o=mc_kick(spec['N'],D,eta,g,th0,0.,1.,uni,ti,tk,dk,ts,spec['sd']+b);R.append(o)
        R=np.array(R);dth=(R[:,:,1,4]-R[:,:,0,4]);dth_m=dth.mean(0);dth_s=dth.std(0,ddof=1)/4
        nm=nrule(D);f=hier_rhs_factory(D,eta,g,th0,0.,1.,nm,uni)
        y0=cn_init(ti,th0,nm);s=solve_ivp(f,(0,tk),y0,method='DOP853',rtol=1e-9,atol=1e-11);y=s.y[:,-1].copy()
        c=y[1:nm+1]+1j*y[nm+1:];n=np.arange(1,nm+1);ck=c*np.exp(1j*n*4*dk);yk=np.concatenate(([y[0]+dk],ck.real,ck.imag))
        s0=solve_ivp(f,(0,ts[-1]-tk),y,t_eval=ts-tk,method='DOP853',rtol=1e-9,atol=1e-11);s1=solve_ivp(f,(0,ts[-1]-tk),yk,t_eval=ts-tk,method='DOP853',rtol=1e-9,atol=1e-11)
        dH=s1.y[0]-s0.y[0]
        r=dict(name=name,D=D,eta=eta,g=g,th0=th0,uni=uni,dk=dk,taus=spec['taus'],dth_mc=list(dth_m),dth_se=list(dth_s),dth_H=list(dH),th_ss_H=float(y[0]+th0),th_ss_mc=float(R[:,0,0,4].mean()),pH_mc=float((1+R[:,0,0,0].mean())/2),pH_H=float((1+c[0].real)/2))
    json.dump(r,open(fn+'.tmp','w'));os.replace(fn+'.tmp',fn)
    return name,f'{time.time()-t0:.0f}s'
def specs():
    S=[];sd=10**6
    def add(**k):
        nonlocal sd;k['sd']=sd;sd+=100;S.append(k)
    logt=lambda T,n=40:list(np.geomspace(.05,T,n))
    pi=np.pi
    # E1 long-time hopping at constant chi from a locked start
    for D,T,N in [(.1,3000,3000),(.25,3000,8000),(.5,3000,20000)]:
        for g in [.001,.01,.1]:
            add(kind='state',name=f'E1_D{D}_g{g}',D=D,g=g,ti=[pi/8],ts=logt(T),N=N)
    for D in [.025,.05,.1,.25]:
        add(kind='state',name=f'E1_D{D}_g0.5',D=D,g=.5,ti=[pi/8],ts=logt(300),N=int(1e8/(.5/D*300)))
    # E2 initial states far from locked
    for D in [.025,.1]:
        N=20000 if D==.025 else 40000
        for g in [.01,.1]:
            add(kind='state',name=f'E2sep_D{D}_g{g}',D=D,g=g,ti=[3*pi/8],ts=logt(60),N=N)                      # on unstable point
            add(kind='state',name=f'E2sepoff_D{D}_g{g}',D=D,g=g,ti=[3*pi/8+.02],ts=logt(60),N=N)
            add(kind='state',name=f'E2mix1_D{D}_g{g}',D=D,g=g,ti=[pi/8,pi/8+pi/2],ts=logt(200),N=N)           # two adjacent branches
            add(kind='state',name=f'E2mixfar_D{D}_g{g}',D=D,g=g,ti=[pi/8+2*pi,pi/8+4*pi+pi/2],ts=logt(200),N=N)
            add(kind='state',name=f'E2unif_D{D}_g{g}',D=D,g=g,ti=list(np.linspace(0,pi/2,64,endpoint=False)),ts=logt(200),N=N) # uniform on one period
            add(kind='state',name=f'E2gau_D{D}_g{g}',D=D,g=g,ti=list(pi/8+0.3*np.random.default_rng(1).normal(size=2048)),ts=logt(60),N=N)
    # E2b lock edge: gamma theta/eta -> 1
    for frac in [.5,.9,.98,1.1,2.]:
        g=.01;D=.05;kk=round(frac*.5/g/(pi/2));th=pi/8+kk*pi/2
        add(kind='state',name=f'E2edge_f{frac}',D=D,g=g,ti=[th],ts=logt(400),N=20000)
    # E3 unlocked: small eta/gamma
    for eta,g in [(.05,.1),(.05,.5),(.01,.1),(.01,.5),(.002,.1),(.05,.01),(.01,.01)]:
        for D in [.01,.05,.2]:
            for A,om in [(1.,.5),(1.,2.)]:
                if eta/D<.01: continue
                add(kind='area',name=f'E3_eta{eta}_g{g}_D{D}_A{A}_w{om}',eta=eta,D=D,g=g,A=A,om=om,ti=[pi/8],Ts=max(3*2*pi/om,6/g),N=int(min(1e6,1e8/max(eta/D*max(3*2*pi/om,6/g)+256,300))))
    # E4 unipolar under drive, area + state
    for th0 in [0.,pi/16,pi/4-.01,pi/4]:
        for D,g in [(.025,.01),(.05,.01),(.1,.01),(.05,.05)]:
            for A,om in [(.2,.5),(.2,2.),(1.,.5),(1.,2.)]:
                Ts=max(3*2*pi/om,8/g)
                add(kind='area',name=f'E4_th{th0:.4f}_D{D}_g{g}_A{A}_w{om}',uni=True,D=D,eta=.5,g=g,th0=th0,A=A,om=om,ti=[pi/4-.1],Ts=Ts,N=int(1.2e8/(.5/D*Ts+256)),models=['H','H2'])
    # E5 unipolar const chi + kick response, near-dark (paper: th0=pi/16, eta/g=50)
    for th0,g in [(pi/16,.01),(pi/4-.05,.01),(pi/4-.01,.01),(0.,.01),(pi/16,.1)]:
        for D in [.01,.025,.05,.1]:
            taus=list(np.geomspace(.05,300,30))
            for dk in [.02,.05]:
                add(kind='kick',name=f'E5kick_th{th0:.4f}_g{g}_D{D}_dk{dk}',D=D,eta=.5,g=g,th0=th0,uni=True,tk=800.,dk=dk,ti=pi/4-.08,taus=taus,N=int(1.5e8/(.5/D*1100)))
    # E6 signed kick response: T5 lambda1
    for D in [.025,.05,.1,.25,.5]:
        add(kind='kick',name=f'E6kick_D{D}',D=D,eta=.5,g=.01,th0=0.,uni=False,tk=40.,dk=.02,ti=pi/8,taus=list(np.geomspace(.02,8,30)),N=int(2e8/(.5/D*50)))
    # E7 T5 stationary, signed, high stats on settled locked state
    for D in [.025,.05,.1,.25,.5]:
        add(kind='state',name=f'E7ss_D{D}',D=D,g=.01,ti=[pi/8],ts=[20.,40.,60.,80.,100.],N=int(1e9/(.5/D*100)/16))
    # E8 high precision at flagged low-gamma points
    for D,A,om in [(.1,.2,6.),(.025,2.,6.),(.1,2.,6.),(.05,2.,2.)]:
        Ts=20.;c=.5/D*(Ts+2*pi/om)+256
        add(kind='area',name=f'E8_D{D}_A{A}_w{om}',D=D,g=.01,A=A,om=om,ti=[pi/8],Ts=Ts,N=int(7.5e8/c),models=['H','H2','Hstat'])
    # E9 long-time rerun of points not settled on the first pass
    import glob
    for f in glob.glob(f'{NZ}/cmp/*.json'):
        r=json.load(open(f))
        if r['settled'] or r['om']==.1: continue
        Tp=2*pi/r['om'];Ts=max(8*Tp,100.);c=.5/r['D']*(Ts+Tp)+256
        add(kind='area',name=f"E9_D{r['D']}_A{r['A']}_w{r['om']}_g{r['g']}",D=r['D'],g=r['g'],A=r['A'],om=r['om'],ti=[pi/8],Ts=Ts,N=int(6e7/c),models=['H','H2'])
    return S
if __name__=='__main__':
    os.makedirs(OUT,exist_ok=True)
    S=specs();np.random.default_rng(2).shuffle(S)
    print(len(S),'jobs',flush=True)
    with mp.Pool(int(sys.argv[1]),maxtasksperchild=20) as p:
        for i,(n,m) in enumerate(p.imap_unordered(job,S)):print(i,n,m,flush=True)
