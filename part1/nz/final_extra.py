import os
NZ = os.environ.get("NZ_DIR", "results/part1/nz")
import numpy as np,glob,json
from nzlib import *
from scipy.integrate import solve_ivp
print('=== scalar mean-field theta law (the paper MF, |c|=1) vs MC, all grid')
rows=[]
for f in glob.glob(f'{NZ}/grid/*.npz'):
    Z=np.load(f);D,A,om,g=[float(Z[k]) for k in('D','A','om','g')];ts=Z['ts'];x=Z['x'];ap=Z['apair'];Amc=ap.mean();se=ap.std(ddof=1)/np.sqrt(len(ap))
    y=[]
    for s in(1,-1):
        F=lambda t,v:[.5*np.cos(4*v[0]-s*A*np.sin(om*t))-g*v[0]]
        r=solve_ivp(F,(0,ts[-1]),[np.pi/8],t_eval=np.concatenate(([0.],ts)),method='DOP853',rtol=1e-9,atol=1e-11);y.append(np.cos(4*r.y[0][1:]-s*A*np.sin(om*ts)))
    Am=.5*(shoelace(x,y[0])+shoelace(-x,y[1]));rows.append((D,A,om,g,Am,Amc,(Am-Amc)/abs(Amc)*100,A*om<1.9))
rows=np.array(rows,float)
for D in [.025,.05,.1,.25,.5]:
    out=[]
    for g in [.001,.01,.1,.5]:
        m=(rows[:,0]==D)&(rows[:,3]==g);out.append((g,round(float(np.abs(rows[m,6]).max()),1),round(float(np.abs(rows[m&(rows[:,7]>0),6]).max()),1)))
    print(D,'(g,max all,max locked)',out)
print('signed max at g=.01 by D:',[(D,float(rows[(rows[:,0]==D)&(rows[:,3]==.01),6][np.argmax(np.abs(rows[(rows[:,0]==D)&(rows[:,3]==.01),6]))])) for D in [.025,.05,.1,.25,.5]])
print('=== GC time-local closure, kick response d theta/dk vs H (nmax rule) vs MC E6; signed')
def nrule(D):return int(np.ceil(np.pi/(2*D)))-1
for D in [.025,.05,.1,.25,.5]:
    try: r=json.load(open(f'{NZ}/exps/E6kick_D{D}.json'))
    except: r=None
    taus=np.array([.1,.5,1,2,4]) 
    res={}
    for cl,nm in (('GC',1),('H',nrule(D))):
        f=hier_rhs_factory(D,.5,.01,0.,0.,1.,nm,False,cl)
        y0=cn_init(np.pi/8,0.,nm)
        if nm>1:
            n=np.arange(1,nm+1);c=np.exp(1j*n*np.pi/2-n*n*D);y0=np.concatenate(([np.pi/8],c.real,c.imag))
        s=solve_ivp(f,(0,40),y0,method='DOP853',rtol=1e-10,atol=1e-12);y=s.y[:,-1]
        c=y[1:nm+1]+1j*y[nm+1:];n=np.arange(1,nm+1);ck=c*np.exp(1j*n*4*.02);yk=np.concatenate(([y[0]+.02],ck.real,ck.imag))
        a=solve_ivp(f,(0,4),y,t_eval=taus,method='DOP853',rtol=1e-10,atol=1e-12);b=solve_ivp(f,(0,4),yk,t_eval=taus,method='DOP853',rtol=1e-10,atol=1e-12)
        res[cl]=(b.y[0]-a.y[0])/.02
    mc=''
    if r:
        t=np.array(r['taus']);ii=[np.argmin(abs(t-x)) for x in taus];mc=' MC t='+str(np.round(t[ii],2).tolist())+' '+str(np.round(np.array(r['dth_mc'])[ii]/.02,4).tolist())
    print(D,'GC',np.round(res['GC'],4).tolist(),'H',np.round(res['H'],4).tolist(),mc,flush=True)
print('=== author n choices: n1=ceil(sqrt(40/D))+8 (nzd2), n2=min(n1,int(pi/2D)-2): max Re eig frozen H')
from stab_h import Hmat
for D in [.025,.05,.1,.25]:
    n1=int(np.ceil(np.sqrt(40/D)))+8;n2=min(n1,int(np.pi/(2*D))-2)
    print(D,'n1',n1,round(float(np.linalg.eigvals(Hmat(D,0.,n1)).real.max()),4),'n2',n2,round(float(np.linalg.eigvals(Hmat(D,0.,n2)).real.max()),4),flush=True)
    # time-domain author init, gamma=.01, chi=0, max dev from nmax=n2 over t<=600
    ts=np.linspace(0,600,1201)
    for nm in (n1,):
        f=hier_rhs_factory(D,.5,.01,0.,0.,1.,nm)
        n=np.arange(1,nm+1);c=np.exp(1j*n*np.pi/2-n*n*D);y0=np.concatenate(([np.pi/8],c.real,c.imag))
        try:
            s=solve_ivp(lambda t,y:(_ for _ in ()).throw(ValueError) if np.abs(y).max()>1e3 else f(t,y),(0,600),y0,t_eval=ts,method='DOP853',rtol=1e-8,atol=1e-10)
            print('   n1 run to',s.t[-1],'max|c1|',np.abs(s.y[1]+1j*s.y[nm+1]).max(),flush=True)
        except ValueError: print('   n1 run BLOW before t=600',flush=True)
