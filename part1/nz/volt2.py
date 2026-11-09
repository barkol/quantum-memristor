import os
NZ = os.environ.get("NZ_DIR", "results/part1/nz")
import numpy as np,glob
from nzlib import *
from volterra import *
from scipy.optimize import fsolve
def nrule(D): return int(np.ceil(np.pi/(2*D)))-1
print('=== A. Volterra two-time kernel (cut tau_c=8) vs event-driven MC on grid points; pair area; h~0.005')
for D,A,om,g in [(.025,1.,2.,.01),(.1,1.,2.,.01),(.1,2.,6.,.01),(.25,1.,2.,.01),(.1,1.,2.,.1),(.1,1.,.5,.1),(.025,2.,6.,.1),(.1,2.,.5,.01),(.5,1.,2.,.01)]:
    Z=np.load(f'{NZ}/grid/D{D}_A{A}_w{om}_g{g}.npz');Ts=float(Z['Ts']);Tp=2*np.pi/om;x=Z['x'];ap=Z['apair'];Amc=ap.mean();se=ap.std(ddof=1)/np.sqrt(len(ap))
    k=max(1,int(np.ceil((Tp/256)/.005)));h=Tp/256/k;nm=nrule(D)
    ist=int(round(Ts/h));T=(ist+256*k)*h
    # note Ts is integer multiple of Tp in the runner except for roundoff
    res=[]
    for s in (1,-1):
        V=volterra(D,s*A,om,g,nm,h,T,cut=8.)
        res.append(V)
    yp=res[0]['c'].real[ist:ist+256*k:k];ym=res[1]['c'].real[ist:ist+256*k:k]
    Av=.5*(shoelace(x,yp)+shoelace(-x,ym))
    print(f'D={D} A={A} w={om} g={g} nmax={nm} Ts={Ts:.1f} h={h:.4f}: Amc={Amc:.5f}+-{se:.5f} AV={Av:.5f} err {(Av-Amc)/abs(Amc)*100:+.2f}% z={(Av-Amc)/se:+.1f} max|c|={max(np.abs(r["c"]).max() for r in res):.4f} maxit={max(r["its"].max() for r in res)}',flush=True)
print('=== B. cutoff at Delta=.5 and .25 (nmax rule+) ; max|c|')
for D,A,om,g,nm in [(.5,1.,2.,.01,3),(.5,.2,2.,.01,3),(.25,1.,.5,.01,6)]:
    Tp=2*np.pi/om;T=max(4*Tp,40.);h=.01
    t=np.arange(int(round(T/h))+1)*h;H=hier(D,ETA,g,0.,A,om,nm,np.pi/8,t,rtol=1e-10,atol=1e-12);M=int(round(Tp/h));x=np.sin(A*np.sin(om*t[-M:]));AH=shoelace(x,H['c'].real[-M:])
    out=[]
    for cut in [1,2,4,8,16,32]:
        V=volterra(D,A,om,g,nm,h,T,cut=cut)
        if V['blow']:out.append((cut,'BLOW'));continue
        out.append((cut,f"{(shoelace(x,V['c'].real[-M:])-AH)/abs(AH)*100:+.3f}%",f"max|c|={np.abs(V['c']).max():.3f}"))
    print(D,A,om,g,'AH=%.5f'%AH,out,flush=True)
print('=== C. small cutoff: does broken positivity give |c|>1 or blow-up? (A=1,w=2, g=.01, T=100)')
for D,nm in [(.025,20),(.1,15),(.25,6)]:
    out=[]
    for cut in [.1,.25,.5,1,2]:
        V=volterra(D,1.,2.,.01,nm,.01,100.,cut=cut);out.append((cut,V['blow'],round(float(np.abs(V['c']).max()),3),round(float(V['t'][-1]),1)))
    print(D,out,flush=True)
print('=== D. long-time marching stability: T=2000 vs hierarchy, h=.02, cut=8')
for D,A,om,g,nm in [(.1,1.,.5,.01,15),(.25,1.,.5,.01,6),(.1,1.,2.,.1,15)]:
    T=2000.;h=.02;V=volterra(D,A,om,g,nm,h,T,cut=8.);t=V['t'];H=hier(D,ETA,g,0.,A,om,nm,np.pi/8,t,rtol=1e-9,atol=1e-11,tmax=500)
    if H is None or 'fail' in H: print('H fail',H);continue
    e=np.abs(V['c']-H['c']);print(f'D={D} A={A} w={om} g={g}: max|V-H| first 100 {e[:5000].max():.2e}, last 100 {e[-5000:].max():.2e}, dtheta end {abs(V["th"][-1]-H["th"][-1]):.2e} maxit {V["its"].max()}',flush=True)
print('=== E. time-local closures: stationary + spectrum (no memory)')
for D in [.025,.05,.1,.25]:
    G,An,Bn=coef(D,ETA,2);g=.01
    def F(v,cl):
        c=v[1]+1j*v[2];c2={'GC':c*c*abs(c)**2,'MF':c*c}[cl];d=(An[0]-1j*4*g*v[0])*c+Bn[0]*(1+c2);return np.array([-g*v[0]+ETA*c.real,d.real,d.imag])
    for cl in ['GC','MF']:
        v0=np.array([.39,0.,np.exp(-D)]);v,inf,ier,_=fsolve(F,v0,args=(cl,),full_output=True,xtol=1e-12)
        J=np.zeros((3,3))
        for i in range(3):
            e=np.zeros(3);e[i]=1e-6;J[:,i]=(F(v+e,cl)-F(v-e,cl))/2e-6
        print(f'D={D} {cl}: ier={ier} th={v[0]:.4f} |c|={abs(v[1]+1j*v[2]):.5f} eig',np.round(np.sort_complex(np.linalg.eigvals(J)),4),flush=True)
