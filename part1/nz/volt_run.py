import numpy as np,time,json
from nzlib import *
from volterra import *
np.set_printoptions(precision=3,linewidth=150)
def nrule(D): return int(np.ceil(np.pi/(2*D)))-1
def refH(D,A,om,g,nm,T,h,**kw):
    t=np.arange(int(round(T/h))+1)*h
    return t,hier(D,ETA,g,0.,A,om,nm,kw.get('th_i',np.pi/8),t,uni=kw.get('uni',False),rtol=1e-11,atol=1e-13)
print('=== 1. step convergence vs hierarchy (same nmax), max|c_V-c_H| over last period')
cases=[(.1,1,2,.01,10),(.025,1,2,.01,10),(.1,2,6,.01,10),(.25,1,2,.01,6),(.1,1,.5,.1,10),(.1,1,2,.5,10),(.5,1,2,.01,3)]
for D,A,om,g,nm in cases:
    Tp=2*np.pi/om;T=max(3*Tp,6.)
    prev=None;line=[]
    for h in [.04,.02,.01,.005]:
        t0=time.time();V=volterra(D,A,om,g,nm,h,T);el=time.time()-t0
        t,H=refH(D,A,om,g,nm,T,h)
        e=np.abs(V['c']-H['c']);l=t>=T-Tp
        line.append((h,float(e[l].max()),float(e.max()),int(V['its'].max()),round(el,1)))
    orders=[np.log2(line[i][1]/line[i+1][1]) for i in range(3)]
    print(f'D={D} A={A} w={om} g={g} nmax={nm}:',[(a,f"{b:.1e}",f"{c:.1e}",d,e) for a,b,c,d,e in line],' order',np.round(orders,2),flush=True)
print('=== 2. memory cutoff: error in c and in area vs tau_c (h=.01)')
for D,A,om,g,nm in [(.1,1,2,.01,10),(.025,1,2,.01,10),(.1,2,6,.01,10),(.25,1,2,.01,6),(.1,1,.5,.01,10)]:
    Tp=2*np.pi/om;T=max(3*Tp,6.);h=.01
    t,H=refH(D,A,om,g,nm,T,h);l=t>=T-Tp-1e-9
    # exact-period window
    M=int(round(Tp/h));ch=(A/2)*np.sin(om*t[-M:]);x=np.sin(2*ch);AH=shoelace(x,H['c'].real[-M:])
    line=[]
    for cut in [.25,.5,1,2,4,8,None]:
        V=volterra(D,A,om,g,nm,h,T,cut=cut)
        if V['blow']: line.append((cut,'BLOW'));continue
        AV=shoelace(x,V['c'].real[-M:]);line.append((cut,f"{np.abs(V['c']-H['c'])[l].max():.1e}",f"{(AV-AH)/abs(AH)*100:.3f}%"))
    print(f'D={D} A={A} w={om} g={g}: AH={AH:.5f}',line,flush=True)
print('=== 3. frozen stationary kernel (w=4 g theta_ss) instead of two-time: error on area')
for D,A,om,g,nm in [(.1,.2,2,.01,10),(.1,1,2,.01,10),(.1,2,2,.01,10),(.1,2,.5,.01,10),(.1,2,6,.01,10),(.025,1,2,.01,10),(.25,1,2,.01,6)]:
    Tp=2*np.pi/om;T=max(3*Tp,6.);h=.01
    t,H=refH(D,A,om,g,nm,T,h);M=int(round(Tp/h));ch=(A/2)*np.sin(om*t[-M:]);x=np.sin(2*ch);AH=shoelace(x,H['c'].real[-M:])
    V=volterra(D,A,om,g,nm,h,T,frozen=4*g*np.pi/8)
    AV=shoelace(x,V['c'].real[-M:]) if not V['blow'] else np.nan
    print(f'D={D} A={A} w={om} g={g}: AH={AH:.5f} AV_frozen={AV:.5f} err {(AV-AH)/abs(AH)*100:.2f}% max|dc| {np.abs(V["c"]-H["c"]).max() if not V["blow"] else np.nan:.2e}',flush=True)
print('=== 4. global waveform iteration on theta(t): sup-norm change per sweep')
for D,A,om,g,nm,T in [(.1,1,2,.01,10,10.),(.1,1,2,.1,10,10.),(.1,1,2,.5,10,10.),(.1,2,6,.5,10,10.),(.1,1,2,.5,10,40.)]:
    hist,c,th=waveform(D,A,om,g,nm,.02,T,niter=8)
    print(f'D={D} A={A} w={om} g={g} T={T}',['%.1e'%x for x in hist],flush=True)
print('=== 5. kernel at w=const: K(tau)/K(0), integral width and tail rate')
from scipy.linalg import eigvals
for D in [.0125,.025,.05,.1,.25,.5]:
    nm=max(int(np.ceil(np.pi/(2*D)))-1,3);G,An,Bn=coef(D,ETA,nm)
    for w in [0.,0.04]:
        H=Hblock(An,Bn,w,nm);h=.005;E=expm(h*H);v=np.zeros(nm-1,complex);v[0]=1;K=[]
        for i in range(int(60/h)):K.append(v[0]);v=E@v
        K=np.array(K)
        ev=np.linalg.eigvals(H);lead=ev[np.argmax(ev.real)]
        tw=h*np.sum(np.abs(K))
        ik=h*np.sum(K)
        t12=np.argmax(np.abs(K)<.5)*h
        print(f'D={D} nm={nm} w={w}: K(0)={(Bn[0]*Bn[1]).real:.4f} int|U22|={tw:.3f} intU22={ik:.3f} first|U|<1/2 at {t12:.3f} lead eig {lead:.3f} (4eta={4*ETA})',flush=True)
