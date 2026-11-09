# Independent implementation. Hierarchy (signed / unipolar), closures, MC wrapper.
import numpy as np, ctypes, os
from scipy.integrate import solve_ivp
_lib=None
def lib():
    global _lib
    if _lib is None:
        _lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mc.so'))
        _lib.simulate.restype=ctypes.c_longlong
        d=ctypes.c_double;ll=ctypes.c_longlong;dp=ctypes.c_void_p
        _lib.simulate.argtypes=[ll,d,d,d,d,d,d,ctypes.c_int,dp,ll,dp,ll,d,ctypes.c_uint64,dp,dp]
    return _lib
NC=8;NB=41
def mc(N,Delta,eta,gam,th0,A,om,uni,theta_init,tsamp,seed,phi_ref=np.pi/2):
    ti=np.atleast_1d(np.asarray(theta_init,float));ts=np.asarray(tsamp,float)
    out=np.zeros((len(ts),2,NC));hist=np.zeros((2,NB))
    nev=lib().simulate(int(N),Delta,eta/Delta,gam,th0,A,om,int(uni),ti.ctypes.data,len(ti),ts.ctypes.data,len(ts),phi_ref,int(seed),out.ctypes.data,hist.ctypes.data)
    return out/N,hist/N,nev
def coef(Delta,eta,nmax,uni=False):
    G=eta/Delta;n=np.arange(1,nmax+2);a=4*n*Delta
    if uni:
        A=(G/2)*(np.exp(1j*a)-1);B=A/2
    else:
        A=G*(np.cos(a)-1)+0j;B=0.5j*G*np.sin(a)
    return G,A,B
def shoelace(x,y):
    return 0.5*np.sum(x*np.roll(y,-1)-np.roll(x,-1)*y)
def hier_rhs_factory(Delta,eta,gam,th0,A,om,nmax,uni=False,closure='H',sgn=1.0):
    """state: [theta_tilde, Re c_1..n, Im c_1..n]. closure: H plain; MF c2=c^2; GC c2=c^2|c|^2;
    Hcorr: A_n->A_n(1-gam/gam_eff) with local gam_eff=gam+4 eta Im c/|c| (signed) ;
    Hcorrdyn : adds variance ODE; extra state S=var theta, term 16 n^2 gam S c_n"""
    G,An,Bn=coef(Delta,eta,nmax,uni)
    nn=np.arange(1,nmax+1)
    An_=An[:nmax];Bn_=Bn[:nmax]
    def chid(t): return sgn*(A/2)*om*np.cos(om*t)
    def f(t,y):
        th=y[0];c=y[1:nmax+1]+1j*y[nmax+1:2*nmax+1]
        w=4*gam*th+2*chid(t)
        cp=np.empty(nmax+2,complex)
        cp[0]=1.0;cp[1:nmax+1]=c;cp[nmax+1]=0.0
        if closure=='MF' and nmax==1: cp[2]=c[0]**2
        if closure=='GC' and nmax==1: cp[2]=c[0]**2*abs(c[0])**2
        s=cp[2:nmax+2]+cp[0:nmax]
        if uni:
            dc=(An_-1j*nn*w)*c+0.5*An_*s*1.0  # B_n=A_n/2
            dth=-gam*th+0.5*eta*(1+c[0].real)
        else:
            dc=(An_-1j*nn*w)*c+Bn_*s
            dth=-gam*th+eta*c[0].real
        return np.concatenate(([dth],dc.real,dc.imag))
    return f
def cn_init(theta_init,th0,nmax,chi0=0.0):
    ti=np.atleast_1d(theta_init);n=np.arange(1,nmax+1)
    c=np.array([np.mean(np.exp(1j*k*(4*ti-2*chi0))) for k in n])
    return np.concatenate(([np.mean(ti)-th0],c.real,c.imag))
class Abort(Exception): pass
def hier(Delta,eta,gam,th0,A,om,nmax,theta_init,tsamp,uni=False,closure='H',sgn=1.0,rtol=1e-9,atol=1e-11,method='DOP853',max_step=None,tmax=240.,blow=1e3):
    import time as _t
    f0=hier_rhs_factory(Delta,eta,gam,th0,A,om,nmax,uni,closure,sgn)
    t00=_t.time();info={}
    def f(t,y):
        if _t.time()-t00>tmax: info['why']='time';info['t']=t;raise Abort
        if np.abs(y[1:]).max()>blow: info['why']='blow';info['t']=t;raise Abort
        return f0(t,y)
    y0=cn_init(theta_init,th0,nmax)
    kw={} if max_step is None else {'max_step':max_step}
    try:
        s=solve_ivp(f,(0,tsamp[-1]),y0,t_eval=tsamp,method=method,rtol=rtol,atol=atol,**kw)
    except Abort:
        return dict(fail=info['why'],tfail=float(info['t']))
    if not s.success or s.y.shape[1]!=len(tsamp):
        return None
    return dict(th=s.y[0]+th0,c=s.y[1]+1j*s.y[nmax+1],c2=(s.y[2]+1j*s.y[nmax+2]) if nmax>1 else None,cmaxabs=np.max(np.hypot(s.y[1:nmax+1],s.y[nmax+1:2*nmax+1]),axis=1),nfev=s.nfev)
def mc_kick(N,Delta,eta,gam,th0,A,om,uni,theta_init,tk,dk,tsamp,seed):
    L=lib();d=ctypes.c_double;ll=ctypes.c_longlong;dp=ctypes.c_void_p
    L.simulate_kick.restype=ll
    L.simulate_kick.argtypes=[ll,d,d,d,d,d,d,ctypes.c_int,d,d,d,dp,ll,ctypes.c_uint64,dp]
    ts=np.asarray(tsamp,float);out=np.zeros((len(ts),2,NC))
    L.simulate_kick(int(N),Delta,eta/Delta,gam,th0,A,om,int(uni),theta_init,tk,dk,ts.ctypes.data,len(ts),int(seed),out.ctypes.data)
    return out/N
