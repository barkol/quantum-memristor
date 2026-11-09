# Corrected closures. All signed unless uni=True.
import numpy as np,time
from scipy.integrate import solve_ivp
from nzlib import coef
class TooLong(Exception): pass
def run_variant(kind,D,eta,g,th0,A,om,nmax,th_i,ts,sgn=1.,uni=False,tmax=400.,rtol=1e-8,atol=1e-10):
    """kind: 'Hstat' A_n(1-g/geff); 'Hdyn' variance ODE; 'H2' two-index closure (c_n,m1_n,S)."""
    G,An,Bn=coef(D,eta,nmax,uni);An=An[:nmax];Bn=Bn[:nmax];n=np.arange(1,nmax+1);a=4*n*D
    t00=time.time()
    ti=np.atleast_1d(th_i)-th0
    c0=np.array([np.mean(np.exp(1j*k*4*(ti+th0))) for k in n])
    m0=ti.mean();S0=np.mean(ti**2)
    chid=lambda t:sgn*(A/2)*om*np.cos(om*t)
    nb=lambda v,z0:np.concatenate((v[1:],[0]))+np.concatenate(([z0],v[:-1]))
    if kind=='Hstat':
        def f(t,y):
            if time.time()-t00>tmax: raise TooLong
            m=y[0];c=y[1:nmax+1]+1j*y[nmax+1:]
            ge=g+4*eta*max(c[0].imag,0.)
            fac=1-g/ge if ge>0 else 0.
            dc=(An*fac-1j*n*(4*g*m+2*chid(t)))*c+Bn*nb(c,1.)
            return np.concatenate(([-g*m+eta*c[0].real],dc.real,dc.imag))
        y0=np.concatenate(([m0],c0.real,c0.imag));getc=lambda s:s.y[1:nmax+1]+1j*s.y[nmax+1:]
    elif kind=='Hdyn':
        def f(t,y):
            if time.time()-t00>tmax: raise TooLong
            m=y[0];V=y[1];c=y[2:nmax+2]+1j*y[nmax+2:]
            ge=g+4*eta*max(c[0].imag,0.)
            dc=(An+16*n**2*g*V-1j*n*(4*g*m+2*chid(t)))*c+Bn*nb(c,1.)
            return np.concatenate(([-g*m+eta*c[0].real,-2*ge*V+G*D*D],dc.real,dc.imag))
        y0=np.concatenate(([m0,S0-m0**2],c0.real,c0.imag));getc=lambda s:s.y[2:nmax+2]+1j*s.y[nmax+2:]
    elif kind=='H2':
        m10=np.array([np.mean(ti*np.exp(1j*k*4*(ti+th0))) for k in n])
        def f(t,y):
            if time.time()-t00>tmax: raise TooLong
            m=y[0];S=y[1];c=y[2:2+nmax]+1j*y[2+nmax:2+2*nmax];d=y[2+2*nmax:2+3*nmax]+1j*y[2+3*nmax:]
            w=2*chid(t)
            cs=nb(c,1.);ds=nb(d,m)
            m2=S*c+2*m*d-2*m*m*c
            if uni:
                dc=(An-1j*n*w)*c+0.5*An*cs-4j*n*g*d
                dd=(An-g-1j*n*w)*d+0.5*An*ds-4j*n*g*m2+0.5*G*D*np.exp(1j*a)*(c+0.5*cs)
                dm=-g*m+0.5*eta*(1+c[0].real)
                dS=-2*g*S+G*D*(m+d[0].real)+0.5*G*D*D*(1+c[0].real)
            else:
                dc=(An-1j*n*w)*c+Bn*cs-4j*n*g*d
                dd=(An-g-1j*n*w)*d+Bn*ds-4j*n*g*m2+G*D*(1j*np.sin(a)*c+0.5*np.cos(a)*cs)
                dm=-g*m+eta*c[0].real
                dS=-2*g*S+2*eta*d[0].real+G*D*D
            return np.concatenate(([dm,dS],dc.real,dc.imag,dd.real,dd.imag))
        y0=np.concatenate(([m0,S0],c0.real,c0.imag,m10.real,m10.imag));getc=lambda s:s.y[2:2+nmax]+1j*s.y[2+nmax:2+2*nmax]
    try:
        s=solve_ivp(f,(0,ts[-1]),y0,t_eval=ts,method='DOP853',rtol=rtol,atol=atol)
    except TooLong:
        return None
    if not s.success or s.y.shape[1]!=len(ts): return None
    return getc(s)[0],s.y[0]+th0
