# Integral (Volterra) form with two-time kernel under drive: K(t,s)=B1 B2 [U(t,s)]_22, I(t) from c_n(0).
# Marching: exponential-midpoint propagator on n>=2 block, trapezoid in the memory integral.
import numpy as np,sys,time,json,os
from scipy.linalg import expm
from nzlib import *
ETA=.5
def Hblock(An,Bn,w,nmax):
    # n=2..nmax  index i=n-2 ; An[k] refers n=k+1
    m=nmax-1;n=np.arange(2,nmax+1)
    H=np.diag(An[1:nmax]-1j*n*w)
    for i in range(m):
        if i+1<m:H[i,i+1]=Bn[i+1]
        if i>=1:H[i,i-1]=Bn[i+1]
    return H
def volterra(D,A,om,g,nmax,h,T,th_i=np.pi/8,th0=0.,cut=None,uni=False,frozen=None,eta=ETA,maxit=30,tol=1e-11,thetafix=None):
    """cut: memory cutoff time. frozen: if not None, kernel/propagator uses constant w=frozen (stationary kernel).
    thetafix: prescribed theta(t_k) array (waveform iteration): no self-consistency in step."""
    G,An,Bn=coef(D,eta,nmax,uni)
    Nst=int(round(T/h));t=np.arange(Nst+1)*h
    chid=lambda t:(A/2)*om*np.cos(om*t)
    c=np.zeros(Nst+1,complex);th=np.zeros(Nst+1)
    ti=np.atleast_1d(th_i)
    c[0]=np.mean(np.exp(1j*4*ti));th[0]=np.mean(ti)-th0
    z=np.array([np.mean(np.exp(1j*k*4*ti)) for k in range(2,nmax+1)])
    Mc=Nst+1 if cut is None else max(int(round(cut/h)),1)
    V=np.zeros((nmax-1,Mc+1),complex) # columns: v_s, ring buffer by index j%(Mc+1)
    e2=np.zeros(nmax-1,complex);e2[0]=1
    V[:,0]=e2
    A1,B1,B2=An[0],Bn[0],Bn[1]
    if uni: S1=lambda: None
    its=[];E_frozen=None if frozen is None else expm(h*Hblock(An,Bn,frozen,nmax))
    fp=lambda c,th,cdot:0
    def wf(k_t,thv): return 4*g*thv+2*chid(k_t)
    Ik=z[0];mem=0.0
    f_prev=None
    def dth(thv,cv):
        return -g*thv+(0.5*eta*(1+cv.real) if uni else eta*cv.real)
    f0=(A1-1j*wf(t[0],th[0]))*c[0]+B1*(1+Ik)
    f_prev=f0
    for k in range(Nst):
        tn=t[k+1];cn=c[k]+h*f_prev;thn=th[k]+h*dth(th[k],c[k]) if thetafix is None else thetafix[k+1]
        V0=V;z0=z;conv=False
        for it in range(maxit):
            if frozen is None:
                wm=4*g*0.5*(th[k]+thn)+2*chid(t[k]+h/2)
                E=expm(h*Hblock(An,Bn,wm,nmax))
            else:E=E_frozen
            lo=max(0,k+1-Mc)
            # active columns j=lo..k
            idx=[j%(Mc+1) for j in range(lo,k+1)]
            Vn=E@V[:,idx]
            zn=E@z
            Kv=B1*B2*Vn[0]  # K(t_{k+1},t_j), j=lo..k
            # trapezoid: weights h for inner, h/2 at ends
            wgt=np.full(len(idx),h);wgt[0]=h/2
            Mem_exp=np.sum(wgt*Kv*c[lo:k+1])
            K00=B1*B2
            # implicit trapezoid for c_{k+1}; w at t_{k+1}
            wn=wf(tn,thn)
            rhsconst=B1*(1+zn[0])+Mem_exp
            cnew=(c[k]+h/2*f_prev+h/2*rhsconst)/(1-h/2*(A1-1j*wn)-h/2*h/2*K00)
            thnew=(th[k]+h/2*dth(th[k],c[k])+h/2*(dth(0,cnew)))/(1+h/2*g) if thetafix is None else thetafix[k+1]
            err=abs(cnew-cn)+abs(thnew-thn)
            cn=cnew;thn=thnew
            if err<tol: conv=True;its.append(it+1);break
        if not conv: its.append(maxit)
        c[k+1]=cn;th[k+1]=thn
        V[:,idx]=Vn
        V[:,(k+1)%(Mc+1)]=e2
        z=zn
        f_prev=(A1-1j*wf(tn,thn))*cn+rhsconst+h/2*K00*cn
        if abs(cn)>1e6: return dict(t=t[:k+2],c=c[:k+2],th=th[:k+2]+th0,its=np.array(its),blow=True)
    return dict(t=t,c=c,th=th+th0,its=np.array(its),blow=False)
def waveform(D,A,om,g,nmax,h,T,niter=8,**kw):
    # global iteration on theta(t): prescribe theta -> linear Volterra -> update theta by exact relaxation quadrature
    N=int(round(T/h));t=np.arange(N+1)*h
    th=np.full(N+1,np.pi/8);hist=[]
    for m in range(niter):
        R=volterra(D,A,om,g,nmax,h,T,thetafix=th-0.0,**kw)
        c=R['c']
        new=np.zeros(N+1);new[0]=np.pi/8
        for k in range(N):
            new[k+1]=(new[k]*(1-h/2*g)+h/2*ETA*(c[k].real+c[k+1].real))/(1+h/2*g)
        hist.append(float(np.max(np.abs(new-th))));th=new
    return hist,c,th
