import numpy as np
from scipy.optimize import fsolve
from nzlib import *
np.set_printoptions(precision=4,suppress=True,linewidth=150)
ETA=.5
def Hmat(D,w,nmax,uni=False,eta=ETA):
    G,A,B=coef(D,eta,nmax,uni)
    n=np.arange(1,nmax+1)
    H=np.diag(A[:nmax]-1j*n*w)
    for i in range(nmax):
        if i+1<nmax: H[i,i+1]=B[i]
        if i-1>=0: H[i,i-1]=B[i]
    return H
print('=== A. max Re eig of frozen hierarchy matrix H (n=1..nmax), signed; rule nmax<pi/(2D)')
for D in [.025,.05,.1,.25,.5,.7]:
    rule=int(np.ceil(np.pi/(2*D)))-1
    row=[]
    for nm in [rule,rule+1,rule+2,2*rule+1,4*rule+1,200]:
        vals=[np.linalg.eigvals(Hmat(D,w,nm)).real.max() for w in [0,.02,.1,.5,1,2,5]]
        row.append((nm,round(vals[0],4),round(max(vals),4),[0,.02,.1,.5,1,2,5][int(np.argmax(vals))]))
    print(f'D={D} rule={rule}: (nmax,maxRe@w=0,maxRe over w,argw)',row)
print('=== A2. same, unipolar')
for D in [.025,.05,.1,.25,.5]:
    rule=int(np.ceil(np.pi/(2*D)))-1
    row=[]
    for nm in [rule,rule+1,2*rule+1,200]:
        vals=[np.linalg.eigvals(Hmat(D,w,nm,True)).real.max() for w in [0,.02,.1,.5,1,2,5]]
        row.append((nm,round(vals[0],4),round(max(vals),4)))
    print(f'D={D} rule={rule}',row)
print('=== A3. generic Delta scan: signed, is any eigenvalue Re>0 below/above rule? (w=0)')
for D in [.03,.04,.0625,.0785398,.08,.13,.15,.2,.3,.35,.39,.45,.6,.785398]:
    rule=int(np.ceil(np.pi/(2*D)))-1
    r=[(nm,round(np.linalg.eigvals(Hmat(D,0,nm)).real.max(),4)) for nm in [max(rule,1),rule+1,rule+2,rule+5,3*rule+3]]
    print(D,rule,r)
print('=== B. Fixed point + linear spectrum at const chi (signed): theta, c_n; T5 numbers')
def fp(D,gam,th0=0.,nmax=None,uni=False,phi0=np.pi/2,eta=ETA):
    if nmax is None: nmax=int(np.ceil(np.pi/(2*D)))-1
    f=hier_rhs_factory(D,eta,gam,th0,0.,1.,nmax,uni)
    # time-integrate then Newton
    y0=cn_init((phi0)/4,th0,nmax)
    from scipy.integrate import solve_ivp
    s=solve_ivp(f,(0,300),y0,method='LSODA',rtol=1e-9,atol=1e-11)
    y=s.y[:,-1]
    y,info,ier,msg=fsolve(lambda v:f(0,v),y,full_output=True,xtol=1e-12)
    J=np.zeros((len(y),len(y)))
    for i in range(len(y)):
        e=np.zeros(len(y));e[i]=1e-6
        J[:,i]=(f(0,y+e)-f(0,y-e))/2e-6
    ev=np.linalg.eigvals(J)
    return y,ev,nmax,ier
for D in [.025,.05,.1,.25,.5]:
    y,ev,nm,ier=fp(D,.01)
    c=y[1]+1j*y[nm+1]
    ev=ev[np.argsort(-ev.real)]
    print(f'D={D} nmax={nm} ier={ier} theta={y[0]:.5f} |c|={abs(c):.5f} e^-D={np.exp(-D):.5f} arg c={np.angle(c):.4f} top eig:',ev[:5])
