# time-local check of the mean-field master equation (c): integrate (x,z) with Omega from rho(t) only, against scalar law
import numpy as np
from scipy.integrate import solve_ivp
a,gam,th0,chi0=3,.1,.1,.2; A=.2; w=.7
chi=lambda t:chi0+A*np.sin(w*t); dchi=lambda t:A*w*np.cos(w*t)
sc=solve_ivp(lambda t,y:[a*gam*np.cos(2*y[0]-chi(t))**2-gam*(y[0]-th0)],[0,60],[th0],rtol=1e-11,atol=1e-13,dense_output=True)
def rhs(t,r):
    # r=(x,z,n): state only; winding tracked continuously
    x,z=r; ph=np.arctan2(x,z)
    k=np.round(((4*sc.sol(0)[0]-2*chi(0))-np.arctan2(np.sin(4*sc.sol(0)[0]-2*chi(0)),np.cos(4*sc.sol(0)[0]-2*chi(0))))/(2*np.pi))
    th=(ph+2*np.pi*k+2*chi(t))/4
    Om=a*gam*(1+z)/2-gam*(th-th0)-dchi(t)/2
    return [4*Om*z,-4*Om*x]
ph0=4*th0-2*chi(0)
s=solve_ivp(rhs,[0,60],[np.sin(ph0),np.cos(ph0)],rtol=1e-11,atol=1e-13,t_eval=np.linspace(0,60,3001))
phs=np.unwrap(np.arctan2(s.y[0],s.y[1])); th=sc.sol(s.t)[0]; ph=4*th-2*chi(s.t)
print('max |dphi| local vs scalar',np.max(np.abs(phs+ (ph[0]-phs[0])-ph)),'range',ph.min(),ph.max())
