import numpy as np
from scipy.optimize import minimize
X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]).astype(complex)
def ket(t,p): return np.array([np.cos(t/2),np.exp(1j*p)*np.sin(t/2)])
def obs(t,p): return np.sin(t)*np.cos(p)*X+np.sin(t)*np.sin(p)*Y+np.cos(t)*Z
def S(par,phi):
    # S3 with honest-qubit measurement A_x on k; pairing X-Z,Z-X,Y-Y; product input; CZ(phi) after EB loops
    a=ket(par[0],par[1]);b=ket(par[2],par[3]);psi=np.diag([1,1,1,np.exp(1j*phi)])@np.kron(a,b)
    s=0
    for i,B in enumerate([Z,Y,X]):
        A=obs(par[4+2*i],par[5+2*i]); s+=np.real(psi.conj()@np.kron(A,B)@psi)
    return -s
rng=np.random.default_rng(0)
for phi in (0,.1,.25,.5,.75,1.0,np.pi/2,np.pi):
    best=min(minimize(S,rng.uniform(0,2*np.pi,10),args=(phi,),method='BFGS').fun for _ in range(60))
    print(f'phi2={phi:.3f} max S3={-best:.5f} excess={-best-3**.5:.5f} phi^2/(2sqrt3)={phi**2/(2*3**.5):.5f}')
