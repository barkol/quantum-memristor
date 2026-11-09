# independent Trotter gate; and max|Ediag|dt of every setting quoted
import numpy as np, itertools
def tab(n):
    return 1-2*((np.arange(2**n)[:,None]>>np.arange(n))&1).astype(float)
def anneal_probs(J,h,Z,T,K,Om=1.):
    n=len(h); dt=T/K; E=-.5*np.einsum('ai,ij,aj->a',Z,J,Z,optimize=True)-Z@h
    psi=np.full(2**n,2**(-n/2),complex)
    for k in range(K):
        s=(k+.5)/K; a=(1-s)*Om*dt; psi=psi*np.exp(-.5j*dt*s*E)
        c,si=np.cos(a),1j*np.sin(a)
        for q in range(n):
            st=1<<q; v=psi.reshape(-1,2,st); x=v[:,0,:].copy(); v[:,0,:]=c*x+si*v[:,1,:]; v[:,1,:]=si*x+c*v[:,1,:]
        psi=psi*np.exp(-.5j*dt*s*E)
    return abs(psi)**2,E
rng=np.random.default_rng(11)
print('n f h  max|E|dt  m_readout  expected')
for n in(8,12,14):
    Z=tab(n); nf=int(np.floor(.1875*n+.5))
    for h in(1.,2.,3,4,6,8,12,16,20,24,28,30,31,32,33,36,40,50,63,64,70,100):
        ms=[];q=[]
        for net in range(3 if n==14 else 6):
            P=rng.choice([-1.,1.],size=(3,n)); J=P.T@P/n; np.fill_diagonal(J,0)
            xi=P[rng.integers(3)]; t=xi.copy(); t[rng.permutation(n)[:nf]]*=-1
            Pr,E=anneal_probs(J,h*t,Z,12.,120); ms.append(abs(Z[Pr.argmax()]@xi)/n); q.append(abs(E).max()*.1)
        print(n,nf,h,round(max(q),2),round(np.mean(ms),3),round(min(ms),3),abs(n-2*nf)/n,flush=True)
# largest quantity at reported settings (draw 200 nets each)
print('reported settings')
for n,L,h,T in [(8,3,.6,12),(8,3,.6,50),(8,3,2.4,12),(12,3,2.4,12),(14,3,2.4,12),(14,3,1.2,12),(14,5,.6,12),(10,10,.6,12),(8,20,.6,12)]:
    Z=tab(n); q=[]
    for net in range(30):
        P=rng.choice([-1.,1.],size=(L,n)); J=P.T@P/n; np.fill_diagonal(J,0)
        xi=P[0]; t=xi.copy(); E=-.5*np.einsum('ai,ij,aj->a',Z,J,Z,optimize=True)-Z@(h*t); q.append(abs(E).max()*T/120)
    print(n,L,h,T,'max|E|dt',round(max(q),2))
