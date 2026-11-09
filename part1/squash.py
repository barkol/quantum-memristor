import numpy as np, itertools
X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]).astype(complex)
def ev(P):
    w,v=np.linalg.eigh(P); return v[:,1],v[:,0]
def M(P,m):
    a,b=ev(P); A=a; B=b
    for _ in range(m-1): A=np.kron(A,a); B=np.kron(B,b)
    return np.outer(A,A.conj())-np.outer(B,B.conj())
for m in range(1,7):
    best=0
    Ms=[M(P,m) for P in (X,Y,Z)]
    for s in itertools.product((1,-1),repeat=3):
        best=max(best,np.linalg.eigvalsh(sum(c*Q for c,Q in zip(s,Ms)))[-1])
    print(m,round(best,5))
