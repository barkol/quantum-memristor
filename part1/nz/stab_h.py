import numpy as np
from nzlib import coef
def Hmat(D,w,nmax,uni=False,eta=.5):
    G,A,B=coef(D,eta,nmax,uni);n=np.arange(1,nmax+1);H=np.diag(A[:nmax]-1j*n*w)
    for i in range(nmax):
        if i+1<nmax:H[i,i+1]=B[i]
        if i>=1:H[i,i-1]=B[i]
    return H
