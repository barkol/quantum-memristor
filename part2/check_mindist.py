"""Distribution of the minimum Hamming distance d of 32 random +-1 words of length n = 256, against the Hadamard words (d = 128).
Quoted in Part II, Sec. "Codes": the minimum distance of random Hebbian patterns fluctuates (median 104, P(d <= 102) = 0.29)
and the decoding radius t = floor((d-1)/2) with it. 1e5 draws, fixed seed. Takes about a minute on a laptop core.
Output: printed (a copy is stored in results/check_mindist.txt)."""
import numpy as np
from scipy.linalg import hadamard
rng=np.random.default_rng(1)
n,M,R=256,32,100000
iu=np.triu_indices(M,1)
dm=np.empty(R,int); ds=np.empty(R,int)
for r in range(R):
    P=rng.integers(0,2,(M,n),dtype=np.int8)*2-1
    G=P.astype(np.int32)@P.T.astype(np.int32)
    d=(n-G[iu])//2
    dm[r]=d.min(); ds[r]=np.minimum(d,n-d).min()
for name,x in [("plain Hamming",dm),("sign-blind min(d,n-d)",ds)]:
    print(name,"mean %.2f median %.0f sd %.2f"%(x.mean(),np.median(x),x.std()),
      "q05,q25,q75,q95",np.percentile(x,[5,25,75,95]),"P(<=102)=%.3f P(<=101)=%.3f P(==102)=%.3f"%((x<=102).mean(),(x<=101).mean(),(x==102).mean()),
      "t=floor((d-1)/2): median",(np.median(x)-1)//2)
H=hadamard(256); rows=H[1:33]; G=rows@rows.T; d=(256-G[iu])//2; print("hadamard min d",d.min(),"max",d.max(), "sign-blind",np.minimum(d,256-d).min())
# also with M=32 words from Hadamard incl. row 0
