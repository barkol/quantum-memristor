"""Coherent syndrome-ancilla decoder on a set of four words of length five (d = 3) that is not a linear code but a coset of one, on random word sets, and on a
d = 2 code. Quoted in Part II, Sec. "Coherent recall": the map |x>|0> -> |D(x)>|x xor D(x)> restores the equal superposition of
two stored words with fidelity 1 whenever the SAME error of weight <= floor((d-1)/2) acts on both branches, for any word set;
linearity is needed only to compute the syndrome Hx without a search over the stored words. Runs in seconds.
Output: printed (a copy is stored in results/check_nonlinear_code.txt)."""
import numpy as np, itertools
def build(words, n):
    """isometry |x>|0> -> |D(x)>|x^D(x)>, D = nearest word (ties -> first). returns dict x->(c,e)"""
    W=list(words); m={}
    for x in range(2**n):
        d=[bin(x^w).count('1') for w in W]; c=W[int(np.argmin(d))]
        m[x]=(c,x^c)
    return m
def apply(m,n,psi): # psi: dict x->amp on data (ancilla 0) -> joint state array [data,anc]
    out=np.zeros((2**n,2**n),complex)
    for x,a in psi.items():
        c,e=m[x]; out[c,e]+=a
    return out
def fid(out,ca,cb,n):
    tgt=np.zeros((2**n,2**n),complex)
    # ideal: (|ca>+|cb>)/r2 with ancilla in ANY fixed state e -> compute best product overlap: trace out ancilla
    rho=out.reshape(2**n,2**n)  # data x anc
    t=np.zeros(2**n,complex); t[ca]=t[cb]=1/np.sqrt(2)
    v=t.conj()@rho
    return float(np.sum(np.abs(v)**2))
n=5
# nonlinear 4-word code with d>=3: brute force
best=None
for W in itertools.combinations(range(32),4):
    if all(bin(a^b).count('1')>=3 for a,b in itertools.combinations(W,2)):
        lin = all((a^b) in W for a,b in itertools.combinations(W,2))
        if not lin: best=W;break
W=best; print("nonlinear code",[format(w,'05b') for w in W],"min dist",min(bin(a^b).count('1') for a,b in itertools.combinations(W,2)))
m=build(W,n)
def test(W,m,n,label):
  res=[]
  for ca,cb in itertools.combinations(W,2):
    for e in range(2**n):   # same error operator on both branches
        psi={ca^e:1/np.sqrt(2)}; psi[cb^e]=psi.get(cb^e,0)+1/np.sqrt(2)
        res.append((bin(e).count('1'),fid(apply(m,n,psi),ca,cb,n)))
  for w in sorted(set(r[0] for r in res)):
    f=[r[1] for r in res if r[0]==w]; print(f"{label} wt(e)={w}: n={len(f)} min F={min(f):.3f} max F={max(f):.3f}")
test(W,m,n,"nonlinear d=3, same e")
# different errors per branch
f=[]
for ca,cb in itertools.combinations(W,2):
  for e1 in [1,2,4,8,16]:
    for e2 in [1,2,4,8,16]:
      if e1==e2: continue
      psi={ca^e1:1/np.sqrt(2),cb^e2:1/np.sqrt(2)}
      f.append(fid(apply(m,n,psi),ca,cb,n))
print("different weight-1 errors per branch: F min/max",min(f),max(f))
# arbitrary (random, non-code) set of 3 words n=12, weight-1 errors
rg=np.random.default_rng(1); n2=9
for trial in range(3):
    while True:
        Wr=rg.choice(2**n2,3,replace=False); d=min(bin(a^b).count('1') for a,b in itertools.combinations(Wr,2))
        if d>=3: break
    m2=build(Wr,n2); ff=[]
    for ca,cb in itertools.combinations(Wr,2):
        for i in range(n2):
            e=1<<i; psi={ca^e:2**-.5,cb^e:2**-.5}
            ff.append(fid(apply(m2,n2,psi),ca,cb,n2))
    print("random set d=",d,"wt1 same e: min F",min(ff))
# weight above t, same e, d=3 code
Wr=W
ff=[]
for ca,cb in itertools.combinations(W,2):
  for e in range(32):
    if bin(e).count('1')==2:
      ff.append(fid(apply(m,n,{ca^e:2**-.5,cb^e:2**-.5}),ca,cb,n))
print("wt 2 > t=1: min F",min(ff),"max",max(ff),"mean",np.mean(ff))
# word-dependent: tie/ambiguity example: d=2 code
W2=(0b00000,0b00011,0b11100); m3=build(W2,5)
ff=[fid(apply(m3,5,{ca^e:2**-.5,cb^e:2**-.5}),ca,cb,5) for ca,cb in itertools.combinations(W2,2) for e in (1,2,4,8,16)]
print("d=2 code, wt1 same e:",sorted(set(round(x,3) for x in ff)))
