"""Independent redraws of the 60-network disorder average at the cell n = 8, L = 3, f = 0.1875 (bias h = 0.6, K = 120, T = 12, Omega0 = 1):
biased greedy, simulated annealing and quantum annealing, with the paired differences and their standard errors. Seed 20260912 is the
tuning draw of the Appendix (it also runs the tuned schedule T = 50, Omega0 = 0.5); the seeds 1 to 5 are the further draws quoted in the
text, whose mean gaps are 0.020 (greedy minus simulated annealing) and 0.007 (simulated annealing minus quantum). About 14 s per draw.
Usage: python redraw_panel_b.py [seed ...]   (default seeds: 1 2 3 4 5, plus 20260912). Prints the table; writes results/redraw_panel_b.json."""
import numpy as np, sys, time, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_qhnm as m
from fixes_compute import greedy_biased_batch, corrupt
N,L,F,BIAS=8,3,0.1875,0.6
NET,TR=60,100
spins=m.spin_table(N); nflip=int(np.floor(F*N+0.5))
def run(seed, cells):
    rng=np.random.default_rng(seed); m.RNG=np.random.default_rng(seed+1)
    nets=[]
    for _ in range(NET):
        P=rng.choice([-1.0,1.0],size=(L,N))
        J,_=m.stream_memristor_couplings(P,eta=1.0/N,gamma=0.01,dt=1.0,steps_per_pattern=1)
        tgt=rng.integers(0,L,TR); tb=P[tgt]; nb=corrupt(rng,tb,nflip,N); nets.append((J,tb,nb))
    out={k:[] for k in ["gb","sa"]+[f"q_{T}_{O}" for T,O in cells]}
    for J,tb,nb in nets:
        out["gb"].append(m.overlap_batch(greedy_biased_batch(J,nb,BIAS),tb).mean())
        out["sa"].append(m.overlap_batch(m.retrieve_classical_sa_batch(J,nb,BIAS),tb).mean())
        for T,O in cells:
            R=m.retrieve_quantum_anneal_batch(J,nb,Omega0=O,n_steps=120,T=T,bias_strength=BIAS,spins=spins)
            out[f"q_{T}_{O}"].append(m.overlap_batch(R,tb).mean())
    return {k:np.array(v) for k,v in out.items()}
def s(x): return x.mean(), x.std(ddof=1)/np.sqrt(len(x))
res={}
for seed in [20260912] + ([int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 3, 4, 5]):
    t=time.time(); cells=[(12.0,1.0),(50.0,0.5)] if seed==20260912 else [(12.0,1.0)]
    o=run(seed,cells); print("seed",seed,"time",round(time.time()-t),flush=True)
    for k,v in o.items(): print("  %-10s %.4f se %.4f (95%% %.4f)"%(k,*s(v),1.96*s(v)[1]))
    d=lambda a,b,n: print("  %-14s %+.4f se %.4f (95%% %.4f)"%((n,)+(s(a-b)[0],s(a-b)[1],1.96*s(a-b)[1])))
    d(o["gb"],o["sa"],"gb-sa"); d(o["sa"],o["q_12.0_1.0"],"sa-q(12,1)"); d(o["gb"],o["q_12.0_1.0"],"gb-q(12,1)")
    if seed==20260912: d(o["q_50.0_0.5"],o["gb"],"q(50,.5)-gb"); d(o["q_50.0_0.5"],o["sa"],"q(50,.5)-sa")
    res[seed]={k:float(v.mean()) for k,v in o.items()}
    res[seed].update(gb_sa=float((o["gb"]-o["sa"]).mean()),sa_q=float((o["sa"]-o["q_12.0_1.0"]).mean()))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "redraw_panel_b.json"), "w"), indent=1)
