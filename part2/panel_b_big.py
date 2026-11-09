#!/usr/bin/env python3
"""Regenerates figures/data/fix_panel_b_big.npz, the cache plotted as Fig. 2(b)
(recall overlap versus input noise f, four retrieval rules, 60 disorder draws).

Protocol (identical to fixes_compute.block_A, with 60 instead of 20 networks):
n=8 spins, L=3 random +-1 patterns streamed through the memristor couplings
(eta=1/8, gamma=0.01, dt=1), 17 noise levels f=0,1/32,...,1/2 with
nflip=floor(f*n+1/2) flipped spins, 100 trials per (network, f) with the target
drawn uniformly from the L patterns.  Rules: quantum anneal (Omega0=1, T=12,
K=120 steps, bias h=0.6), simulated annealing on the biased problem (h=0.6),
zero-T greedy descent without bias (per_greedy) and with the same bias h=0.6
(per_greedy_bias).  Stored keys (all (60, 17) except noise, n_nets, n_trials):
  per_q, per_sa, per_greedy, per_greedy_bias : mean sign-blind overlap with target
  ident   : fraction of trials on which SA and biased greedy return the
            identical spin configuration
  exact_g : fraction of trials on which the UNBIASED greedy descent recovers the
            target exactly (overlap > 0.999)

Random numbers: ONE stream, numpy default_rng(SEED), SEED = 20260911, consumed
network after network exactly as in the original run (it also drives the
random sweep order of the greedy/SA routines through sim_qhnm.RNG).  With the
default seed the output is bit-identical to the cache shipped in the
repository.  Another draw: --seed N --out other.npz.

Checkpoint/resume: after every network the finished rows AND the full state of
the random generator are written atomically to <out>_ckpt.npz; a restarted run
continues from there and gives the same numbers as an uninterrupted run.

Runtime: about 3.5 s per network on one core, about 4 min in total.
Usage:   python3 part2/panel_b_big.py [--seed 20260911] [--out PATH] [--force]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_qhnm as m
from fixes_compute import greedy_biased_batch, corrupt

DATA = Path(__file__).resolve().parent.parent / "figures" / "data"
NET, TR = 60, 100
n, L, eta, gamma = 8, 3, 1.0 / 8, 0.01
noise = np.arange(0.0, 0.5001, 1.0 / 32.0)
K = len(noise)
KEYS = ("q", "sa", "greedy", "greedy_bias")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=20260911)
ap.add_argument("--out", type=Path, default=DATA / "fix_panel_b_big.npz")
ap.add_argument("--nets", type=int, default=NET)
ap.add_argument("--force", action="store_true", help="overwrite an existing output")
ap.add_argument("--stop-after", type=int, default=None,
                help="(testing) stop after this many networks, keep checkpoint")
a = ap.parse_args()
OUT = a.out
CKPT = OUT.with_name(OUT.stem + "_ckpt.npz")
if OUT.exists() and not a.force:
    print(f"{OUT} exists, nothing to do (use --force to overwrite)"); sys.exit(0)

per = {k: np.zeros((a.nets, K)) for k in KEYS}
ident = np.zeros((a.nets, K)); exact_g = np.zeros((a.nets, K))
m.RNG = np.random.default_rng(a.seed)
start = 0
if CKPT.exists():
    z = np.load(CKPT, allow_pickle=False)
    if int(z["seed"]) == a.seed and int(z["nets"]) == a.nets:
        start = int(z["done"])
        for k in KEYS: per[k] = z[f"per_{k}"]
        ident, exact_g = z["ident"], z["exact_g"]
        m.RNG.bit_generator.state = json.loads(str(z["rng_state"]))
        print(f"resuming from network {start}", flush=True)

spins = m.spin_table(n)
for net in range(start, a.nets):
    RNG = m.RNG
    P = RNG.choice([-1, 1], size=(L, n))
    J, _ = m.stream_memristor_couplings(P, eta=eta, gamma=gamma, dt=1.0,
                                        steps_per_pattern=1)
    for fi, f in enumerate(noise):
        nflip = int(np.floor(f * n + 0.5))
        tb = P[RNG.integers(0, L, size=TR)].copy()
        nb = corrupt(RNG, tb, nflip, n)
        Rq = m.retrieve_quantum_anneal_batch(J, nb, Omega0=1.0, spins=spins)
        Rsa = m.retrieve_classical_sa_batch(J, nb)
        Rg = m.retrieve_classical_batch(J, nb)
        Rb = greedy_biased_batch(J, nb)
        per["q"][net, fi] = m.overlap_batch(Rq, tb).mean()
        per["sa"][net, fi] = m.overlap_batch(Rsa, tb).mean()
        per["greedy"][net, fi] = m.overlap_batch(Rg, tb).mean()
        per["greedy_bias"][net, fi] = m.overlap_batch(Rb, tb).mean()
        ident[net, fi] = np.mean(np.all(Rsa == Rb, axis=1))
        exact_g[net, fi] = np.mean(m.overlap_batch(Rg, tb) > 0.999)
    tmp = CKPT.with_name(CKPT.stem + "_tmp.npz")
    np.savez(tmp, done=net + 1, seed=a.seed, nets=a.nets, ident=ident,
             exact_g=exact_g, rng_state=json.dumps(m.RNG.bit_generator.state),
             **{f"per_{k}": v for k, v in per.items()})
    os.replace(tmp, CKPT)
    print(f"  net {net + 1}/{a.nets}", flush=True)
    if a.stop_after is not None and net + 1 >= a.stop_after and net + 1 < a.nets:
        print("stopping early (test)"); sys.exit(0)

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, noise=noise, n_nets=a.nets, n_trials=TR, ident=ident,
         exact_g=exact_g, **{f"per_{k}": v for k, v in per.items()})
os.remove(CKPT)
print(f"wrote {OUT}", flush=True)
