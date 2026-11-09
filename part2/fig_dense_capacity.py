r"""Dense (modern) Hopfield capacity: two-body (CZ) vs three-body (CCZ) energy
functions.  Storage capacity L_max (largest pattern set retrievable at >=90%
success from 8%-corrupted inputs, by zero-temperature p-body energy descent)
versus network size n.

The descent is VECTORISED across a whole batch of probe states (the per-spin
update runs over n spins, batched over B = L*trials states), so we can afford
high statistics (many realisations/trials) and obtain smooth, monotonic curves.
"""
import os

import numpy as np
import figstyle
figstyle.apply()
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

rng = np.random.default_rng(11)


def descend_batch(S, P, p, nn, sweeps=80):
    """Zero-T greedy p-body energy descent on a batch S (B, nn) of states.
    Energy E=-sum_k m_k^p with overlaps m=S P^T; flip s_i when it lowers E."""
    m = S @ P.T                                  # (B, L) overlaps with each pattern
    for _ in range(sweeps):
        moved = np.zeros(S.shape[0], dtype=bool)
        for i in rng.permutation(nn):
            ds = -2.0 * S[:, i]                  # (B,) change in s_i if flipped
            mnew = m + np.outer(ds, P[:, i])     # (B, L)
            dE = -(np.sum(mnew ** p, axis=1) - np.sum(m ** p, axis=1))
            flip = dE < -1e-9
            S[flip, i] *= -1
            m[flip] = mnew[flip]
            moved |= flip
        if not moved.any():
            break
    return S


def succ(P, p, nn, f=0.08, trials=60):
    """Fraction of (pattern, trial) probes recovered to >=99% spin overlap."""
    L = P.shape[0]
    nflip = max(1, int(round(f * nn)))
    reps = np.repeat(np.arange(L), trials)       # target pattern per probe
    S = P[reps].astype(float).copy()             # (L*trials, nn)
    fl = np.argsort(rng.random((S.shape[0], nn)), axis=1)[:, :nflip]
    S[np.repeat(np.arange(S.shape[0]), nflip), fl.ravel()] *= -1
    out = descend_batch(S, P, p, nn)
    return np.mean(np.mean(out == P[reps], axis=1) >= 0.99)


def cap(nn, p, thr=0.9, reals=40, trials=60):
    """Mean (over realisations) largest L retrievable at >= thr success."""
    out = []
    for _ in range(reals):
        Lc = 0
        for L in range(1, nn * nn):
            if succ(rng.choice([-1, 1], size=(L, nn)), p, nn, trials=trials) >= thr:
                Lc = L
            else:
                break
        out.append(Lc)
    return np.mean(out), np.std(out) / np.sqrt(len(out))


n2 = np.array([8, 12, 16, 20, 24, 28, 32])
n3 = np.array([8, 11, 14, 17, 20, 23])
c2, e2 = np.array([cap(n, 2) for n in n2]).T
c3, e3 = np.array([cap(n, 3) for n in n3]).T
print("p2", dict(zip(n2.tolist(), np.round(c2, 2).tolist())))
print("p3", dict(zip(n3.tolist(), np.round(c3, 2).tolist())))
s2 = np.polyfit(np.log(n2), np.log(c2), 1)[0]
s3 = np.polyfit(np.log(n3), np.log(c3), 1)[0]
print(f"slopes  p2={s2:.2f}  p3={s3:.2f}")

fig, ax = plt.subplots(figsize=figstyle.SINGLE)
ax.errorbar(n2, c2, yerr=e2, fmt='o', color=figstyle.BLUE, ms=8, capsize=3,
            label=rf"2-body (CZ): slope ${s2:.2f}$")
ax.errorbar(n3, c3, yerr=e3, fmt='s', color=figstyle.RED, ms=8, capsize=3,
            label=rf"3-body (CCZ): slope ${s3:.2f}$")
ax.loglog(n2, 0.14 * n2, ':', color=figstyle.GREY, lw=1.6,
          label=r"$0.14\,n$ (classical bound)")
for nn, cc, col in [(n2, c2, figstyle.BLUE), (n3, c3, figstyle.RED)]:
    fpar = np.polyfit(np.log(nn), np.log(cc), 1)
    xx = np.linspace(nn.min(), nn.max(), 40)
    ax.loglog(xx, np.exp(np.polyval(fpar, np.log(xx))), '--', color=col, lw=1.3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"network size $n$"); ax.set_ylabel(r"storage capacity $L_{\max}$")
ax.legend(loc="upper left", fontsize=0.78 * plt.rcParams["font.size"])
ax.set_xticks([8, 12, 16, 20, 24, 32])
ax.set_xticklabels(["8", "12", "16", "20", "24", "32"])
ax.xaxis.set_minor_formatter(NullFormatter())   # no stray 3x10^1 minor label
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "papers", "figures", "paper_dense_capacity.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("saved", os.path.abspath(out))
