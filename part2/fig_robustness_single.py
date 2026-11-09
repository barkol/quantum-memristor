r"""Single-device no-denoising figure (persp_robustness_single.pdf).

Correct geometry: a HWP with fast axis at angle theta (measured from |0>)
reflects a linear polarization at angle a to 2*theta - a [Eq. (2) of Part II:
U(theta) (cos a, sin a)^T = (cos(2theta-a), sin(2theta-a))^T].
The reflection axis drawn dashed is at theta (NOT 2*theta); angle preservation
|delta_in| = |delta_out| is the point of the figure.

Pure drawing (no simulation data).  Output: ../papers/figures/persp_robustness_single.pdf
"""
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
figstyle.apply(base_fontsize=9)
import matplotlib.pyplot as plt

THETA = np.deg2rad(30.0)     # HWP axis, measured from |0> (vertical)
A     = np.deg2rad(78.0)     # ideal input angle from |0>
DELTA = np.deg2rad(14.0)     # perturbation

def dirvec(alpha):
    """Unit vector at angle alpha measured from the vertical |0> axis."""
    return np.array([np.sin(alpha), np.cos(alpha)])

fig, ax = plt.subplots(figsize=(3.3, 3.3))
ax.set_aspect("equal"); ax.axis("off")

# disc + basis axes
tt = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(tt), np.sin(tt), color="0.75", lw=1.0)
ax.plot([0, 0], [-1.02, 1.02], color="0.55", lw=0.8)
ax.plot([-1.02, 1.02], [0, 0], color="0.55", lw=0.8)
ax.text(0.03, 1.06, r"$\ket{0}$", ha="left", va="bottom", fontsize=9)
ax.text(1.06, 0.02, r"$\ket{1}$", ha="left", va="bottom", fontsize=9)

def polar_line(alpha, color, ls, lw, label=None, r=0.97, zorder=3):
    v = dirvec(alpha)
    ax.annotate("", xy=tuple(r*v), xytext=tuple(-r*v),
                arrowprops=dict(arrowstyle="<->,head_width=0.18,head_length=0.30",
                                color=color, lw=lw, linestyle=ls), zorder=zorder)
    if label:
        ax.plot([], [], color=color, ls=ls, lw=lw, label=label)

# HWP axis at theta (dashed grey)
vax = dirvec(THETA)
ax.plot([-1.1*vax[0], 1.1*vax[0]], [-1.1*vax[1], 1.1*vax[1]],
        color="0.35", lw=1.0, ls=(0, (5, 3)), zorder=2)
ax.text(1.14*vax[0], 1.14*vax[1], r"HWP axis ($\theta$)", color="0.25",
        ha="left", va="bottom", fontsize=8)

# inputs (solid) and reflected outputs (dashed): a -> 2 theta - a
polar_line(A,                 figstyle.BLUE, "-",  2.0, r"ideal in $\ket{\psi_0}$")
polar_line(A + DELTA,         figstyle.RED,  "-",  2.0, r"perturbed in $\ket{\tilde\psi}$")
polar_line(2*THETA - A,         figstyle.BLUE, (0, (4, 2.5)), 1.6, r"$\hat U$ output (ideal)")
polar_line(2*THETA - A - DELTA, figstyle.RED,  (0, (4, 2.5)), 1.6, r"$\hat U$ output (perturbed)")

# delta arcs (same opening angle on input and output side)
def arc(a1, a2, r, color):
    aa = np.linspace(a1, a2, 40)
    ax.plot(r*np.sin(aa), r*np.cos(aa), color=color, lw=1.1)
    mid = dirvec(0.5*(a1 + a2))
    ax.text(1.12*r*mid[0], 1.12*r*mid[1], r"$\delta$", color=color,
            ha="center", va="center", fontsize=8)
arc(A, A + DELTA, 0.80, "0.2")
arc(2*THETA - A - DELTA, 2*THETA - A, 0.62, "0.2")

ax.set_title(r"HWP $\hat U(\theta)$: reflect polarization across the axis at $\theta$",
             fontsize=8.5, pad=8)
ax.legend(loc="lower left", fontsize=6.8, frameon=True, framealpha=0.9,
          borderpad=0.5, handlelength=2.2)
ax.set_xlim(-1.25, 1.35); ax.set_ylim(-1.25, 1.3)

out = Path(__file__).resolve().parent.parent / "papers" / "figures" / "persp_robustness_single.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("Saved:", out)
