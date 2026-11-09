#!/usr/bin/env python3
r"""Geometry of the concept figure (schematic, no simulation data).

An energy landscape with two wells (the stored words c1, c2) and the corrupted
branches c1+e, c2+e.  Saved to figures/data/fig_concept.npz and read by the render
script, so that restyling the labels never touches the geometry or the render."""
from pathlib import Path
import numpy as np
from matplotlib import colormaps

DATA = Path(__file__).resolve().parent.parent / "figures" / "data"
W1, W2 = np.array([-1.35, 0.0]), np.array([1.35, 0.0])


def energy(x, y):
    g = lambda c, s: np.exp(-((x-c[0])**2 + (y-c[1])**2)/(2*s**2))
    return 1.25 + 0.022*(x**2 + 1.6*y**2) - 1.05*g(W1, 0.62) - 1.05*g(W2, 0.62)


x = np.linspace(-2.55, 2.55, 511)
y = np.linspace(-1.45, 1.45, 291)
X, Y = np.meshgrid(x, y)
Z = energy(X, Y)

# Turbo restricted to its cold half: the warm end is kept for the quantum objects.
t = (Z - Z.min())/(Z.max() - Z.min())
rgb = colormaps["turbo"](0.10 + 0.50*t)[..., :3]
# Contour lines baked into the vertex colours, so the relief reads at small size.
lev = 0.22
d = np.abs(((Z + lev/2) % lev) - lev/2)
rgb = rgb*(1 - 0.30*np.exp(-(d/0.006)**2))[..., None]

np.savez_compressed(DATA/"fig_concept.npz", X=X, Y=Y, Z=Z, rgb=rgb, w1=W1, w2=W2,
                    warm=colormaps["turbo"](0.88)[:3], blue=colormaps["turbo"](0.12)[:3], gold=colormaps["turbo"](0.70)[:3],
                    grey=np.array([0.45, 0.45, 0.45]))
print("wrote", DATA/"fig_concept.npz", Z.min(), Z.max())
