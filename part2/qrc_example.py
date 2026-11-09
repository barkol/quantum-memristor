#!/usr/bin/env python3
r"""Target and prediction of the nominal reservoir on one NARMA-10 test sequence (seed 0 of qrc_benchmark.py), panel (a) of the reservoir figure of
Part I. Output: figures/data/fig_qrc_example.npz."""
from pathlib import Path
import numpy as np
import qrc_benchmark as b, sim_qrc_narma as q
u, t = b.data(0)
n, yp, yt = q.evaluate(b.reservoir(u, b.GAMMA0, b.DEV), t, b.TRAIN, b.TEST)
np.savez(Path(__file__).resolve().parent.parent/"figures"/"data"/"fig_qrc_example.npz", yt=yt, yp=yp, nmse=n)
print("seed 0 NMSE", round(float(n), 3))
