#!/usr/bin/env python3
# Fig. of the Nakajima-Zwanzig equation. Data in figures/data/paper_nz_kernel.npz (computed if missing), drawing only reads it.
import glob, json, os, sys
import numpy as np
from scipy.linalg import expm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NZ = os.environ.get("NZ_DIR", os.path.join(REPO, "results", "part1", "nz"))
CMP = os.path.join(NZ, "cmp")
GRID = os.path.join(NZ, "grid")
DATA = os.path.join(REPO, "figures", "data", "paper_nz_kernel.npz")
ETA = 0.5
DS = [0.025, 0.05, 0.1, 0.25]


def nrule(D):
    return int(np.ceil(np.pi / (2 * D))) - 1


def u22(D, tau):
    # propagator of the block n>=2 at w=0
    G = ETA / D
    nm = nrule(D)
    n = np.arange(2, nm + 1)
    a = 4 * n * D
    M = np.diag(G * (np.cos(a) - 1) + 0j)
    B = 0.5j * G * np.sin(a)
    for i in range(len(n) - 1):
        M[i, i + 1] = B[i]
        M[i + 1, i] = B[i + 1]
    return np.array([expm(M * t)[0, 0] for t in tau]).real


def scalar_err(r):
    # scalar mean-field law of the paper, |c|=1, on the stored times of the records
    from scipy.integrate import solve_ivp
    Z = np.load(f"{GRID}/D{r['D']}_A{r['A']}_w{r['om']}_g{r['g']}.npz")
    A, om, g, ts, x = r["A"], r["om"], r["g"], Z["ts"], Z["x"]
    ar = []
    for s in (1, -1):
        F = lambda t, v: [ETA * np.cos(4 * v[0] - s * A * np.sin(om * t)) - g * v[0]]
        q = solve_ivp(F, (0, ts[-1]), [np.pi / 8], t_eval=np.concatenate(([0.0], ts)), method="DOP853", rtol=1e-9, atol=1e-11)
        y = np.cos(4 * q.y[0][1:] - s * A * np.sin(om * ts))
        xs = s * x
        ar.append(0.5 * np.sum(xs * np.roll(y, -1) - np.roll(xs, -1) * y))
    Am = 0.5 * (ar[0] + ar[1])
    return dict(A=Am, err=(Am - r["Amc"]) / abs(r["Amc"]) * 100)


def compute():
    R = [json.load(open(f)) for f in sorted(glob.glob(CMP + "/*.json"))]
    for r in R:
        if r["g"] == 0.01:
            r["MF"] = scalar_err(r)
    tau = np.linspace(0, 3, 301)
    out = dict(tau=tau, Ds=np.array(DS))
    for D in DS:
        out[f"u22_{D}"] = u22(D, tau)
    # (b) area against omega, Delta=0.25, A=1, gamma=0.01
    pts = sorted([r for r in R if r["D"] == 0.25 and r["A"] == 1.0 and r["g"] == 0.01], key=lambda r: r["om"])
    out["om"] = np.array([r["om"] for r in pts])
    out["Amc"] = np.array([abs(r["Amc"]) for r in pts])
    out["Ase"] = np.array([r["Ase"] for r in pts])
    for m in ("H", "MF", "GC"):
        out["A" + m] = np.array([abs(r[m]["A"]) for r in pts])
    # (c) max |error| against Delta, gamma=0.01, whole grid
    allD = sorted({r["D"] for r in R})
    out["allD"] = np.array(allD)
    for m in ("H", "MF", "GC"):
        out["err" + m] = np.array([max(abs(r[m]["err"]) for r in R if r["D"] == D and r["g"] == 0.01 and "err" in r[m]) for D in allD])
    # (d) max |error| of H against gamma/gamma_eff
    gs = sorted({r["g"] for r in R})
    out["gr"] = np.array([g / (g + 4 * ETA) for g in gs])
    for D in DS:
        out[f"errg_{D}"] = np.array([max(abs(r["H"]["err"]) for r in R if r["D"] == D and r["g"] == g and "err" in r["H"]) for g in gs])
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez(DATA, **out)


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paneltags import tag, place
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 7, "axes.labelsize": 7,
                         "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5,
                         "lines.linewidth": 0.9, "lines.markersize": 2.8, "legend.frameon": False,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    Z = np.load(DATA)
    col = plt.cm.turbo(np.linspace(0.1, 0.9, 4))
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.55), layout="constrained")
    for D, c in zip(DS, col):
        ax[0].plot(Z["tau"], Z[f"u22_{D}"], color=c, label=rf"${D}$")
    ax[0].axhline(0, color="0.7", lw=0.4)
    ax[0].axvline(0.5, color="0.5", lw=0.5, ls=":")
    ax[0].set(xlabel=r"$\tau$", ylabel=r"$\mathcal{K}_\mathrm{NZ}(\tau)/\mathcal{K}_\mathrm{NZ}(0)$", xlim=(0, 3))
    ax[0].legend(title=r"$\Delta\theta$", title_fontsize=6, handlelength=1.2, labelspacing=0.2)
    tag(ax[0], "(a)")
    ax[1].errorbar(Z["om"], Z["Amc"], Z["Ase"], fmt="o", color="k", label="records", zorder=5)
    for m, lab, c, ls in (("H", "Eq.~(20)", col[0], "-"), ("GC", "local", col[3], "--"), ("MF", "mean field", col[2], ":")):
        ax[1].plot(Z["om"], Z["A" + m], ls, color=c, label=lab)
    ax[1].set(xscale="log", xlabel=r"$\omega$", ylabel=r"$|A_Z|$")
    ax[1].legend(handlelength=1.6, labelspacing=0.2)
    tag(ax[1], "(b)")
    for m, lab, c, mk in (("MF", "mean field", col[2], "s"), ("GC", "local", col[3], "^"), ("H", "Eq.~(20)", col[0], "o")):
        ax[2].plot(Z["allD"], Z["err" + m], "-" + mk, color=c, label=lab)
    ax[2].set(xscale="log", yscale="log", xlabel=r"$\Delta\theta$", ylabel=r"max.\ area error (\%)")
    ax[2].set_ylim(0.1, 3e4)
    ax[2].legend(handlelength=1.6, labelspacing=0.2, loc="upper left")
    tag(ax[2], "(c)")
    for D, c in zip(DS, col):
        ax[3].plot(Z["gr"], Z[f"errg_{D}"], "-o", color=c, label=rf"${D}$")
    ax[3].set(xscale="log", yscale="log", xlabel=r"$\gamma/\gamma_\mathrm{eff}$", ylabel=r"max.\ area error (\%)")
    ax[3].legend(title=r"$\Delta\theta$", title_fontsize=6, handlelength=1.2, labelspacing=0.2)
    tag(ax[3], "(d)")
    place(fig)
    fig.savefig(os.path.join(REPO, "figures", "paper_nz_kernel.pdf"))


if __name__ == "__main__":
    if not os.path.exists(DATA):
        compute()
    draw()
    Z = np.load(DATA)
    print({k: np.round(Z[k], 4).tolist() for k in ("allD", "errH", "errGC", "errMF", "gr", "om", "Amc", "AH", "AGC", "AMF")})
    print({f"errg_{D}": np.round(Z[f"errg_{D}"], 3).tolist() for D in DS})
    print("u22 half width", {D: float(Z["tau"][np.argmax(Z[f"u22_{D}"] < 0.5)]) for D in DS})
