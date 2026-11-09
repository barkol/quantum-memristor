"""
Frame sequences for Beamer \\animategraphics (Tier-2 animations).

Generates PNG frame stacks under ../figures/anim/<name>/frame_###.png, one
stack per animation. Each is derived from the SAME verified models used for
the static figures, so the moving versions stay faithful:

  pinched  -- the pinched i-v hysteresis loop drawing itself over one cycle
  freq     -- the loop collapsing as the drive frequency omega rises past gamma
  theta    -- the leaky-integrator memory theta(t) lagging the drive v(t)
  basis    -- the three measurement-basis responses (Z loop / X flat / Y arch)

All frames in a stack share identical figure size and axis limits (required by
\\animategraphics). usetex is disabled here for speed (120+ frames); mathtext
serif keeps them visually close to the static usetex figures.

Run:  python3 make_animations.py [name ...]   (default: all)
"""

import os
import sys

import numpy as np

import figstyle
figstyle.apply()
import matplotlib.pyplot as plt

# Fast path for many frames: mathtext serif instead of a LaTeX call per frame.

import quantum_memristor_phase_presentation as qm   # pinched-loop model
import basis_hysteresis as bh                        # basis-dependent model

# rcParams AFTER the model imports: qm/bh call figstyle.apply() on import,
# which would otherwise clobber font.size (back to 18) and set savefig.bbox='tight'.
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    # CONSTANT APPARENT FONT: each figure's width (inches) is set equal to its
    # on-slide display width (DISP_W), so scale=1 and these point sizes ARE the
    # apparent on-slide sizes -- identical text size across every figure.
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.9,
    "savefig.bbox": None,
    "legend.frameon": True,
})

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")
ANIM_DIR = os.path.join(FIG_DIR, "anim")

BLUE, RED, GREY = figstyle.BLUE, figstyle.RED, figstyle.GREY
THIRD = getattr(figstyle, "THIRD", "0.45")

# Compact figure sizes for slide animations (text stays legible after scaling).
ANIM_SINGLE = (4.3, 3.2)

# --- Constant apparent on-slide font size -------------------------------------
TW_IN = 5.512          # beamer aspectratio=169 \textwidth, inches (measured)
APPARENT = 9.5         # target on-slide font size (pt); == rcParams font.size
# On-slide DISPLAY width (inches) of each animation = wfac * colfrac * TW_IN.
DISP_W = {
    "pinched": 1.98, "bloch_hyst": 1.79, "theta": 1.87, "transient": 1.82,
    "bloch_transient": 1.75, "freq": 2.23, "basis": 2.65, "fq": 2.54,
    "bloch_traj": 2.91, "bloch_traj_mod": 2.91, "cphase": 3.31, "chsh": 2.91,
}
def anim_figsize(name, aspect=0.75):
    """Figure size whose WIDTH equals the on-slide display width -> scale 1:1,
    so the rcParams point sizes render at that exact apparent size on the slide."""
    w = DISP_W[name]
    return (w, w * aspect)


def _stack_dir(name):
    d = os.path.join(ANIM_DIR, name)
    os.makedirs(d, exist_ok=True)
    # clear stale frames so frame counts never mismatch the \animategraphics range
    for f in os.listdir(d):
        if f.startswith("frame_") and f.endswith(".png"):
            os.remove(os.path.join(d, f))
    return d


def _save(fig, d, k):
    # NOT zero-padded: \animategraphics builds names frame_0.png .. frame_N.png
    fig.savefig(os.path.join(d, f"frame_{k}.png"), dpi=132)
    plt.close(fig)


def _box(ax, xlim, ylim, xlabel, ylabel):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# ----------------------------------------------------------------------
def anim_pinched(nframes=36):
    """Settled i-v loop revealed progressively, with a moving marker."""
    d = _stack_dir("pinched")
    v, i = qm.loop_last_cycles(n_cycles=8, skip=7)   # one settled cycle
    # close the loop
    v = np.append(v, v[0]); i = np.append(i, i[0])
    vline = np.linspace(v.min() * 1.05, v.max() * 1.05, 50)
    xlim = (v.min() * 1.15, v.max() * 1.15)
    ylim = (i.min() - 0.05, i.max() + 0.62)
    n = len(v)
    for k in range(nframes):
        frac = (k + 1) / nframes
        j = max(2, int(frac * n))
        fig, ax = plt.subplots(figsize=(4.8, 2.7))
        ax.plot(vline, qm.memoryless_line(vline), ls=":", color="0.45", lw=1.4,
                label="frozen memory")
        ax.plot(v[:j], i[:j], color=BLUE, lw=1.8)
        ax.plot(v[j - 1], i[j - 1], "o", color=BLUE, ms=5)
        ax.plot(0, 0, "o", color=RED, ms=4)
        _box(ax, xlim, ylim, r"Input drive $v$", r"Output $i=G(\theta)\,v$")
        ax.legend(loc="upper center", ncol=2, framealpha=1.0, facecolor="white", edgecolor="0.7")
        fig.tight_layout()
        _save(fig, d, k)
    print(f"pinched: {nframes} frames -> {d}")


# ----------------------------------------------------------------------
def anim_freq(nframes=30):
    """Loop collapses as omega/gamma rises from ~peak through fast drive."""
    d = _stack_dir("freq")
    ratios = np.linspace(1.0, 12.0, nframes)        # omega/gamma
    gamma = qm.GAMMA
    # fixed axes from the fattest loop
    v0, i0 = qm.loop_last_cycles(omega=gamma * 1.0, n_cycles=10, skip=8)
    xlim = (v0.min() * 1.15, v0.max() * 1.15)
    # i = G(theta) v can be negative (v<0 -> i<0): show the FULL loop, both signs.
    ylim = (i0.min() - 0.06, i0.max() + 0.10)
    vline = np.linspace(xlim[0], xlim[1], 50)
    for k, r in enumerate(ratios):
        omega = gamma * r
        v, i = qm.loop_last_cycles(omega=omega, n_cycles=12, skip=10)
        v = np.append(v, v[0]); i = np.append(i, i[0])
        fig, ax = plt.subplots(figsize=(4.8, 2.7))
        ax.plot(vline, qm.memoryless_line(vline), ls=":", color="0.45", lw=1.4)
        ax.plot(v, i, color=BLUE, lw=1.8)
        ax.text(0.05, 0.92, rf"$\omega/\gamma = {r:0.1f}$", transform=ax.transAxes,
                va="top", ha="left", color=RED)
        _box(ax, xlim, ylim, r"Input drive $v$", r"Output $i=G(\theta)\,v$")
        fig.tight_layout()
        _save(fig, d, k)
    print(f"freq: {nframes} frames -> {d}")


# ----------------------------------------------------------------------
def anim_theta(nframes=40):
    """Memory theta(t) (red) lagging the drive v(t) (blue) over time."""
    d = _stack_dir("theta")
    t, theta = qm.solve_theta(n_cycles=4)
    v = qm.drive(t)
    # normalise both to [-1,1] for a shared axis
    vN = v / np.max(np.abs(v))
    thN = (theta - qm.THETA0)
    thN = thN / np.max(np.abs(thN))
    xlim = (t.min(), t.max())
    ylim = (-1.2, 1.8)
    n = len(t)
    for k in range(nframes):
        j = max(2, int((k + 1) / nframes * n))
        fig, ax = plt.subplots(figsize=(4.8, 2.7))
        ax.axhline(0, color=GREY, lw=0.6)
        ax.plot(t[:j], vN[:j], color=BLUE, lw=1.5, label=r"drive $v(t)$")
        ax.plot(t[:j], thN[:j], color=RED, lw=1.8,
                label=r"memory $\theta(t)$")
        ax.plot(t[j - 1], thN[j - 1], "o", color=RED, ms=5)
        _box(ax, xlim, ylim, r"time $t$", r"normalised amplitude")
        ax.legend(loc="upper right", framealpha=1.0, facecolor="white",
                  edgecolor="0.7")
        fig.tight_layout()
        _save(fig, d, k)
    print(f"theta: {nframes} frames -> {d}")


# ----------------------------------------------------------------------
def anim_basis(nframes=34):
    """Z (loop) / X (flat) / Y (arch) basis responses drawn progressively."""
    d = _stack_dir("basis")
    bases = ["Z", "X", "Y"]
    cols = {"Z": BLUE, "X": RED, "Y": THIRD}
    styles = {"Z": "-", "X": "-", "Y": "--"}
    lws = {"Z": 1.8, "X": 2.2, "Y": 1.8}
    labs = {"Z": r"$Z$-basis", "X": r"$X$-basis", "Y": r"$Y$-basis"}
    curves = {}
    t_start = 3 * bh.T_CYCLE
    for b in bases:
        t, th = bh.solve_theta(b)
        mask = t >= t_start
        I = bh.input_signal(t)[mask]
        O = bh.output_observable(b, th, t)[mask]
        curves[b] = (I, O)
    n = len(curves["Z"][0])
    for k in range(nframes):
        j = max(2, int((k + 1) / nframes * n))
        fig, ax = plt.subplots(figsize=(4.8, 2.7))
        for b in bases:
            I, O = curves[b]
            ax.plot(I[:j], O[:j], color=cols[b], ls=styles[b], lw=lws[b],
                    label=labs[b])
        _box(ax, (-1.1, 1.1), (-1.25, 1.25),
             r"Input $I=\sin 2\alpha$", r"Output $\langle\hat\sigma_z\rangle$")
        # legend fully OUTSIDE, above the axes (horizontal) -> covers no curve
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
                  framealpha=1.0, facecolor="white", edgecolor="0.7",
                  columnspacing=1.2, handlelength=1.6, borderpad=0.3)
        fig.tight_layout()
        _save(fig, d, k)
    print(f"basis: {nframes} frames -> {d}")


def anim_fq(nframes=None):
    """Schematic operation of a FULLY-QUANTUM memristor: a SYNCHRONIZED TRAIN of
    photons, injected at constant intervals (one injection per loop round-trip),
    flows through a 3-qubit programmable gate via two coherent delay loops
    (target -> program -> control), no measurement inside the loops. At steady
    state several photons populate the device at once -- as one returns through a
    loop the next is entering the gate -- and the coherent memory |Theta> forms."""
    from matplotlib.patches import FancyBboxPatch, Wedge, FancyArrowPatch
    d = _stack_dir("fq")

    # ---- geometry -----------------------------------------------------
    GX0, GX1, GY0, GY1 = 3.0, 5.0, 1.3, 4.0          # gate box
    yT, yP, yC = 3.4, 2.6, 1.8                        # port heights
    xin, xout = GX0, GX1                              # port x (box edges)
    DET = (6.7, yC)

    def bez(P0, P1, P2, s):
        s = np.asarray(s)
        x = (1 - s) ** 2 * P0[0] + 2 * (1 - s) * s * P1[0] + s ** 2 * P2[0]
        y = (1 - s) ** 2 * P0[1] + 2 * (1 - s) * s * P1[1] + s ** 2 * P2[1]
        return x, y

    # delay-loop control points (A: t-out->program, arc over top; B: p-out->control, arc under)
    LA = ((xout, yT), (3.9, 6.8), (xin, yP))
    LB = ((xout, yP), (3.9, -1.2), (xin, yC))

    # ---- photon path: list of (kind, pts, nframes, tag) ---------------
    segs = [
        ("line", ((0.6, yT), (xin, yT)), 7, "enter"),
        ("line", ((GX0, yT), (GX1, yT)), 6, "gate1"),
        ("bez", LA, 10, "loopA"),
        ("line", ((GX0, yP), (GX1, yP)), 6, "gate2"),
        ("bez", LB, 10, "loopB"),
        ("line", ((GX0, yC), (GX1, yC)), 6, "gate3"),
        ("line", ((GX1, yC), DET), 6, "detect"),
    ]
    total = sum(s[2] for s in segs)

    def photon_at(k):
        """Return (x, y, tag) for global frame index k."""
        acc = 0
        for kind, pts, nf, tag in segs:
            if k < acc + nf:
                s = (k - acc) / max(nf - 1, 1)
                if kind == "line":
                    (x0, y0), (x1, y1) = pts
                    return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, tag
                else:
                    x, y = bez(pts[0], pts[1], pts[2], s)
                    return float(x), float(y), tag
            acc += nf
        return DET[0], DET[1], "done"

    def tags_upto(k):
        """Set of segment tags the photon has reached by frame k."""
        acc, seen = 0, set()
        for kind, pts, nf, tag in segs:
            if k >= acc:
                seen.add(tag)
            acc += nf
        return seen

    sgrid = np.linspace(0, 1, 60)
    SPACING = 17                 # path-frames between consecutive photons in the train
    NF = total                   # 51 frames = 3 seamless periods (51 = 3*SPACING)
    DXIN = 0.4                   # x advance per frame along the entry lane (for queued dots)

    def train_at(k):
        """Synchronized train at frame k. Returns (in_device, queued):
        in_device = [(x,y,tag)] for photons currently inside the device,
        queued    = [(x,y)] faint photons on the entry approach (next to arrive).
        The lattice {base + m*SPACING} is identical for k and k+SPACING -> seamless."""
        base = k % SPACING
        in_device = [photon_at(p) for p in range(base, total, SPACING)]
        queued = []
        p = base - SPACING
        while p > -3 * SPACING:           # photons not yet entered, on the approach lane
            x = 0.6 + p * DXIN            # extend the entry-lane slope to x < 0.6
            if x > -1.55:
                queued.append((x, yT))
            p -= SPACING
        return in_device, queued

    XLO, XHI = -2.3, 8.0                     # diagram x-extent (room for edge text)
    XMID = 2.85                              # device center (gate at x=4)
    for k in range(NF):
        in_device, queued = train_at(k)
        cur_tags = {t for (_, _, t) in in_device}
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        ax.set_xlim(XLO, XHI)
        ax.set_ylim(-2.7, 6.1)
        ax.margins(0)
        ax.axis("off")
        ax.set_position([0.005, 0.01, 0.99, 0.98])

        # delay loops (red dashed); brighten whichever loop the train occupies now
        for L, t in ((LA, "loopA"), (LB, "loopB")):
            x, y = bez(L[0], L[1], L[2], sgrid)
            active = (t in cur_tags)
            ax.plot(x, y, ls="--", color=RED, lw=2.2 if active else 1.2,
                    alpha=1.0 if active else 0.5, zorder=1)
        # coherent delay loops: "1 step" labels placed OUTSIDE the arcs (right of
        # the apex) so the looping photon dots never sit on top of the text
        ax.text(5.75, 4.7, "delay\n1 step", color=RED, ha="left", va="center",
                linespacing=0.95)
        ax.text(2.3, 0.2, "delay\n1 step", color=RED, ha="center", va="center",
                linespacing=0.95)

        # gate box (highlight whenever any train photon is mid-gate)
        on_gate = bool(cur_tags & {"gate1", "gate2", "gate3"})
        ax.add_patch(FancyBboxPatch((GX0, GY0), GX1 - GX0, GY1 - GY0,
                     boxstyle="round,pad=0.02",
                     ec=BLUE, lw=1.8,
                     fc=("#cfe2f6" if on_gate else "#e9f2f8"), zorder=2))
        # gate title ABOVE the box (outside, one line) -> never overlaps photons
        ax.text((GX0 + GX1) / 2, 6.0, "3-qubit programmable gate",
                ha="center", va="center", color=BLUE, zorder=3)

        # ports -- labels lifted above their rows so they clear the entry arrow head
        for y, lab in ((yT, "target"), (yP, "program"), (yC, "control")):
            ax.plot([GX0], [y], ">", color=BLUE, ms=5, zorder=3)
            ax.text(GX0 - 0.18, y + 0.20, lab, ha="right", va="bottom")
            ax.plot([GX1], [y], "o", color=BLUE, ms=3, zorder=3)

        # synchronized injection: shorter entry arrow (stops clear of port label) +
        # queued photons on the approach lane + interval label
        ax.annotate("", xy=(GX0 - 0.05, yT), xytext=(GX0 - 1.05, yT),
                    arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.2))
        ax.text(-2.2, 5.0,
                "synchronized\nphoton train",
                va="center", ha="left", color="0.30", linespacing=0.95)
        for (qx, qy) in queued:
            ax.plot([qx], [qy], "o", color=BLUE, ms=8, alpha=0.30, zorder=4)
            ax.plot([qx], [qy], "o", color="white", ms=3, alpha=0.30, zorder=5)

        # detector (lit whenever a photon is exiting / being measured)
        det_on = bool(cur_tags & {"detect", "done"})
        ax.add_patch(Wedge(DET, 0.34, 250, 110, width=0.34,
                     fc=(RED if det_on else "white"), ec="0.3", lw=1.2, zorder=2))
        ax.annotate("", xy=(DET[0] - 0.36, yC), xytext=(GX1 + 0.06, yC),
                    arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.2))
        ax.text(DET[0], yC - 0.45, "measure", ha="center", va="top")

        # the train photons (each a distinct photon flowing through the device)
        for (px, py, t) in in_device:
            ax.plot([px], [py], "o", color=BLUE, ms=10, zorder=5)
            ax.plot([px], [py], "o", color="white", ms=3.5, zorder=6)

        # bottom caption: two clear lines BELOW the whole diagram, clear of photons.
        # Long ket equation simplified to the compact label "coherent memory |Theta>".
        ax.text(XMID, -1.65, r"coherent memory  $|\Theta\rangle$",
                ha="center", va="center", color=BLUE)
        ax.text(XMID, -2.25, "no measurement inside loops",
                ha="center", va="center", style="italic", color="0.35")

        _save(fig, d, k)
    print(f"fq (train): {NF} frames, {len(train_at(0)[0])} photons in-device -> {d}")


def anim_fq2(nframes=None):
    """TWO fully-quantum memristors whose photon OUTPUTS are not measured but
    interact in a CZ gate. Two synchronized photon trains (clocked together)
    each recirculate through a memristor's coherent delay loop; the paired
    photons then meet in a controlled-Z gate and leave ENTANGLED and unmeasured
    (a CHSH test on the exits certifies it). Strong loop loss would recover two
    independent classical memristors."""
    from matplotlib.patches import FancyBboxPatch
    d = _stack_dir("fq2")
    ENT = "#b5179e"                                   # entanglement-link colour

    # ---- geometry -----------------------------------------------------
    yA, yB = 3.7, 1.3                                 # the two rails
    BXL, BXR = 2.2, 4.0                               # memristor box x-extent
    xCZ = 5.65                                        # CZ-gate column
    XEXIT = 7.7
    XEN0 = 0.2                                        # photon becomes visible here

    def bez(P0, P1, P2, s):
        s = np.asarray(s)
        x = (1 - s) ** 2 * P0[0] + 2 * (1 - s) * s * P1[0] + s ** 2 * P2[0]
        y = (1 - s) ** 2 * P0[1] + 2 * (1 - s) * s * P1[1] + s ** 2 * P2[1]
        return x, y

    # per-rail path: enter -> into box -> coherent loop (arc) -> out -> CZ -> exit
    SEGS = [("enter", 8), ("m_in", 4), ("loop", 12), ("m_out", 6),
            ("cz", 6), ("exit", 12)]
    total = sum(n for _, n in SEGS)                   # 48
    SPACING = total // 3                              # 16  -> 3 photons / rail
    NF = SPACING * 3                                  # 48 frames, 3 seamless periods

    def loop_pts(y, ldir, s):
        return bez((2.7, y), (3.0, y + ldir * 2.0), (3.3, y), s)

    def photon_at(p, y, ldir):
        """(x, y, tag) at path-index p on a rail at height y; ldir=+1 loop above."""
        acc = 0
        for tag, nf in SEGS:
            if p < acc + nf:
                s = (p - acc) / max(nf - 1, 1)
                if tag == "enter":  return XEN0 + (BXL - XEN0) * s, y, tag
                if tag == "m_in":   return BXL + (2.7 - BXL) * s, y, tag
                if tag == "loop":
                    x, yy = loop_pts(y, ldir, s); return float(x), float(yy), tag
                if tag == "m_out":  return 3.3 + (5.15 - 3.3) * s, y, tag
                if tag == "cz":     return 5.15 + (6.15 - 5.15) * s, y, tag
                if tag == "exit":   return 6.15 + (XEXIT - 6.15) * s, y, tag
            acc += nf
        return XEXIT, y, "done"

    DXIN = (BXL - XEN0) / 7.0                          # entry-lane slope for queued dots
    sgrid = np.linspace(0, 1, 60)
    RAILS = ((yA, +1), (yB, -1))

    for k in range(NF):
        base = k % SPACING
        ps = list(range(base, total, SPACING))         # synchronized: same indices on both rails
        # current tags per rail
        tags = {(rail_i): {photon_at(p, y, ld)[2] for p in ps}
                for rail_i, (y, ld) in enumerate(RAILS)}
        cz_active = any("cz" in tags[i] for i in (0, 1))

        fig, ax = plt.subplots(figsize=(4.8, 2.7))
        ax.set_xlim(-1.7, 8.2)
        ax.set_ylim(-1.4, 6.1)
        ax.axis("off")

        # --- per-rail device: coherent loop, memristor box, labels ---
        for (y, ld), name in zip(RAILS, ("Memristor 1", "Memristor 2")):
            lx, lyv = loop_pts(y, ld, sgrid)
            ax.plot(lx, lyv, ls="--", color=RED, lw=1.8, alpha=0.9, zorder=1)
            ax.text(3.0, y + ld * 1.32, "coherent\nloop", color=RED, ha="center",
                    va="center", fontsize=11, alpha=0.9)
            ax.add_patch(FancyBboxPatch((BXL, y - 0.5), BXR - BXL, 1.0,
                         boxstyle="round,pad=0.02", ec=BLUE, lw=1.6,
                         fc="#e9f2f8", zorder=2))
            ax.text((BXL + BXR) / 2, y + ld * 0.30, name, ha="center", va="center",
                    fontsize=11, color=BLUE, zorder=3)   # off rail-centre so photons don't cover it
            # entry arrow
            ax.annotate("", xy=(BXL - 0.04, y), xytext=(-1.6, y),
                        arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.1))
            # output wire from box to CZ and on to exit
            ax.plot([BXR, XEXIT], [y, y], color="0.55", lw=1.0, zorder=1)

        ax.text(-1.62, (yA + yB) / 2,
                "synchronized\ntrains\n(both clocked\nto the CZ)",
                fontsize=11, va="center", ha="left", color="0.30")

        # --- CZ gate: vertical link + two control dots, brightens when a pair meets ---
        czc = ENT if cz_active else "0.25"
        ax.plot([xCZ, xCZ], [yB, yA], color=czc, lw=2.4 if cz_active else 1.6, zorder=3)
        for y in (yA, yB):
            ax.plot([xCZ], [y], "o", color=czc, ms=9, zorder=4)
        ax.text(xCZ, yA + 0.7, "CZ gate", ha="center", va="bottom",
                fontsize=11, color=czc, fontweight="bold")
        ax.text(xCZ, yB - 0.7, "no measurement", ha="center", va="top",
                fontsize=11, style="italic", color="0.35")

        # --- exit annotation ---
        ax.text(XEXIT + 0.05, (yA + yB) / 2, "entangled\noutput\nCHSH $>2$",
                fontsize=11, va="center", ha="left", color=ENT)

        # --- entanglement links for pairs that have passed the CZ ---
        for p in ps:
            xa, ya, ta = photon_at(p, yA, +1)
            xb, yb, tb = photon_at(p, yB, -1)
            if ta in ("cz", "exit") and tb in ("cz", "exit"):
                ax.plot([xa, xb], [ya, yb], ls=(0, (2, 2)), color=ENT,
                        lw=1.6, alpha=0.9, zorder=4)

        # --- queued photons on the approach lanes (next to arrive) ---
        for (y, ld) in RAILS:
            p = base - SPACING
            while p > -3 * SPACING:
                x = XEN0 + p * DXIN
                if x > -1.55:
                    ax.plot([x], [y], "o", color=BLUE, ms=7, alpha=0.30, zorder=4)
                    ax.plot([x], [y], "o", color="white", ms=2.6, alpha=0.30, zorder=5)
                p -= SPACING

        # --- the train photons on both rails ---
        for (y, ld) in RAILS:
            for p in ps:
                px, py, t = photon_at(p, y, ld)
                ax.plot([px], [py], "o", color=BLUE, ms=9, zorder=6)
                ax.plot([px], [py], "o", color="white", ms=3.2, zorder=7)

        fig.tight_layout()
        _save(fig, d, k)
    print(f"fq2 (two memristors + CZ): {NF} frames, {len(ps)} photons/rail -> {d}")


def anim_transient(nframes=44):
    """Transient dynamics: from rest the output current i(t)=G(theta)v builds up
    and locks into its steady nonlinear response as the memory settles. Companion
    to anim_theta (which shows the drive v(t) and the lagging memory theta(t))."""
    d = _stack_dir("transient")
    t, theta = qm.solve_theta(n_cycles=4)
    v = qm.drive(t)
    i = np.sin(2 * theta) ** 2 * v                  # G(theta) = sin^2(2 theta)
    vN = v / np.max(np.abs(v))
    iN = i / np.max(np.abs(i))
    xlim = (t.min(), t.max())
    ylim = (-1.25, 1.85)
    n = len(t)
    # mark where the response has effectively settled (last cycle)
    t_settle = t.min() + 0.75 * (t.max() - t.min())
    for k in range(nframes):
        j = max(2, int((k + 1) / nframes * n))
        fig, ax = plt.subplots(figsize=(4.8, 2.7))
        ax.axhline(0, color=GREY, lw=0.6)
        ax.axvspan(t_settle, t.max(), color="0.85", alpha=0.5, lw=0)
        ax.plot(t[:j], vN[:j], color=BLUE, lw=1.4, label=r"drive $v(t)$")
        ax.plot(t[:j], iN[:j], color="#b5179e", lw=1.9,
                label=r"output $i(t)=G(\theta)\,v$")
        ax.plot(t[j - 1], iN[j - 1], "o", color="#b5179e", ms=5)
        if t[j - 1] >= t_settle:
            ax.text(0.5 * (t_settle + t.max()), 1.55, "settled",
                    ha="center", va="center", fontsize=9, color="0.45")
        _box(ax, xlim, ylim, r"time $t$", r"normalised")
        ax.legend(loc="upper left", framealpha=1.0,
                  facecolor="white", edgecolor="0.7")
        fig.tight_layout()
        _save(fig, d, k)
    print(f"transient: {nframes} frames -> {d}")


def _bloch_stack(name, xs, ys, zs, nframes, accumulate):
    """Render a Bloch-sphere trajectory frame stack with qutip.
    accumulate=True -> the path draws itself from rest (transient);
    accumulate=False -> full meridian arc shown, a marker sweeps it (settled)."""
    import qutip as qt
    d = _stack_dir(name)
    n = len(xs)
    for k in range(nframes):
        j = max(2, int((k + 1) / nframes * n))
        b = qt.Bloch()
        b.figsize = [4.0, 4.0]               # square; width = on-slide width -> 1:1
        b.frame_alpha = 0.10
        b.sphere_alpha = 0.04
        b.font_size = 13                                 # constant apparent font (9.5 pt)
        b.vector_color = [RED]
        b.vector_width = 3
        b.point_color = [BLUE]
        b.point_marker = ["o"]
        b.point_size = [22]
        jl = j if accumulate else n
        b.add_points([xs[:jl], ys[:jl], zs[:jl]], meth="l")   # trajectory line
        b.add_vectors([xs[j - 1], ys[j - 1], zs[j - 1]])      # current state
        b.render()
        ax = b.axes                                            # zoom out slightly -> top margin for |0>
        ax.set_xlim3d(-1.25, 1.25)
        ax.set_ylim3d(-1.25, 1.25)
        ax.set_zlim3d(-1.25, 1.25)
        b.fig.savefig(os.path.join(d, f"frame_{k}.png"), dpi=132)
        plt.close(b.fig)
    print(f"{name}: {nframes} frames -> {d}")


def anim_bloch_hyst(nframes=36):
    """Bloch trajectory of the polarization qubit over ONE settled drive cycle --
    companion to the pinched-loop / hysteresis animation. After a HWP at angle
    theta(t) the state is linear polarization at 2theta: Bloch (sin4theta,0,cos4theta),
    swinging along the X-Z meridian (no circular component -> sigma_y = 0)."""
    t, theta = qm.solve_theta(n_cycles=8)
    T = 2 * np.pi / qm.OMEGA
    th = theta[t >= (t.max() - T)]                            # one settled cycle
    xs, ys, zs = np.sin(4 * th), np.zeros_like(th), np.cos(4 * th)
    _bloch_stack("bloch_hyst", xs, ys, zs, nframes, accumulate=False)


def anim_bloch_transient(nframes=44):
    """Bloch trajectory from REST: the state starts at the bias point (theta0) and
    the swing grows to its steady amplitude -- companion to the transient animation."""
    t, theta = qm.solve_theta(n_cycles=4)
    xs, ys, zs = np.sin(4 * theta), np.zeros_like(theta), np.cos(4 * theta)
    _bloch_stack("bloch_transient", xs, ys, zs, nframes, accumulate=True)


def _memristor_bloch_traj(name, modulated, nframes=46):
    """Quantum-trajectory view of the output qubit DRIVEN BY THE SAME MODULATION
    as the pinched hysteresis loops. H(t)=2*thetadot(t)*sigma_y reproduces the
    memristor Bloch path (sin4theta,0,cos4theta) with theta(t) from qm.solve_theta;
    a tap channel (amplitude damping, rate modulated by the drive) unravels it into
    individual mcsolve trajectories (coloured). The red ensemble-average vector's
    LENGTH then DECOHERES and RECOVERS once per drive cycle -- the non-Markovian
    coherence revival of the device's memory. modulated=True spreads the input
    starting points along an input-modulation arc."""
    import qutip as qt
    from matplotlib.lines import Line2D
    d = _stack_dir(name)
    sx, sy, sz, sm = qt.sigmax(), qt.sigmay(), qt.sigmaz(), qt.sigmam()
    t, theta = qm.solve_theta(n_cycles=2)
    thetadot = qm.ETA * qm.drive(t) - qm.GAMMA * (theta - qm.THETA0)
    H = [[sy, 2.0 * thetadot]]                        # SAME modulation as the pinched loop
    per = 2 * np.pi / qm.OMEGA
    # tap rate modulated by the drive: strong on the down-swing (decohere), ~0 on the
    # up-swing where the drive re-coheres the state (recover) -> visible revival
    gamma_t = 0.55 * (1 + np.cos(qm.OMEGA * t)) / 2 + 0.02
    c_ops = [[sm, np.sqrt(gamma_t)]]
    eops = [sx, sy, sz]

    def psi(th):                                      # R_y(4 th)|0> -> Bloch (sin4th,0,cos4th)
        return (np.cos(2 * th) * qt.basis(2, 0) + np.sin(2 * th) * qt.basis(2, 1)).unit()

    cols = ["#1f77b4", "#2ca02c", "#e31a8c"]   # blue / green / magenta -- all distinct, and from RED
    ORANGE = "#ff7f0e"
    sett = t >= (t.max() - per)
    th_amp = 0.5 * (theta[sett].max() - theta[sett].min())
    th_in = ([qm.THETA0 + th_amp * np.sin(2 * np.pi * i / len(cols)) for i in range(len(cols))]
             if modulated else [qm.THETA0] * len(cols))
    trajs = []
    for i, th0 in enumerate(th_in):
        mc = qt.mcsolve(H, psi(th0), t, c_ops, e_ops=eops, ntraj=1, seeds=i + 1)
        trajs.append([np.array(e) for e in mc.expect])
    rho0 = sum((qt.ket2dm(psi(th0)) for th0 in th_in),
               0 * qt.ket2dm(psi(th_in[0]))) / len(th_in)
    me = qt.mesolve(H, rho0, t, c_ops, e_ops=eops)
    ax_, ay_, az_ = (np.array(e) for e in me.expect)
    rlen = np.sqrt(ax_ ** 2 + ay_ ** 2 + az_ ** 2)
    print(f"  {name}: |r_avg| decoheres/recovers in [{rlen.min():.2f}, {rlen.max():.2f}]")

    # input-modulation arc + start markers (Bloch angle = 4*theta); qutip maps (x,y,z)->(y,-x,z)
    tarc = np.linspace(qm.THETA0 - th_amp, qm.THETA0 + th_amp, 40)
    arc_x, arc_z = np.sin(4 * tarc), np.cos(4 * tarc)
    st_x = np.array([np.sin(4 * th0) for th0 in th_in])
    st_z = np.array([np.cos(4 * th0) for th0 in th_in])
    n = len(t)
    for k in range(nframes):
        j = max(2, int((k + 1) / nframes * n))
        b = qt.Bloch()
        b.figsize = [4.6, 2.6]               # LANDSCAPE: short height (no vertical overflow);
        b.frame_alpha = 0.10                 # extra width on the right holds the legend
        b.sphere_alpha = 0.04
        b.font_size = 13                                 # constant apparent font (9.5 pt)
        for (tx, ty, tz), c in zip(trajs, cols):
            b.add_points([tx[:j], ty[:j], tz[:j]], meth="l", colors=[c])
        b.vector_color = [RED]
        b.vector_width = 3
        b.add_vectors([ax_[j - 1], ay_[j - 1], az_[j - 1]])
        b.render()
        ax = b.axes
        ax.set_xlim3d(-1.25, 1.25)                             # zoom out slightly -> top margin for |0>
        ax.set_ylim3d(-1.25, 1.25)
        ax.set_zlim3d(-1.25, 1.25)
        if modulated:
            ax.plot(np.zeros_like(arc_x), -arc_x, arc_z, color=ORANGE, lw=1.6, ls="--", alpha=0.9)
            ax.scatter(np.zeros(len(st_x)), -st_x, st_z, color=ORANGE, s=26, marker="D",
                       depthshade=False, zorder=6)
        handles = []
        if modulated:
            handles.append(Line2D([0], [0], color=ORANGE, ls="--", lw=1.8, marker="D",
                                  label="input start (modulated)"))
        handles += [Line2D([0], [0], color=c, lw=2, label=f"trajectory {i + 1}")
                    for i, c in enumerate(cols)]
        handles.append(Line2D([0], [0], color=RED, lw=2.5, label="ensemble average"))
        # shrink the 3D axes to the LEFT half-ish so the sphere keeps its size but
        # frees a right-hand margin; put the legend there (adds NO height).
        ax.set_position([0.0, 0.0, 0.74, 1.0])
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5),
                  ncol=1, fontsize=7.5, framealpha=1.0, facecolor="white",
                  edgecolor="0.7", handlelength=1.4, borderpad=0.4, labelspacing=0.4,
                  columnspacing=1.0)
        b.fig.savefig(os.path.join(d, f"frame_{k}.png"), dpi=132)
        plt.close(b.fig)
    print(f"{name}: {nframes} frames -> {d}")


def anim_bloch_traj(nframes=46):
    _memristor_bloch_traj("bloch_traj", modulated=False, nframes=nframes)


def anim_bloch_traj_mod(nframes=46):
    _memristor_bloch_traj("bloch_traj_mod", modulated=True, nframes=nframes)


def anim_chsh(nframes=48):
    """CHSH value S of two independent memristor-output qubits entangled by a
    CPHASE(phi) gate, as phi ramps 0->pi: S rises from the classical bound 2 to
    the Tsirelson bound 2*sqrt(2). Single clean panel (setup is in the slide text)."""
    import qutip as qt
    d = _stack_dir("chsh")
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi_in = qt.tensor(plus, plus)
    paulis = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]

    def cphase(phi):
        return qt.Qobj(np.diag([1, 1, 1, np.exp(1j * phi)]), dims=[[2, 2], [2, 2]])

    phis = np.linspace(0.0, np.pi, 130)
    Sv = []
    for phi in phis:
        rho = qt.ket2dm(cphase(phi) * psi_in)
        T = np.array([[qt.expect(qt.tensor(a, b), rho).real for b in paulis] for a in paulis])
        ev = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
        Sv.append(2.0 * np.sqrt(max(ev[0] + ev[1], 0.0)))
    Sv = np.array(Sv)
    TS = 2 * np.sqrt(2)
    print(f"  chsh: S(0)={Sv[0]:.2f}, S(pi)={Sv[-1]:.2f} (Tsirelson {TS:.2f})")
    n = len(phis)
    for k in range(nframes):
        j = max(2, int((k + 1) / nframes * n))
        phi_now, S_now = phis[j - 1] / np.pi, Sv[j - 1]
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        ax.axhspan(1.8, 2.0, color="0.88", zorder=0)              # classical (LHV) band
        ax.axhline(2.0, color="0.4", ls="--", lw=1.1)
        ax.axhline(TS, color=RED, ls=":", lw=1.3)
        ax.plot(phis[:j] / np.pi, Sv[:j], color=BLUE, lw=2.4)
        ax.plot(phi_now, S_now, "o", color=BLUE, ms=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(1.83, 3.02)
        ax.set_xlabel(r"CPHASE phase $\phi/\pi$")
        ax.set_ylabel(r"CHSH value $S$")
        ax.text(0.5, 1.90, "classical bound", color="0.4", ha="center", va="center", fontsize=9)
        ax.text(0.04, TS + 0.03, r"Tsirelson $2\sqrt{2}$", color=RED, va="bottom", fontsize=9)
        viol = S_now > 2.002
        ax.text(0.96, 0.08, rf"$S={S_now:.2f}$" + ("  violation!" if viol else ""),
                transform=ax.transAxes, ha="right", va="bottom",
                color=(RED if viol else "0.4"),
                fontweight=("bold" if viol else "normal"))
        fig.tight_layout()
        _save(fig, d, k)
    print(f"chsh: {nframes} frames -> {d}")


def anim_cphase(nframes=64):
    """Concurrence sweeping in as the memory builds: two driven memristors,
    fixed maximally-entangling CZ. Top: memory theta(t) building from modulated
    inputs; bottom: post-CZ concurrence C(t) drawing itself with a moving marker."""
    import cphase_two_memristors as cp
    d = _stack_dir("cphase")

    t = np.linspace(0.0, 140.0, 1600)
    eta, gamma = 0.2, 0.035
    aA, thA = cp.run_driven_memristor(t, 1.0, 0.35, eta, gamma, 0.0, 0.0)   # omega_A
    aB, thB = cp.run_driven_memristor(t, 1.0, 0.55, eta, gamma, 0.0, 0.0)   # omega_B (different)
    C = np.array([cp.concurrence_pure(
        cp.CZ @ np.kron(cp.hwp_output_state(thA[i], aA[i]),
                        cp.hwp_output_state(thB[i], aB[i]))) for i in range(len(t))])
    tau = 1.0 / gamma
    n = len(t)

    for k in range(nframes):
        j = max(2, int((k + 1) / nframes * n))
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(5.0, 3.4), sharex=True,
            gridspec_kw=dict(height_ratios=[1.0, 1.5]), layout="constrained")

        ax0.plot(t, aA, color=BLUE, lw=0.8, ls=":", alpha=0.35)
        ax0.plot(t[:j], thA[:j], color=BLUE, lw=1.6, label=r"$\theta_A,\ \omega_A=0.35$")
        ax0.plot(t[:j], thB[:j], color=RED, lw=1.6, label=r"$\theta_B,\ \omega_B=0.55$")
        ax0.set_xlim(0, 140)
        ax0.set_ylim(-1.15, 1.55)
        ax0.set_ylabel(r"$\theta$ (rad)")
        ax0.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                   columnspacing=1.2, handlelength=1.4,
                   framealpha=1.0, facecolor="white", edgecolor="0.7")

        ax1.axvspan(0, tau, color=THIRD, alpha=0.10)
        ax1.axvline(tau, color=GREY, ls=":", lw=0.9)
        ax1.text(tau + 3, 1.22, r"transient $\sim 1/\gamma$", color="0.35",
                 ha="left", va="center", fontsize=9)
        ax1.plot(t[:j], C[:j], color=BLUE, lw=1.8)
        ax1.plot(t[j - 1], C[j - 1], "o", color=BLUE, ms=6)
        ax1.set_xlim(0, 140)
        ax1.set_ylim(-0.05, 1.35)
        ax1.set_ylabel(r"concurrence $C$")
        ax1.set_xlabel(r"time $t$")

        _save(fig, d, k)
    print(f"cphase: {nframes} frames -> {d}")


ALL = {
    "pinched": anim_pinched,
    "cphase": anim_cphase,
    "freq": anim_freq,
    "theta": anim_theta,
    "transient": anim_transient,
    "basis": anim_basis,
    "fq": anim_fq,
    "fq2": anim_fq2,
    "bloch_hyst": anim_bloch_hyst,
    "bloch_transient": anim_bloch_transient,
    "bloch_traj": anim_bloch_traj,
    "bloch_traj_mod": anim_bloch_traj_mod,
    "chsh": anim_chsh,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for nm in names:
        ALL[nm]()
    print("\nDone. Frame counts:")
    for nm in names:
        d = os.path.join(ANIM_DIR, nm)
        n = len([f for f in os.listdir(d) if f.endswith(".png")])
        print(f"  {nm}: {n} frames (range 0..{n-1})")
