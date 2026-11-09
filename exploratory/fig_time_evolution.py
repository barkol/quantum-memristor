"""Fig: time evolution of the photonic memristor -- input drive v(t) (dashed) and
output response i(t)=G(theta)v (solid), same i-v model/params as the pinched
hysteresis figure (eta=0.15, gamma=0.1, omega=0.3, theta0=pi/8). Single panel,
house style, legend above the axes (clear of the curves)."""
import numpy as np
from scipy.integrate import solve_ivp
import figstyle; figstyle.apply()
import matplotlib.pyplot as plt
eta, gamma, omega, th0 = 0.15, 0.1, 0.3, np.pi/8
v = lambda t: np.sin(omega*t)
G = lambda th: np.sin(2*th)**2
T = 2*np.pi/omega; tmax = 4*T; t = np.linspace(0, tmax, 2000)
th = solve_ivp(lambda t,th: eta*v(t)-gamma*(th-th0), [0,tmax],[th0],
               t_eval=t, rtol=1e-10, atol=1e-12).y[0]
i = G(th)*v(t)
fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.axhline(0, color=figstyle.GREY, lw=0.8)
ax.plot(t, v(t), '--', color="0.45", lw=2, label=r"input drive $v(t)$")
ax.plot(t, i, color=figstyle.BLUE, lw=2.2, label=r"output response $i(t)$")
ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"signal"); ax.set_xlim(0, tmax)
# mark the transient -> settled ("frozen" periodic limit cycle) transition.
# The memory relaxes on ~1/gamma; the output locks into a repeating waveform
# after about one drive period.
t_trans = T
ax.axvspan(0, t_trans, color="0.90", lw=0, zorder=0)
ax.axvline(t_trans, color=figstyle.GREY, ls=":", lw=1.2, zorder=1)
_xt = ax.get_xaxis_transform()
ax.text(t_trans * 0.5, 0.90, "transient", transform=_xt, ha="center",
        va="top", color="0.35", fontsize=0.8 * plt.rcParams["font.size"])
ax.text((t_trans + tmax) / 2.0, 0.90, "frozen (periodic)", transform=_xt,
        ha="center", va="top", color="0.35",
        fontsize=0.8 * plt.rcParams["font.size"])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
fig.tight_layout()
figstyle.cap_ticks(fig)          # house rule: 4-5 stripped labels per axis
import os; out=os.path.join(os.path.dirname(__file__),"..","figures","memristor_time_evolution.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf",".png"), dpi=150)
print("saved", os.path.abspath(out))
