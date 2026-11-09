r"""Fig: reservoir network of quantum memristors (matplotlib PDF, replacing the
former inline TikZ). Blue solid arrows = quantum state transfer; red dashed
self-loops = classical memory-parameter updates (each memristor updates its own
theta from its own detections). Column width, house style."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle; figstyle.apply()
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import os

nodes={'i1':(0,1.5,r'$|\psi_1\rangle$','q'),'i2':(0,0,r'$|\psi_2\rangle$','q'),
       'm1':(2,1.5,r'$M_1$','m'),'m2':(2,0,r'$M_2$','m'),'m3':(4,0.75,r'$M_3$','m'),
       'out':(6,0.75,r'$|\psi_{\rm out}\rangle$','q')}
LB='#cfe2f3'; LO='#fce5cd'; F=plt.rcParams['font.size']
# COLUMN figure: generate at the template width (7.0 in) so that, displayed at
# \columnwidth (~3.375 in), the in-figure font lands at the uniform ~9 pt shared
# by all manuscript figures (apparent = base*3.375/7.0).  The earlier 3.4-in
# width rendered the labels at ~18 pt and crowded the boxes.  Tight axes limits
# keep the diagram filling the column with no wasted corner whitespace.
fig,ax=plt.subplots(figsize=(7.0,5.0)); P={}
QW,QH=1.40,1.04   # quantum-state node ellipse: wide enough for |psi_out> with breathing space
BW,BH=1.06,0.74  # memristor box full width/height (enlarged)
for k,(x,y,lbl,t) in nodes.items():
    if t=='q': p=Ellipse((x,y),QW,QH,facecolor=LB,edgecolor='k',lw=1.6,zorder=3)
    else:      p=FancyBboxPatch((x-BW/2,y-BH/2),BW,BH,boxstyle='round,pad=0.015',
                                facecolor=LO,edgecolor='k',lw=1.6,zorder=3)
    ax.add_patch(p); P[k]=p
    ax.text(x,y,lbl,ha='center',va='center',fontsize=0.95*F,color='k',zorder=4)

def arr(a,b,col,ls='-',rad=0.0,lw=2.2):
    ax.add_patch(FancyArrowPatch(posA=(nodes[a][0],nodes[a][1]),posB=(nodes[b][0],nodes[b][1]),
        patchA=P[a],patchB=P[b],connectionstyle=f'arc3,rad={rad}',arrowstyle='-|>',
        mutation_scale=18,color=col,lw=lw,ls=ls,zorder=2))

# blue quantum state-transfer arrows
for a,b in [('i1','m1'),('i2','m2'),('m1','m3'),('m2','m3'),('m3','out')]: arr(a,b,figstyle.BLUE)

def self_loop(k,col,dy=1):
    """Small red-dashed curved self-loop on memristor box k -- the internal
    classical memory-parameter (theta) update of that device."""
    x,y,_,_=nodes[k]
    yedge=y+dy*BH/2          # top (dy=+1) or bottom (dy=-1) box edge
    x0,x1=x-0.30,x+0.30
    ax.add_patch(FancyArrowPatch(posA=(x0,yedge),posB=(x1,yedge),
        connectionstyle=f'arc3,rad={-dy*0.9}',arrowstyle='-|>',
        mutation_scale=12,color=col,lw=1.8,ls='--',zorder=5))

self_loop('m1',figstyle.RED,dy=+1)
self_loop('m2',figstyle.RED,dy=-1)
self_loop('m3',figstyle.RED,dy=+1)

ax.legend(handles=[Line2D([0],[0],color=figstyle.BLUE,lw=2.2,label='quantum state transfer'),
                   Line2D([0],[0],color=figstyle.RED,lw=2.0,ls='--',label='memory-parameter update')],
          loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=1,frameon=False,fontsize=0.80*F,
          handlelength=1.6,labelspacing=0.3,borderpad=0.1)
# tight limits -> diagram fills the column, minimal margins
ax.set_xlim(-0.80,6.80); ax.set_ylim(-0.95,2.55); ax.set_aspect('equal'); ax.axis('off')
fig.subplots_adjust(left=0.01,right=0.99,top=0.84,bottom=0.01)
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','papers','figures','memristor_network.pdf')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out,bbox_inches='tight',pad_inches=0.02)
fig.savefig(out.replace('.pdf','.png'),dpi=150,bbox_inches='tight',pad_inches=0.02)
print('saved',os.path.abspath(out))
