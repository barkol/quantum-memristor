# add-on: fragility of the assemblage-specific gain, closed forms, realistic table
from sdp_budget_lib import *
print('purity test: absent-branch state p|+><+|+(1-p)I/2; N=3,V=1, eps=eta^2 split')
def assem2(N,V,epre,epost,mode,p):
    rho=V*rho0+(1-V)*np.eye(4)/4
    rhoB=rho.reshape(2,2,2,2).trace(axis1=0,axis2=2)
    pl=p*np.outer(plus,plus.conj())+(1-p)*I2/2
    null=epre*(1-epost)*rhoB+(1-epre)*pl
    out=[]
    for P in SET[N]:
        sp=epre*epost*cond(rho,P,1); sm=epre*epost*cond(rho,P,-1)
        out.append([sp,sm,null] if mode=='third' else [sp+null/2,sm+null/2])
    return out
for p in (1,.99,.95,.9,.7,.0):
    t=bis(lambda e:robust(assem2(3,1,e**.5,e**.5,'third',p)),.05,1)
    r=bis(lambda e:robust(assem2(3,1,e**.5,e**.5,'random',p)),.05,1)
    print(f'p={p}: third eps={t:.4f} eta={t**.5:.4f} {-10*log10(t**.5):.2f} dB | random eps={r:.4f} eta={r**.5:.4f} {-10*log10(r**.5):.2f} dB'); sys.stdout.flush()
print('post threshold at fixed pre (V=1,N=3,third,p=1):')
for pre in (1,.7,.5,.3):
    t=bis(lambda e:robust(assem2(3,1,pre,e,'third',1)),.05,1); print(f'pre={pre}: post>{t:.4f}'); sys.stdout.flush()
# replaced model with noisy pre-gate photon k: k input eta_pre|+><+|+(1-eta_pre)I/2, post-gate white eta each
def replaced(eta,N=3):
    rk=eta*np.outer(plus,plus.conj())+(1-eta)*I2/2
    r=CZ@np.kron(rk,np.outer(plus,plus.conj()))@CZ.conj().T
    # white noise on each photon, visibility eta
    def dep(r,q):
        # q=0 k;1 k+1
        if q==0:
            rB=r.reshape(2,2,2,2).trace(axis1=0,axis2=2); return eta*r+(1-eta)*np.kron(I2/2,rB)
        rA=r.reshape(2,2,2,2).trace(axis1=1,axis2=3); return eta*r+(1-eta)*np.kron(rA,I2/2)
    r=dep(dep(r,0),1)
    return [[cond(r,P,1),cond(r,P,-1)] for P in SET[N]]
t=bis(lambda e:robust(replaced(e)),.5,1); print('replaced+pre-gate noise SDP eta',t,-10*log10(t),'dB; S3 test eta^2(1+2eta)=sqrt3 ->',)
from scipy.optimize import brentq
e=brentq(lambda e:e*e*(1+2*e)-sqrt(3),.5,1); print(e,-10*log10(e),'dB')
# closed table
print('per-pass budget table dB')
def L(eps_th,pre,etaA):
    # eps=eta^(1+pre)*etaA
    if eps_th is None or eps_th>=etaA: return None
    return -10*log10((eps_th/etaA)**(1/(1+pre)))
for N in (3,2):
 for Vg in (1,.95,.9):
  V=Vg**2
  r=1/(V*sqrt(N)); t=closed_third(N,V)
  for etaA in (1,.9,.63):
    f=lambda x:'empty' if x is None else f'{x:.2f}'
    print(f'N={N} Vg={Vg} etaA={etaA}: random: viaLoop {f(L(r,1,etaA))} direct {f(L(r,0,etaA))} | third: viaLoop {f(L(t,1,etaA))} direct {f(L(t,0,etaA))}')
