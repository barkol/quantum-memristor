# Symbolic verification of T1, T2, T3 from the generator of P(theta,t)
import sympy as sp
th,chi,D,G,g,th0=sp.symbols('theta chi Delta Gamma gamma theta_0',real=True)
n=sp.symbols('n',integer=True,positive=True)
chid=sp.symbols('chidot',real=True)
phi=4*th-2*chi
f=sp.exp(sp.I*n*phi)
pH=(1+sp.cos(phi))/2;pV=1-pH
fp=f.subs(th,th+D); fm=f.subs(th,th-D)
drift=-g*(th-th0)*sp.diff(f,th)
expl=sp.diff(f,chi)*chid
gen_signed=G*(pH*(fp-f)+pV*(fm-f))+drift+expl
gen_uni=G*pH*(fp-f)+drift+expl
a=4*n*D
A_s=G*(sp.cos(a)-1);B_s=sp.I*G/2*sp.sin(a)
A_u=G/2*(sp.exp(sp.I*a)-1);B_u=A_u/2
def fn(k): return sp.exp(sp.I*(n+k)*phi)
# "w" split into state-dependent part: -i n (4 gamma (th-th0)+2 chidot) f
wfull=4*g*(th-th0)+2*chid
rhs_s=(A_s-sp.I*n*wfull)*f+B_s*(fn(1)+fn(-1))
rhs_u=(A_u-sp.I*n*wfull)*f+B_u*(fn(1)+fn(-1))
def z(e):
    e=e.rewrite(sp.exp)
    e=sp.simplify(sp.expand(sp.simplify(e/f)))
    return e
print('signed residual',z(gen_signed-rhs_s))
print('unipolar residual',z(gen_uni-rhs_u))
# sign test numeric too
vals={th:0.37,chi:0.21,D:0.13,G:3.3,g:0.07,th0:0.11,chid:0.9,n:3}
print('num signed',complex((gen_signed-rhs_s).subs(vals).evalf()),'uni',complex((gen_uni-rhs_u).subs(vals).evalf()))
# mean theta (needed to close w) 
thf=th
print('d<theta>/dt signed integrand',sp.simplify(G*(pH*D-pV*D)-g*(th-th0)),'  unipolar',sp.simplify(G*pH*D-g*(th-th0)))
# First Kramers-Moyal check of the sign of drift term:  drift generator for dtheta/dt=-g(th-th0) is  -g(th-th0) d/dth
# T3
s=sp.symbols('s')
K0=sp.simplify(B_s.subs(n,1)*B_s.subs(n,2))
print('K(0)=B1B2=',K0)
print('small Delta K(0)',sp.series(K0.subs(G,sp.Symbol('eta')/D),D,0,3))
# continued fraction / Laplace of elimination for constant w: verify by 3-level truncated linear algebra
w=sp.symbols('w',real=True)
A1,A2,A3,B1,B2,B3=sp.symbols('A1 A2 A3 B1 B2 B3')
c1=sp.symbols('c1');c2,c3=sp.symbols('c2 c3')
sol=sp.solve([(s-A2+2*sp.I*w)*c2-B2*(c1+c3),(s-A3+3*sp.I*w)*c3-B3*c2],[c2,c3],dict=True)[0]
g3=1/(s-A3+3*sp.I*w)
g2=1/(s-A2+2*sp.I*w-B2*B3*g3)
print('c2 - g2*B2*c1 =',sp.simplify(sol[c2]-g2*B2*c1))
print('kernel Laplace check: B1*c2 = K~ c1 with K~=B1B2 g2:',sp.simplify(B1*sol[c2]-B1*B2*g2*c1))
