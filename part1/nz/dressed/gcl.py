# Domkniecie gaussowskie (LNA): srednia eb i wariancja S zaleza od cyklu napedu.
# eb' = -kappa e^{-S/2} sin eb - g (eb+2chi) - 2 chi'; S' = -2(kappa e^{-S/2} cos eb + g) S + kappa*delta
# O = -e^{-S/2} sin eb.  Liczymy iloraz G(A) = area(A)/area_MF(A) w dwu rzedach A.
import numpy as np
def area_gcl(delta,A,g,w,n=512,kappa=2.0):
    T=2*np.pi/w;h=T/n;t=h*np.arange(n)
    def f(y,tt):
        eb,S=y; ch=A*np.sin(w*tt); cd=A*w*np.cos(w*tt); e=np.exp(-S/2)
        return np.array([-kappa*e*np.sin(eb)-g*(eb+2*ch)-2*cd, -2*(kappa*e*np.cos(eb)+g)*S+kappa*delta])
    y=np.array([0.,delta/2]); 
    for per in range(100000):
        Ys=np.empty((n,2)); y0=y.copy()
        for k in range(n):
            Ys[k]=y;tt=t[k];k1=f(y,tt);k2=f(y+h*k1/2,tt+h/2);k3=f(y+h*k2/2,tt+h/2);k4=f(y+h*k3,tt+h);y=y+h*(k1+2*k2+2*k3+k4)/6
        if np.abs(y-y0).max()<1e-11: break
    O=-np.exp(-Ys[:,1]/2)*np.sin(Ys[:,0])
    return np.sum(O*2*A*w*np.cos(2*A*np.sin(w*t))*np.cos(w*t))*h
