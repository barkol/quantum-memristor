# Pole srednie przy skonczonym A: stan okresowy, pole przez regule trapezow (spektralna). Bufor w npz.
import numpy as np, os
def area_mf(eta,A,gamma,omega,theta0=np.pi/8,n=2048):
    T=2*np.pi/omega; h=T/n; t=h*np.arange(n)
    # podwojna siatka dla RK4
    def f(u,tt): return eta*np.cos(4*(u+theta0)-2*A*np.sin(omega*tt))-gamma*u
    # start: stan ustalony przy chi~0
    u=2*A*0*np.sin(0)
    u=0.0 if theta0==np.pi/8 else 0.0
    prev=None
    for per in range(100000):
        us=np.empty(n)
        for k in range(n):
            us[k]=u; tt=t[k]
            k1=f(u,tt);k2=f(u+h*k1/2,tt+h/2);k3=f(u+h*k2/2,tt+h/2);k4=f(u+h*k3,tt+h)
            u+=h*(k1+2*k2+2*k3+k4)/6
        if abs(u-us[0])<1e-13: break
    O=np.cos(4*(us+theta0)-2*A*np.sin(omega*t))
    dS=2*A*omega*np.cos(2*A*np.sin(omega*t))*np.cos(omega*t)
    return np.sum(O*dS)*h, O.mean()
if __name__=='__main__':
    eta=.5
    for (A,g,w) in [(1e-3,0,2),(1e-3,.01,2),(1e-3,.1,.5),(.1,.01,2),(.3,.01,2),(.3,.1,20)]:
        a,_=area_mf(eta,A,g,w,n=512); K=4*eta+g
        print(A,g,w,'A_MF/A^2',a/A**2,'16pi law',16*np.pi*eta*w/(w**2+K**2),'4pi law',4*np.pi*eta*w/(w**2+K**2),'ratio to 16pi law',a/A**2/(16*np.pi*eta*w/(w**2+K**2)))
