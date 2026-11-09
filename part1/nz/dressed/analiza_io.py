import numpy as np, struct
# common reader of the kz_sim checkpoint
HDR=struct.Struct('<Q5dQ4Q3d3q2qq4d')  # see the St struct
NB=1024
def read(fn):
    b=open(fn,'rb').read()
    off=0
    def take(fmt):
        nonlocal off
        v=struct.unpack_from('<'+fmt,b,off); off+=struct.calcsize('<'+fmt); return v
    magic,=take('Q'); Delta,A,gamma,omega,theta0=take('5d'); seed,=take('Q'); s=take('4Q'); dev,te,tloc=take('3d')
    burn,per_batch,nbt,ib,nb,cnt,events=take('7q'); cur=take('4d')
    I=np.frombuffer(b,float,NB,off); off+=NB*8; S=np.frombuffer(b,float,NB,off); off+=NB*8
    E=np.frombuffer(b,float,NB,off); off+=NB*8; C2=np.frombuffer(b,float,NB,off)
    return dict(Delta=Delta,A=A,gamma=gamma,omega=omega,theta0=theta0,seed=seed,dev=dev,burn=burn,per_batch=per_batch,ib=ib,nb=nb,events=events,I=I[:nb],S=S[:nb],E=E[:nb],C2=C2[:nb])
