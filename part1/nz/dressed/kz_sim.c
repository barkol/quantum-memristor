// Symulacja zdarzeniowa rejestru: Poisson(Gamma), skok +-Delta z p+-=(1+-cos phi)/2,
// miedzy zdarzeniami scisla relaksacja theta-theta0 -> (theta-theta0) exp(-gamma dt).
// Pole liczone regula M punktow na okres (scisla dla harmonicznych < M), z batch means.
// Checkpoint: atomowy zapis stanu co CK_SEC s i na koncu; wznowienie deterministyczne.
// uzycie: kz_sim out.bin Delta A gamma omega seed cost_units [theta0]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#define M 16
#define NB 1024
#define CK_SEC 180
typedef struct {
  uint64_t magic; double Delta,A,gamma,omega,theta0; uint64_t seed;
  uint64_t s[4]; double dev,te,tloc; // dev=theta-theta0, te=czas nast. zdarzenia wzgledem poczatku okresu
  int64_t burn, per_batch, nbatch_target; int64_t ib; // ib=biezacy okres (od poczatku, ujemny w burnie)
  int64_t nb; int64_t cnt_in_batch; int64_t events;
  double curI,curS,curE,curC2; // biezace sumy
  double I[NB],S[NB],E[NB],C2[NB];
} St;
static inline uint64_t rotl(uint64_t x,int k){return (x<<k)|(x>>(64-k));}
static inline uint64_t nxt(uint64_t*s){uint64_t r=rotl(s[0]+s[3],23)+s[0],t=s[1]<<17;s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rotl(s[3],45);return r;}
static inline double u01(uint64_t*s){return ((nxt(s)>>11)+0.5)*(1.0/9007199254740992.0);}
static uint64_t sm(uint64_t*x){uint64_t z=(*x+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
static void save(const char*f,St*st){char t[1024];snprintf(t,1024,"%s.tmp",f);FILE*o=fopen(t,"wb");if(!o){perror("save");exit(2);}fwrite(st,sizeof(St),1,o);fflush(o);fclose(o);rename(t,f);}
static volatile sig_atomic_t stopflag=0; static void onsig(int s){(void)s;stopflag=1;}
int main(int c,char**v){
  if(c<8){fprintf(stderr,"uzycie\n");return 1;}
  const char*f=v[1]; St*st=calloc(1,sizeof(St));
  double Delta=atof(v[2]),A=atof(v[3]),gamma=atof(v[4]),omega=atof(v[5]); uint64_t seed=strtoull(v[6],0,10); double cost=atof(v[7]);
  double theta0=c>8?atof(v[8]):M_PI/8; double eta=0.5,Gam=eta/Delta;
  double T=2*M_PI/omega,dt_s=T/M;
  FILE*in=fopen(f,"rb"); int resumed=0;
  if(in){ if(fread(st,sizeof(St),1,in)==1&&st->magic==0x4b5a5349u&&st->Delta==Delta&&st->A==A&&st->gamma==gamma&&st->omega==omega&&st->theta0==theta0&&st->seed==seed) resumed=1; else {fprintf(stderr,"zly checkpoint %s\n",f);return 3;} fclose(in);}
  if(!resumed){
    st->magic=0x4b5a5349u;st->Delta=Delta;st->A=A;st->gamma=gamma;st->omega=omega;st->theta0=theta0;st->seed=seed;
    uint64_t x=seed;for(int i=0;i<4;i++)st->s[i]=sm(&x);
    st->dev=0;st->tloc=0;st->te=-log(u01(st->s))/Gam;
    double tb=100+ (gamma>0?30/gamma:0); st->burn=(int64_t)ceil(tb/T);
    double rate=Gam+0.5*M/T; int64_t per=(int64_t)(cost/rate/T); if(per<NB)per=NB;
    st->per_batch=per/NB; st->nbatch_target=NB; st->ib=-st->burn;
  }
  double wI[M],chi[M]; for(int k=0;k<M;k++){double t=k*dt_s,x=2*A*sin(omega*t);chi[k]=x;wI[k]=2*A*omega*cos(x)*cos(omega*t)*dt_s;}
  signal(SIGTERM,onsig);signal(SIGINT,onsig);
  time_t last=time(0); double phi,cp;
  double dev=st->dev,te=st->te; uint64_t*s=st->s; int64_t ev=st->events;
  double twoA=2*A, gg=gamma, Dl=Delta;
  while(st->nb<NB){
    double sI=0,sS=0,sE=0,sC=0, tnow=0;
    for(int k=0;k<M;k++){
      double tk=k*dt_s;
      while(te<tk){
        if(gg>0)dev*=exp(-gg*(te-tnow)); tnow=te;
        phi=4*(dev+theta0)-twoA*sin(omega*te); cp=cos(phi);
        if(u01(s)<0.5*(1+cp))dev+=Dl; else dev-=Dl;
        te+=-log(u01(s))/Gam; ev++;
      }
      if(gg>0&&tk>tnow){dev*=exp(-gg*(tk-tnow));} if(tk>tnow)tnow=tk;
      phi=4*(dev+theta0)-chi[k]; cp=cos(phi); double sp=sin(phi);
      sI+=cp*wI[k]; sS+=sp; double e=4*dev-chi[k]; sE+=e*(-cp); sC+=cp*cp; // eps sin eps: sin eps=-cos phi
    }
    // zdarzenia az do konca okresu
    while(te<T){
      if(gg>0)dev*=exp(-gg*(te-tnow)); tnow=te;
      phi=4*(dev+theta0)-twoA*sin(omega*te); cp=cos(phi);
      if(u01(s)<0.5*(1+cp))dev+=Dl; else dev-=Dl;
      te+=-log(u01(s))/Gam; ev++;
    }
    if(gg>0)dev*=exp(-gg*(T-tnow));
    te-=T;
    if(st->ib>=0){ st->curI+=sI; st->curS+=sS/M; st->curE+=sE/M; st->curC2+=sC/M; st->cnt_in_batch++;
      if(st->cnt_in_batch==st->per_batch){int n=st->nb; st->I[n]=st->curI/st->per_batch;st->S[n]=st->curS/st->per_batch;st->E[n]=st->curE/st->per_batch;st->C2[n]=st->curC2/st->per_batch;st->curI=st->curS=st->curE=st->curC2=0;st->cnt_in_batch=0;st->nb++;}
    }
    st->ib++;
    if((st->ib&1023)==0||stopflag){ time_t now=time(0); if(now-last>=CK_SEC||stopflag){st->dev=dev;st->te=te;st->events=ev;save(f,st);last=now; if(stopflag){return 10;}} }
  }
  st->dev=dev;st->te=te;st->events=ev;save(f,st);
  return 0;
}
