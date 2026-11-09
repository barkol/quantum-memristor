// Event-driven exact simulation of the signed / unipolar register. No time step in the dynamics:
// Poisson arrivals, exact exponential relaxation between events, outcome from exact phi(t_k).
// Each path is run twice with common random numbers: chi(t) (index 0) and -chi(t) (index 1).
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static inline uint64_t rotl(uint64_t x,int k){return (x<<k)|(x>>(64-k));}
typedef struct{uint64_t s[4];} rng_t;
static uint64_t sm(uint64_t*x){uint64_t z=(*x+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
static void seed(rng_t*r,uint64_t x){for(int i=0;i<4;i++)r->s[i]=sm(&x);}
static inline uint64_t nx(rng_t*r){uint64_t*s=r->s;uint64_t res=rotl(s[0]+s[3],23)+s[0];uint64_t t=s[1]<<17;s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rotl(s[3],45);return res;}
static inline double un(rng_t*r){return ((nx(r)>>11)+0.5)*(1.0/9007199254740992.0);}
#define NC 8
// out: nsamp*2*NC doubles (sums over paths): cos,sin,cos2,sin2,th,th^2,th*cos,th*sin
// hist: 2*NB counts of branch k at last sample; branch k=round((phi-phi_ref)/2pi), offset NB/2
#define NB 41
long long simulate(long long N,double Delta,double Gamma,double gamma_,double theta0,double A,double omega,
   int unipolar,const double*theta_init,long long ninit,const double*tsamp,long long nsamp,
   double phi_ref,uint64_t sd,double*out,double*hist){
  rng_t r;seed(&r,sd);
  long long nev=0;
  memset(out,0,sizeof(double)*nsamp*2*NC);memset(hist,0,sizeof(double)*2*NB);
  for(long long p=0;p<N;p++){
    double th[2];th[0]=th[1]=theta_init[p%ninit];
    double t=0.0;long long j=0;
    double tn=-log(1.0-un(&r))/Gamma;
    double Tend=tsamp[nsamp-1];
    while(1){
      while(j<nsamp&&tsamp[j]<=tn){
        double dt=tsamp[j]-t,e=exp(-gamma_*dt);
        for(int s=0;s<2;s++){
          double chi=(s==0?1:-1)*(A/2)*sin(omega*tsamp[j]);
          double x=theta0+(th[s]-theta0)*e;
          double phi=4*x-2*chi;
          double c=cos(phi),si=sin(phi);
          double*o=out+(j*2+s)*NC;
          o[0]+=c;o[1]+=si;o[2]+=2*c*c-1;o[3]+=2*si*c;o[4]+=x;o[5]+=x*x;o[6]+=x*c;o[7]+=x*si;
          if(j==nsamp-1){long k=lround((phi-phi_ref)/(2*M_PI))+NB/2;if(k<0)k=0;if(k>=NB)k=NB-1;hist[s*NB+k]+=1;}
        }
        j++;
      }
      if(j>=nsamp||tn>Tend)break;
      double dt=tn-t,e=exp(-gamma_*dt);
      double chi0=(A/2)*sin(omega*tn);
      double u=un(&r);
      for(int s=0;s<2;s++){
        double chi=(s==0)?chi0:-chi0;
        double x=theta0+(th[s]-theta0)*e;
        double pH=0.5*(1+cos(4*x-2*chi));
        if(u<pH)x+=Delta; else if(!unipolar)x-=Delta;
        th[s]=x;
      }
      t=tn;nev++;
      tn=t-log(1.0-un(&r))/Gamma;
    }
  }
  return nev;
}
// Kick response: chi(t) identical for both copies, copy 1 gets +dk at t=tk; common random numbers.
long long simulate_kick(long long N,double Delta,double Gamma,double gamma_,double theta0,double A,double omega,
   int unipolar,double theta_init,double tk,double dk,const double*tsamp,long long nsamp,uint64_t sd,double*out){
  rng_t r;seed(&r,sd);long long nev=0;
  memset(out,0,sizeof(double)*nsamp*2*NC);
  for(long long p=0;p<N;p++){
    double th[2];th[0]=th[1]=theta_init;double t=0.0;long long j=0;int kicked=0;
    double tn=-log(1.0-un(&r))/Gamma;
    while(1){
      double nextstop=tn; int stop=0; // 0 event,1 sample,2 kick
      if(j<nsamp&&tsamp[j]<=nextstop){nextstop=tsamp[j];stop=1;}
      if(!kicked&&tk<=nextstop){nextstop=tk;stop=2;}
      double dt=nextstop-t,e=exp(-gamma_*dt);
      th[0]=theta0+(th[0]-theta0)*e;th[1]=theta0+(th[1]-theta0)*e;t=nextstop;
      if(stop==1){
        double chi=(A/2)*sin(omega*t);
        for(int s=0;s<2;s++){double x=th[s],phi=4*x-2*chi,c=cos(phi),si=sin(phi);double*o=out+(j*2+s)*NC;
          o[0]+=c;o[1]+=si;o[2]+=2*c*c-1;o[3]+=2*si*c;o[4]+=x;o[5]+=x*x;o[6]+=x*c;o[7]+=x*si;}
        j++;if(j>=nsamp)break;
      }else if(stop==2){th[1]+=dk;kicked=1;}
      else{
        double chi=(A/2)*sin(omega*t),u=un(&r);
        for(int s=0;s<2;s++){double pH=0.5*(1+cos(4*th[s]-2*chi));if(u<pH)th[s]+=Delta;else if(!unipolar)th[s]-=Delta;}
        nev++;tn=t-log(1.0-un(&r))/Gamma;
      }
    }
  }
  return nev;
}
