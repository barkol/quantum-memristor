import os
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
import sys,json;import os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from retrieval_core import *
TH={}
for l in open('../results/trained_schedule.jsonl'):
    r=json.loads(l);TH[r['key'].split('|')[1]]=np.array(r['th'])
U=np.linspace(0,1,6)
Zc={}
ci=lambda x:1.96*np.std(x,ddof=1)/np.sqrt(len(x))
def job(a):
    key,thk,n,L,net,f,hv,tr,base=a
    th=TH[thk];s_k=np.clip(th[:6],0,1);a_k=np.clip(th[6:12],0,None)
    pats,mu,xi,xit,k=make_probes(base,n,L,net,f,tr)
    J=hebbian(pats);h=hv*xit.astype(float)
    if n not in Zc: Zc[n]=spins_table(n).astype(float)
    Z=Zc[n];EJ=energy_J_table(J,Z);Ed=EJ[None,:]-h@Z.T;Emin=Ed.min(1)
    P=anneal(J,h,Z,EJ,T=12.0,K=120,sched=lambda u:np.interp(u,U,s_k),omega_sched=lambda u:np.interp(u,U,a_k))
    P0=anneal(J,h,Z,EJ)
    rng=np.random.default_rng([base,n,L,net,int(f*1e6)])
    out={}
    def rec(nm,s):
        d=(xi*s).sum(1)/n;out['m_'+nm]=float(np.abs(d).mean());out['ms_'+nm]=float(d.mean())
        out['pgs_'+nm]=float((energies(J,h,s.astype(float))<=Emin+1e-9).mean())
    rec('opt',Z[P.argmax(1)]);rec('qa',Z[P0.argmax(1)]);rec('gb',greedy(J,h,xit));rec('gu',greedy(J,0*h,xit));rec('sa',sa(J,h,xit,rng,start='input'));rec('gs',Z[Ed.argmin(1)])
    rec('gb3',greedy(J,.3*xit.astype(float),xit))
    d=xit@pats.T;b=np.argmax(np.abs(d),axis=1);ns=pats[b]*np.sign(d[np.arange(len(b)),b])[:,None];rec('near',ns)
    mall=np.abs(xi@Z.T)/n;out['samp_opt']=float((P*mall).sum(1).mean());out['Eres_opt']=float(((P*Ed).sum(1)-Emin).mean())
    # finite-shot mode readout with 100 and 1000 shots
    for S in(100,1000):
        tot=0
        for t in range(tr):
            c=rng.multinomial(S,P[t]/P[t].sum());tot+=abs((xi[t]*Z[c.argmax()]).sum())/n
        out[f'mode{S}_opt']=tot/tr
    return key,out
if __name__=="__main__":
    todo=[]
    for thk,f in(('0.1875',.1875),('0.125',.125)):
        for net in range(240): todo.append((f'A|{thk}|{f}|8|3|{net}',thk,8,3,net,f,.6,100,5400))
    for f in(.125,.25): 
        for net in range(120): todo.append((f'B|0.1875|{f}|8|3|{net}',"0.1875",8,3,net,f,.6,100,5500))
    for n,L,N,tr in((12,3,20,30),(12,5,20,30),(14,3,12,12),(14,5,12,12)):
        for net in range(N): todo.append((f'C|0.1875|.1875|{n}|{L}|{net}',"0.1875",n,L,net,.1875,.6,tr,5600))
    R=[]
    with Pool(4) as p:
        for k,r in p.imap_unordered(job,todo): r['key']=k;R.append(r)
    json.dump(R,open('../results/trained_schedule_test.json','w'))
    import collections
    G=collections.defaultdict(list)
    for r in R: G[tuple(r['key'].split('|')[:5])].append(r)
    for g,X in sorted(G.items()):
        print(g,len(X),' '.join(x+'=%.3f'%np.mean([r['m_'+x] for r in X]) for x in('opt','qa','gb','gb3','sa','gs','gu','near')),'opt-gb %+.4f±%.4f opt-gb3 %+.4f±%.4f signed opt-gb %+.4f samp %.3f mode100 %.3f mode1000 %.3f pGS opt %.3f gb %.3f Eres %.3f'%(np.mean([r['m_opt']-r['m_gb'] for r in X]),ci([r['m_opt']-r['m_gb'] for r in X]),np.mean([r['m_opt']-r['m_gb3'] for r in X]),ci([r['m_opt']-r['m_gb3'] for r in X]),np.mean([r['ms_opt']-r['ms_gb'] for r in X]),np.mean([r['samp_opt'] for r in X]),np.mean([r['mode100_opt'] for r in X]),np.mean([r['mode1000_opt'] for r in X]),np.mean([r['pgs_opt'] for r in X]),np.mean([r['pgs_gb'] for r in X]),np.mean([r['Eres_opt'] for r in X])))
