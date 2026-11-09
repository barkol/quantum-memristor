# generuje jobs.txt; wznowienie: uruchomic run_all.sh ponownie
cost={0.1:8e9,0.2:3e9,0.3:2e9}
L=[];s=1000
for g in (0,0.01,0.1):
  for D in (0.025,0.05,0.1,0.15,0.25):
    L.append((D,0.0,g,2.0,2e9,None))
    for A in (0.1,0.2,0.3):
      for w in (0.5,2,6,20):
        L.append((D,A,g,float(w),cost[A],None))
# wrazliwosc na theta0=0
for w in (0.5,2.0): L.append((0.1,0.2,0.1,w,3e9,0.0))
with open('jobs.txt','w') as f:
  for i,(D,A,g,w,c,t0) in enumerate(L):
    tag=f"D{D}_A{A}_g{g}_w{w}"+("" if t0 is None else f"_t{t0}")
    f.write(f"res/{tag}.bin {D} {A} {g} {w} {1000+i} {c:.0f}"+("" if t0 is None else f" {t0}")+"\n")
print(len(L))
