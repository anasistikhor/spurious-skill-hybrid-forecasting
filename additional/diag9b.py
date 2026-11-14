import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "../src")
import article2_pipeline as ap, article2_tests as at
df=pd.read_csv("./mc_pair_diagnostics.csv").dropna()
print("pairs with identical forecasts dropped:", 5*84-len(df))
print("Spearman rej_hln vs:", {c:round(stats.spearmanr(df.rej_hln,df[c]).statistic,2) for c in ["kurt","acf_sum20","lrv_ratio20_vs_nw"]})
df["stack"]=df.A.eq("Stacking")|df.B.eq("Stacking")
print(df.groupby("stack")[["rej_hln","acf_sum20","lrv_ratio20_vs_nw"]].median().round(3))
# sensitivity: longer NW lag (20) on the 9 primary family, block 10, both n
def mc(D,cols,n,b,lag,reps=5000):
    rng=np.random.default_rng([2026,n,int(b*10)]); prim=np.array([(a,bb) in ap.PRIMARY_PAIRS for _,a,bb in cols])
    f9=f84=0
    for _ in range(reps):
        idx=at.stationary_bootstrap_index(D.shape[0],n,b,rng); X=D[idx]
        nn=X.shape[0]; Xc=X-X.mean(0); g0=(Xc*Xc).sum(0)/nn; lr=g0.copy()
        for k in range(1,lag+1): lr+=2*(1-k/(lag+1))*(Xc[k:]*Xc[:-k]).sum(0)/nn
        with np.errstate(divide="ignore",invalid="ignore"): t=np.where(lr>0,X.mean(0)/np.sqrt(lr/nn),np.nan)*np.sqrt((nn-1)/nn)
        p=np.nan_to_num(2*stats.t.sf(np.abs(t),df=nn-1),nan=1.0)
        f9+=bool(p[prim].min()*prim.sum()<0.05); f84+=bool(p.min()*len(p)<0.05)
    return f9/reps,f84/reps
for s in [42,101,202,303,404]:
    REF={t:pd.read_csv(f"reference_forecasts/{s}/{t}.csv",parse_dates=["Date"],index_col="Date") for t in ap.TICKERS}
    D,cols=at.loss_differentials(REF)
    print(s,"lag5",mc(D,cols,501,10,5),"lag20",mc(D,cols,501,10,20),flush=True)
