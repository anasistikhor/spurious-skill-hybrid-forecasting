import sys, pickle, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "../src")
import article2_pipeline as ap, article2_tests as at
raw=pickle.load(open("./mc_seeds_raw.pkl","rb")); REPS=5000
rows=[]
for s in [42,101,202,303,404]:
    REF={t:pd.read_csv(f"reference_forecasts/{s}/{t}.csv",parse_dates=["Date"],index_col="Date") for t in ap.TICKERS}
    D,cols=at.loss_differentials(REF)
    r=raw[(s,501,10)]
    for j,(t,a,b) in enumerate(cols):
        d=D[:,j]
        acf=[np.corrcoef(d[:-k],d[k:])[0,1] for k in range(1,21)]
        lag=ap.nw_lag(501)
        g0=d.var(); lr_true=g0*(1+2*sum(acf[:20])); lr_nw=g0*(1+2*sum((1-k/(lag+1))*acf[k-1] for k in range(1,lag+1)))
        rows.append(dict(seed=s,ticker=t,A=a,B=b,primary=(a,b) in ap.PRIMARY_PAIRS,rej_hln=r["rej_hln"][j]/REPS,rej_plain=r["rej_plain"][j]/REPS,
            kurt=stats.kurtosis(d),acf1=acf[0],acf_sum20=sum(acf),lrv_ratio20_vs_nw=lr_true/lr_nw if lr_nw>0 else np.nan,
            top1pct_share=np.sort(d**2)[-5:].sum()/(d**2).sum()))
df=pd.DataFrame(rows); df.to_csv("./mc_pair_diagnostics.csv",index=False)
print("NW lag:",ap.nw_lag(501))
print(df.groupby("primary")[["rej_hln","kurt","acf1","acf_sum20","lrv_ratio20_vs_nw","top1pct_share"]].median().round(3))
print(df[df.primary].groupby(["A","B"])[["rej_hln","acf_sum20","lrv_ratio20_vs_nw","kurt"]].median().round(3))
print("corr rej_hln with:", {c:round(stats.spearmanr(df.rej_hln,df[c]).statistic,2) for c in ["kurt","acf1","acf_sum20","lrv_ratio20_vs_nw","top1pct_share"]})
print(df.sort_values("rej_hln",ascending=False).head(12)[["seed","ticker","A","B","primary","rej_hln","acf_sum20","lrv_ratio20_vs_nw","kurt"]].round(3).to_string(index=False))
print(df.groupby("seed").rej_hln.agg(["mean","max"]).round(3))
