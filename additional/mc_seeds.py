import sys, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, "../src")
import article2_pipeline as ap, article2_tests as at
SEEDS=[42,101,202,303,404]; SET=[(501,10),(441,10),(501,5),(501,20),(441,5),(441,20)]; REPS=5000
REFC={s:{t:pd.read_csv(f"reference_forecasts/{s}/{t}.csv",parse_dates=["Date"],index_col="Date") for t in ap.TICKERS} for s in SEEDS}
raw={}
for s in SEEDS:
    D,cols=at.loss_differentials(REFC[s])
    for n,b in SET:
        t0=time.time(); raw[(s,n,b)]=at.run_monte_carlo(D,cols,n,b,REPS)
        r=raw[(s,n,b)]; print(s,n,b,round(r["fam_plain"]/REPS,3),round(r["fam_holm84"]/REPS,3),round(r["fam_holm9"]/REPS,3),f"{time.time()-t0:.0f}s",flush=True)
    raw[("cols",s)]=cols
pickle.dump(raw,open("./mc_seeds_raw.pkl","wb"))
