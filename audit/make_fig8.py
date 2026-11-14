import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from statsmodels.stats.proportion import proportion_confint as ci
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e6e5e0", "#f0efec"; S1="#2a78d6"
matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.edgecolor":INK2,"axes.labelcolor":INK,
 "xtick.color":INK2,"ytick.color":INK2,"text.color":INK,"axes.spines.top":False,"axes.spines.right":False,
 "axes.spines.left":False,"axes.linewidth":0.6,"savefig.dpi":300,"savefig.bbox":"tight","pdf.fonttype":42})
G=[("Evaluation",[("Price R² not the headline",30),("Significance test",13),("Relative or returns metric",8),("Persistence benchmark",3),("Robust variance in the test",1),("Multiple-testing control",0)]),
   ("Pipeline integrity",[("Chronological split stated",23),("Leakage control stated",13),("Same horizon stated",10),("Unit consistency stated",8)]),
   ("Reproducibility",[("Data source and dates given",21),("Code link given",0)])]
n=40; rows=[]; y=0; ylab=[]; gpos=[]
for g,items in G:
    gpos.append((g,y)); y+=0.9
    for lab,k in items:
        lo,hi=ci(k,n,method="wilson"); rows.append((y,k/n*100,lo*100,hi*100,k)); ylab.append((y,lab)); y+=1
    y+=0.5
fig,ax=plt.subplots(figsize=(6.3,4.6))
for yy,p,lo,hi,k in rows:
    ax.plot([lo,hi],[yy,yy],color=S1,lw=1.6,solid_capstyle="round",zorder=2)
    ax.scatter(p,yy,s=34,color=S1,edgecolor="white",linewidth=1.2,zorder=3,clip_on=False)
    ax.text(hi+1.8,yy,f"{k} of {n}",va="center",fontsize=7,color=INK2)
for yy,lab in ylab: ax.text(-2,yy,lab,ha="right",va="center",fontsize=7.5,color=INK)
for g,yy in gpos: ax.text(-2,yy,g,ha="right",va="center",fontsize=8,fontweight="bold",color=INK)
ax.set_xlim(-1.5,100); ax.set_ylim(y-0.3,-0.3); ax.set_yticks([])
ax.set_xticks([0,25,50,75,100]); ax.set_xticklabels(["0%","25%","50%","75%","100%"])
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel("Share of the 40 audited studies reporting the safeguard (dot) with 95% Wilson interval (line)",color=INK2,fontsize=7.5)
fig.savefig("Figure_8_safeguards.png",facecolor="white"); fig.savefig("Figure_8_safeguards.pdf",facecolor="white")
print("ok")
