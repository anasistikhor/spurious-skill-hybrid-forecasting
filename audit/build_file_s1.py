import pandas as pd, json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import os
# Input folder (audit outputs) and output file; override with environment variables if needed
A=os.environ.get("AUDIT_DIR","audit_out/")
out=os.environ.get("S2_OUT","Supplementary_File_S1_Literature_Audit.xlsx")
F=["Persistence benchmark","Relative or returns metric","Price R2 as headline","Significance test","Robust variance","Multiple-testing control","Chronological split","Leakage control stated","Same horizon for all components","Unit consistency stated","Code public","Data source and period stated"]

search=pd.DataFrame([
 {"Item":"Scopus query (12 Nov 2026, via Taylor's University Library)","Value":'TITLE-ABS-KEY((hybrid OR ensemble OR combin*) AND (ARIMA OR ARFIMA OR GARCH OR "exponential smoothing") AND (LSTM OR GRU OR XGBoost OR "random forest" OR "neural network" OR "deep learning" OR "machine learning") AND ("stock price" OR "stock market" OR "stock index" OR "share price" OR equit*) AND (forecast* OR predict*)) AND PUBYEAR > 2021 AND DOCTYPE(ar) AND SRCTYPE(j) AND LANGUAGE(english)'},
 {"Item":"Scopus records","Value":"209"},
 {"Item":"Lens.org query (12 Nov 2026; Scholarly Works; filters Journal Article, 2022-2026)","Value":'(title:(hybrid OR ensemble OR combin*) OR abstract:(hybrid OR ensemble OR combin*) OR keyword:(hybrid OR ensemble OR combin*)) AND (title:(ARIMA OR ARFIMA OR GARCH OR "exponential smoothing") OR abstract:(ARIMA OR ARFIMA OR GARCH OR "exponential smoothing") OR keyword:(ARIMA OR ARFIMA OR GARCH OR "exponential smoothing")) AND (title:(LSTM OR GRU OR XGBoost OR "random forest" OR "neural network" OR "deep learning" OR "machine learning") OR abstract:(LSTM OR GRU OR XGBoost OR "random forest" OR "neural network" OR "deep learning" OR "machine learning") OR keyword:(LSTM OR GRU OR XGBoost OR "random forest" OR "neural network" OR "deep learning" OR "machine learning")) AND (title:("stock price" OR "stock market" OR "stock index" OR "share price" OR equit*) OR abstract:("stock price" OR "stock market" OR "stock index" OR "share price" OR equit*) OR keyword:("stock price" OR "stock market" OR "stock index" OR "share price" OR equit*)) AND (title:(forecast* OR predict*) OR abstract:(forecast* OR predict*) OR keyword:(forecast* OR predict*))'},
 {"Item":"Lens.org records","Value":"383"},
 {"Item":"Total records","Value":"592"},
 {"Item":"Duplicates removed (normalised DOI, then normalised title + year)","Value":"139"},
 {"Item":"Records screened (IDs A0001-A0453, year-then-title order)","Value":"453"},
 {"Item":"Excluded at title and abstract","Value":"272"},
 {"Item":"Full texts assessed","Value":"181"},
 {"Item":"Excluded at full text","Value":"114"},
 {"Item":"Eligible","Value":"67"},
 {"Item":"Random sample coded (seed 2026)","Value":"40"},
 {"Item":"Independent blind re-coding (same generator, drawn after the 40)","Value":"10"},
 {"Item":"Author check (seed 20261114, drawn from the 40)","Value":"5"},
 {"Item":"Web of Science","Value":"Not available through the institution; replaced by Lens.org as provided in the protocol"},
 {"Item":"Google Scholar recall check","Value":"Planned, not run (limitation)"},
])
scr=pd.read_excel(A+"screening_sheet.xlsx").drop(columns=["Abstract"])
ft=pd.read_excel(A+"stage4b/fulltext_check.xlsx")
s40=pd.read_csv(A+"stage4b/audit_sample_40.csv")[["ID","Authors","Year","Title","Journal","DOI"]]
rec=set(pd.read_csv(A+"stage4b/recode_ids.csv").ID)
ac={"A0002","A0051","A0237","A0305","A0344"}
s40["Re-coded (blind)"]=s40.ID.map(lambda i:"Yes" if i in rec else "")
s40["Author check"]=s40.ID.map(lambda i:"Yes" if i in ac else "")
first=pd.read_excel(A+"stage5/coding_sheet_first.xlsx")
recode=pd.read_excel(A+"stage5/coding_sheet_recode.xlsx")
author=pd.read_excel(A+"stage5/coding_sheet_author_check.xlsx")
for d in (first,recode,author):
    if "Coder" in d.columns: d.drop(columns=["Coder"],inplace=True)
kap=pd.read_csv(A+"stage5/audit_kappa.csv")
counts=pd.read_csv(A+"stage5/audit_field_counts.csv")
rules=open(A+"coding_instructions.md").read().split("\n")
code=open("/tmp/nb6_cell4.py").read().split("\n")

ev={
"A0305":["Benchmarks ARIMA, GARCH(1,1), LSTM, hybrid; no random walk or naive (Table 1)","Target closing price; RMSE, MAE, MAPE; log returns only as GARCH input","R2 not reported","No accuracy test; only ADF and Ljung-Box","-","-","Chronological: first 900 days train, last 127 test (Sec. 3.1)","All features Min-Max normalised, fit sample not stated; GARCH returns from training prices; ARIMA sliding-window next-close prediction","ARIMA one-step sliding window; LSTM horizon not stated separately","Errors computed after normalising test data; no back-transformation stated; Table 1 RMSE 4.09-9.47 with MAPE 0.11-0.24%","No code link","Yahoo Finance, 4 Jan 2021 to 25 Apr 2025","Abstract gives no number"],
"A0344":["Random walk only in literature review; no naive results (Tables 1-4)","Open/high/low/close prices; MAD, MAPE, RMSE","R2 not reported","None; lowest error picks model","-","-","'Test sets' mentioned; split method and size not stated","No scaling or input construction described","General h-step formula; horizon used not stated","ANN residual forecast added to ARIMA forecast; scaling not discussed","No code link","Yahoo Finance, '2000 to 2022' (years only); data on request","Abstract gives no number; ARIMA best"],
"A0051":["ANN, LSTM, GARCH-variance hybrids, random-number control; no naive","Target BSE Sensex opening value (level); RMSE, MAPE","R2 not reported","None; LM test for ARCH effects only","-","-","85% train / 15% test; order not stated","GARCH fit sample not stated; scaling not mentioned; inputs same-day Asian opens and previous S&P close","GARCH is an input; same-day opening forecast; horizon not stated as such","Scaling and back-transformation not mentioned","No code link","1 Jan 2007 to 31 Dec 2020; source not named; data on request","Abstract gives no number"],
"A0237":["LSTM, CNN, GRU, GARCH-LSTM and sentiment variants; no naive (Tables VI-VIII)","CSI 300 price, normalised; MAE, MSE, RMSE, MAPE","R2 not reported","No formal test; abstract says 'significantly'","-","-","Prices split 80:20 (1560/391 days), order not stated; extra test Aug 2015-Dec 2018 train, Jan 2019 test","Normalised after split, scaler fit not stated; 22-day input window; GARCH fit sample not stated","LSTM predicts each point from previous 22 days; GARCH and sentiment are inputs","Errors on normalised price; no back-transformation stated","No code link","Yingwei Financial website, 6 Aug 2015 to 28 Sep 2023","Abstract gives no number"],
"A0002":["ARIMA(0,1,0), Prophet, CNN, CART, RF and chaos hybrids; random walk or naive not named","Prices and rates; MSE, MAPE, Dstat, Theil's U in U1 form (Eq. 8)","R2 not mentioned","Diebold-Mariano (R dm.test), 9 datasets x 8 comparisons = 72 (Table 16)","No HAC or HLN mentioned; |statistic| > 1.96","No correction across 72 tests","80/20; Table 5 lists train_test_split; Section 3 indexes sets by time; chronology not stated outright","MinMax scaling, fit sample not stated; polynomial regression also fitted to test-set errors (step 7)","Horizon not stated for CNN or polynomial regression","CNN forecast and error forecast added; back-transformation not mentioned","No code link","Federal Reserve (FX), Investing.com (indices, commodities); exact dates per series (Table 3)","Abstract gives no number"],
}
acrows=[]
for i,v in ev.items():
    r=author[author.ID==i].iloc[0]
    for k,f in enumerate(F+["Headline improvement"]):
        o=first[first.ID==i].iloc[0][f]
        acrows.append({"ID":i,"Field":f,"Evidence summary given to the author (with section)":v[k],"Author's code":r[f],"Original code":o,"Agree":"Yes" if str(r[f]).strip()==str(o).strip() else "No"})
acdf=pd.DataFrame(acrows)

dev=[
("11 Nov 2026","7","Audit random seed fixed at 2026","Fixed before any search"),
("12 Nov 2026","7 (before any search)","Database syntax; duplicates by normalised DOI then title + year; IDs in year-then-title order; eligible set ordered by ID before the draw; re-code 10 drawn after the 40 by the same generator; safeguard score defined","Mechanics not fixed in the protocol text; fixed before searching"),
("12 Nov 2026","7 (databases)","Web of Science not available through the institution; Lens.org used as the second database, as the protocol provides; Google Scholar as recall check only","Access, not preference"),
("12 Nov 2026","7 (search done)","Scopus 209, Lens.org 383 records; Google Scholar recall check not run","Time; the check adds no records; reported as a limitation"),
("14 Nov 2026","7 (eligibility correction and re-draw)","A0031, A0294, A0354, A0431 excluded after full-text re-check (67 eligible, not 71); draw of 40 and re-code 10 repeated with seed 2026 before any result","Eligibility errors found during coding; corrected before results with the same seed and procedure"),
("14 Nov 2026","7 (re-coding interval)","7-day interval not applied; second coding independent and blind to the first codes","Interval guards against recall (Polit, 2014); recall impossible without access to the first codes (Hayes and Krippendorff, 2007)"),
("14 Nov 2026","7 (added author check)","First author coded 5 further papers (seed 20261114) blind to the original codes; 86.7% agreement, pooled kappa 0.79; original codes unchanged","Tests whether the near-perfect re-coding agreement holds for an independent human coder"),
]
devdf=pd.DataFrame(dev,columns=["Date","Protocol section","Change","Reason"])

readme=pd.DataFrame({"Sheet":["S1a_Search","S1b_Screening","S1c_FullText","S1d_Sample","S1e_Coding_Rules","S1f_Coding_First","S1g_Coding_Recode","S1h_Agreement","S1i_Author_Check","S1j_Sampling_Code","S1k_Audit_Deviations"],
"Contents":["Exact queries, dates and record counts at each stage","All 453 screened records with decision, reason and note (abstracts omitted for copyright)","All 181 full-text assessments with eligibility, reason and note","The 40 sampled studies, with the 10 re-coded and the 5 author-checked flagged","Written decision rules for the 13 fields","First coding of the 40 studies, with evidence notes and headline source","Blind independent re-coding of 10 studies","Field counts (n = 40), percent agreement and Cohen's kappa per field, author-check agreement","Per-decision evidence summaries used in the author check, the author's code and the original code","Code that draws the sample of 40 (seed 2026) and the re-code 10","Audit-related protocol deviations (full log in Table S1)"]})

sheets=[("README",readme),("S1a_Search",search),("S1b_Screening",scr),("S1c_FullText",ft),("S1d_Sample",s40),("S1e_Coding_Rules",pd.DataFrame({"Rule text":rules})),("S1f_Coding_First",first),("S1g_Coding_Recode",recode),("S1h_Agreement",None),("S1i_Author_Check",acdf),("S1j_Sampling_Code",pd.DataFrame({"Python (NB6 stage 4)":code})),("S1k_Audit_Deviations",devdf)]
with pd.ExcelWriter(out,engine="openpyxl") as w:
    for n,d in sheets:
        if n=="S1h_Agreement":
            counts.to_excel(w,sheet_name=n,index=False,startrow=1)
            kap.to_excel(w,sheet_name=n,index=False,startrow=1,startcol=6)
            pd.DataFrame([{"Comparison":"Author check vs first coding","Papers":5,"Decisions":60,"Agreement":"86.7%","Pooled kappa":0.794,"Factual fields (1-7, 11, 12)":"44/45","Judgement fields (8-10)":"8/15"}]).to_excel(w,sheet_name=n,index=False,startrow=1,startcol=10)
            ws=w.sheets[n]; ws["A1"]="Field counts (n = 40)"; ws["G1"]="Re-coding agreement (n = 10)"; ws["K1"]="Author check"
        else:
            d.to_excel(w,sheet_name=n,index=False)
wb=__import__("openpyxl").load_workbook(out)
for ws in wb.worksheets:
    for c in ws[1 if ws.title!="S1h_Agreement" else 2]:
        c.font=Font(bold=True); c.fill=PatternFill("solid",fgColor="F0EFEC")
    for col in ws.columns:
        L=get_column_letter(col[0].column)
        m=max((len(str(c.value)) if c.value is not None else 0) for c in col[:200])
        ws.column_dimensions[L].width=min(max(10,m+2),70)
        for c in col[1:]:
            c.alignment=Alignment(wrap_text=True,vertical="top")
    ws.freeze_panes="A2" if ws.title!="S1h_Agreement" else "A3"
wb.save(out)
print("ok",[ (n, None if d is None else d.shape) for n,d in sheets])
