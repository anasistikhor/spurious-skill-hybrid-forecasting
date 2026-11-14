# Reproducing the results

## 1. Environment

The study notebooks were run on Google Colab (Python 3.13). Each notebook writes the exact package versions it used to `05_Outputs/logs/` at run time. The additional analyses in `additional/` were run with:

python 3.13.18, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, xgboost 3.4.1, tensorflow 2.21.0, matplotlib 3.11.2, openpyxl 3.1.5

Install with `pip install -r requirements.txt`.

## 2. Data

Daily adjusted closing prices for JPM, NEE and XOM, 2 January 2015 to 30 December 2024, from Yahoo Finance (yfinance). Splits: training 2015-2020 (1462 days), validation 2021-2022 (503), test 3 January 2023 to 30 December 2024 (501).

The notebooks read a fixed input snapshot (`02_Inputs_v2_snapshot`: data splits, ARFIMA forecasts, combined test forecasts) and check it by SHA-256 before any run.

## 3. Run order

| Step | Notebook | What it produces | Approximate time |
| --- | --- | --- | --- |
| 1 | NB1 Reference seeds | Reference reproduction (seed 42 exact) and five-seed reference forecasts | 20-30 min |
| 2 | NB2 Model faults | F1-F6 on five seeds; materiality outcomes | 30-45 min |
| 3 | NB3 Tests, baselines, Monte Carlo | F7 test families, F8 baselines, Monte Carlo (seed 42 null) | 45-60 min |
| 4 | NB4 Combined and diagnostics | All-fault run, checks C1-C7, detection matrix | 10-15 min |
| 5 | NB5 Figures and tables | Figures 3-8 and the result tables, rebuilt from saved outputs only | 2 min |
| 6 | NB6 Literature audit | Screening, sampling (seed 2026), coding sheets, agreement, Figures 1-2 | minutes |

Each notebook stops if its checks fail (for example, if the reference does not reproduce exactly).

## 4. Seeds

Reference seeds 42, 101, 202, 303, 404. Audit sample seed 2026; author-check draw seed 20261114. Monte Carlo seed 2026.

## 5. Additional analyses (not in the protocol)

Place the five-seed reference forecasts (`<seed>/<TICKER>.csv`, from NB1) in `additional/reference_forecasts/`, then run from `additional/`:

- `python mc_seeds.py` - Monte Carlo null on all five seeds (Table S2a, about 2 minutes)
- `python diag9.py` - per-pair diagnostics of test size (Table S2b)
- `python diag9b.py` - longer Newey-West lag sensitivity

## 6. Audit materials

`audit/coding_rules.md` holds the decision rules; `audit/build_file_s1.py` rebuilds Supplementary File S1 from the coding sheets.
