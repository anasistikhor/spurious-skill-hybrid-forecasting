# Spurious Skill in Hybrid Stock Forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24260524.svg)](https://doi.org/10.5281/zenodo.24260524)

Code and audit materials for the article "Spurious Skill in Hybrid Stock Forecasting: A Fault-Injection Study with a Literature Audit" (Istikhor, Azman and Sangodiah; School of Computer Science, Taylor's University, Malaysia).

## Contents

| Folder | What it holds |
| --- | --- |
| `src/` | Reference pipeline (ARFIMA, LSTM, XGBoost; Serial, Parallel and Stacking hybrids), the eight fault switches (F1-F8), Diebold-Mariano test families and Monte Carlo, literature baselines, and diagnostic checks C1-C7 |
| `notebooks/` | NB1 reference seeds; NB2 model faults; NB3 tests, baselines and Monte Carlo; NB4 combined run and diagnostics; NB5 figures and tables; NB6 literature audit |
| `audit/` | Coding rules, the script that builds Supplementary File S1, and the safeguard-chart script |
| `additional/` | Additional analyses beyond the study protocol: Monte Carlo on all five seeds and test-size diagnostics |

## Versions

- **v2** is the verified reference pipeline used for all reference results.
- **v1** is an earlier, unpublished version of the same pipeline. Its documented errors define faults F1-F8, and the notebooks compare its headline claims with the combined fault run (Supplementary Table S4).

## Reproducing

- Data: daily adjusted closing prices for JPM, NEE and XOM, January 2015 to December 2024, from Yahoo Finance (yfinance).
- Seeds: reference seeds 42, 101, 202, 303, 404; audit sample seed 2026; Monte Carlo seed 2026.
- The notebooks read a fixed input snapshot and stop if the reference does not reproduce exactly.
- Step-by-step instructions, run order and package versions: see `REPRODUCE.md`.

## Licence

MIT. See `LICENSE`.

## Funding

This research was funded by Taylor's University through the Taylor's Research Excellence Scholarship (TRES).
