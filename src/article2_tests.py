"""
article2_tests.py
Part of the code for the fault-injection study (see README).

F7 (test misspecification) and the Monte Carlo test-size study (RQ4),
as frozen in the study protocol, Sections 3 and 5 (frozen 11 Nov 2026).

F7, exactly as Section 3: plain-variance DM (no Newey-West, no HLN), all pairwise comparisons of
the eight Table 4 models (28 pairs x 3 stocks = 84, as in v1), no multiplicity correction, the
winner named by MAE but tested on squared loss.

Monte Carlo, exactly as Section 5: the reference test-period loss differentials of the 84 pairs,
each centred at zero (so no model is truly better), resampled jointly by whole days with the
stationary block bootstrap (Politis and Romano 1994). Procedures: (a) plain-variance DM;
(b) Newey-West DM with HLN; (c) family of 84 plain tests, no correction; (d) family of 84
Newey-West HLN tests with Holm.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import article2_pipeline as ap

MODELS8 = ap.TABLE4_MODELS                     # the eight models tested in v1
PAIRS28 = list(combinations(MODELS8, 2))       # 28 pairs per stock, 84 in total
MC_SEED = 2026                                 # fixed before any simulation (deviation log, 11 Nov)


# ---------------------------------------------------------------- F7
def f7_family(combined):
    """84 tests as in v1, plus the same 84 pairs tested the reference way (NW + HLN + Holm) for comparison."""
    rows = []
    for t, df in combined.items():
        y = df["y_true"].values
        for a, b in PAIRS28:
            ea, eb = y - df[a].values, y - df[b].values
            plain = ap.dm_test(ea, eb, loss="squared", lag=0)       # plain variance, normal p-value
            ref = ap.dm_test(ea, eb, loss="squared")                # Newey-West, HLN
            mae_a, mae_b = np.abs(ea).mean(), np.abs(eb).mean()
            winner_mae = a if mae_a < mae_b else b
            winner_sq = "" if np.isnan(plain["DM"]) else (b if plain["mean_d"] > 0 else a)
            rows.append({"Ticker": t, "Model_A": a, "Model_B": b, "Winner_by_MAE": winner_mae,
                         "Winner_by_squared_loss": winner_sq, "Direction_conflict": winner_mae != winner_sq,
                         "DM_plain": plain["DM"], "p_plain": plain["p_DM"],
                         "DM_HLN": ref["DM_HLN"], "p_HLN": ref["p_HLN"]})
    out = pd.DataFrame(rows)
    out["Sig_F7"] = out["p_plain"] < 0.05
    out["p_Holm84"] = ap.holm(out["p_HLN"].fillna(1.0))
    out["Sig_ref84"] = out["p_Holm84"] < 0.05
    out["Primary_pair"] = [(a, b) in ap.PRIMARY_PAIRS for a, b in zip(out.Model_A, out.Model_B)]
    return out


def f7_summary(fam, ref_primary, seed):
    """Counts and agreement with the reference protocol (9 primary tests, NW + HLN + Holm)."""
    prim = fam[fam.Primary_pair].merge(
        ref_primary[["Ticker", "Model_A", "Model_B", "Sig_Holm"]], on=["Ticker", "Model_A", "Model_B"])
    return {"Seed": seed,
            "F7_significant_of_84": int(fam.Sig_F7.sum()),
            "F7_significant_with_direction_conflict": int((fam.Sig_F7 & fam.Direction_conflict).sum()),
            "Direction_conflicts_of_84": int(fam.Direction_conflict.sum()),
            "Ref_way_significant_of_84": int(fam.Sig_ref84.sum()),
            "Ref_protocol_significant_of_9": int(ref_primary.Sig_Holm.sum()),
            "F7_significant_of_9_primary": int(prim.Sig_F7.sum()),
            "Primary_agreement_of_9": int((prim.Sig_F7 == prim.Sig_Holm).sum())}


# ---------------------------------------------------------------- Monte Carlo
def loss_differentials(combined):
    """n x 84 matrix of squared-loss differentials (columns: ticker, A, B), centred at zero."""
    cols, mats = [], []
    for t, df in combined.items():
        y = df["y_true"].values
        for a, b in PAIRS28:
            mats.append((y - df[a].values) ** 2 - (y - df[b].values) ** 2)
            cols.append((t, a, b))
    D = np.column_stack(mats)
    return D - D.mean(axis=0), cols


def stationary_bootstrap_index(n_src, n_out, mean_block, rng):
    """Politis-Romano stationary bootstrap: blocks start at random days, lengths geometric with mean
    `mean_block`, wrapping round the end of the sample."""
    idx = np.empty(n_out, dtype=int)
    idx[0] = rng.integers(n_src)
    new_block = rng.random(n_out) < 1.0 / mean_block
    jumps = rng.integers(n_src, size=n_out)
    for i in range(1, n_out):
        idx[i] = jumps[i] if new_block[i] else (idx[i - 1] + 1) % n_src
    return idx


def dm_matrix(D, lag, h=1):
    """Vectorised DM for every column of D (identical formulas to ap.dm_test).
    Returns plain-variance normal p-values and Newey-West HLN Student-t p-values."""
    n = D.shape[0]
    dbar = D.mean(axis=0)
    Dc = D - dbar
    g0 = (Dc * Dc).sum(axis=0) / n
    lr = g0.copy()
    for k in range(1, lag + 1):
        lr += 2 * (1 - k / (lag + 1)) * (Dc[k:] * Dc[:-k]).sum(axis=0) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        dm_plain = np.where(g0 > 0, dbar / np.sqrt(g0 / n), np.nan)
        dm_nw = np.where(lr > 0, dbar / np.sqrt(lr / n), np.nan)
    hln = dm_nw * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p_plain = 2 * stats.norm.sf(np.abs(dm_plain))
    p_hln = 2 * stats.t.sf(np.abs(hln), df=n - 1)
    return np.nan_to_num(p_plain, nan=1.0), np.nan_to_num(p_hln, nan=1.0)


def run_monte_carlo(D, cols, n_out, mean_block, reps, seed=MC_SEED, alpha=0.05):
    """One setting. Returns per-test rejection counts and family results."""
    rng = np.random.default_rng([seed, n_out, int(mean_block * 10)])
    lag = ap.nw_lag(n_out)
    k = D.shape[1]
    prim = np.array([(a, b) in ap.PRIMARY_PAIRS for _, a, b in cols])
    rej_plain = np.zeros(k, dtype=int)
    rej_hln = np.zeros(k, dtype=int)
    fam_plain = fam_holm84 = fam_holm9 = 0
    for _ in range(reps):
        idx = stationary_bootstrap_index(D.shape[0], n_out, mean_block, rng)
        p_plain, p_hln = dm_matrix(D[idx], lag)
        rp, rh = p_plain < alpha, p_hln < alpha
        rej_plain += rp
        rej_hln += rh
        fam_plain += bool(rp.any())                          # (c) any of 84, no correction
        fam_holm84 += bool(p_hln.min() * k < alpha)          # (d) Holm rejects something iff min p < alpha/k
        fam_holm9 += bool(p_hln[prim].min() * prim.sum() < alpha)   # context: the protocol's 9 primary tests
    return {"rej_plain": rej_plain, "rej_hln": rej_hln, "fam_plain": fam_plain,
            "fam_holm84": fam_holm84, "fam_holm9": fam_holm9, "lag": lag}


def binom_ci(x, n):
    ci = stats.binomtest(int(x), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def mc_table(results, reps):
    """results: {(n, block): run_monte_carlo output}. One row per setting and procedure."""
    rows = []
    for (n, block), r in results.items():
        for name, counts in [("(a) Plain DM, single test", r["rej_plain"]), ("(b) NW + HLN DM, single test", r["rej_hln"])]:
            rates = counts / reps
            lo, hi = binom_ci(int(np.median(counts)), reps)
            rows.append({"n": n, "Mean_block": block, "Procedure": name, "Rate": float(rates.mean()),
                         "Rate_min_over_84": float(rates.min()), "Rate_max_over_84": float(rates.max()),
                         "CI95_low_median_test": lo, "CI95_high_median_test": hi})
        for name, x in [("(c) 84 plain tests, no correction: P(at least one false winner)", r["fam_plain"]),
                        ("(d) 84 NW + HLN tests, Holm: P(at least one false winner)", r["fam_holm84"]),
                        ("Context: 9 primary NW + HLN tests, Holm", r["fam_holm9"])]:
            lo, hi = binom_ci(x, reps)
            rows.append({"n": n, "Mean_block": block, "Procedure": name, "Rate": x / reps,
                         "Rate_min_over_84": np.nan, "Rate_max_over_84": np.nan,
                         "CI95_low_median_test": lo, "CI95_high_median_test": hi})
    return pd.DataFrame(rows)
