"""
article2_faults.py
Part of the code for the fault-injection study (see README).

Fault injection for model faults F1-F6, as frozen in the study protocol, Section 3
(frozen 11 Nov 2026). Each fault is injected ALONE into the reference pipeline.

Design rule: the reference code in article2_pipeline.py is imported, never edited. This module
only adds (a) a re-implementation of the v2 ARFIMA estimation (v2, line for line),
needed because F1 and F2 change ARFIMA, and (b) one switch per fault. With no fault switched
on, every function here must reproduce the reference exactly; NB2 checks this before any
fault runs.

Fault summary (protocol Section 3)
  F1 Silent estimation failure: ARFIMA candidates are fitted with statsmodels ARIMA, which
     rejects fractional d. A try/except drops every rejected candidate without logging it.
  F2 Horizon mismatch: ARFIMA makes one static multi-step forecast from its last training
     observation. Validation gets steps 1-503 (dates line up). Test gets steps 1-501 placed on
     the 501 test dates (the misalignment). Training values stay one-step. LSTM/XGBoost unchanged.
  F3 Unit mismatch: the target, LSTM and persistence (the lagged target) are expressed in
     training-period MinMax units of the price; ARFIMA, XGBoost and Serial stay in USD.
     Parallel and Stacking are fitted on this mixed validation set; metrics use the mixed set.
  F4 Target leakage: the standalone XGBoost gets one extra input column equal to its target
     (next-day log return). Serial Stage 3 is not changed.
  F5 Placeholder component: inside Parallel and Stacking only, ARFIMA validation and test
     forecasts are replaced by LSTM's. Standalone ARFIMA and Serial are unchanged.
  F6 Pseudo-residual cascade: Serial Stage 2 = ARFIMA + (LSTM - ARFIMA) and
     Stage 3 = Stage 2 + (XGBoost - Stage 2). No model is trained on residuals.
"""

import copy

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import article2_pipeline as ap

MODEL_FAULTS = ["F1", "F2", "F3", "F4", "F5", "F6"]
FAULT_NAMES = {
    "F1": "Silent estimation failure", "F2": "Horizon mismatch", "F3": "Unit mismatch",
    "F4": "Target leakage", "F5": "Placeholder component", "F6": "Pseudo-residual cascade",
}
# Which reference parts each fault changes (everything else is reused from the same-seed reference)
NEEDS = {
    "F1": {"arfima", "serial"}, "F2": {"arfima", "serial"}, "F3": set(),
    "F4": {"xgb"}, "F5": set(), "F6": set(),
}

ALL_MODELS = ["ARFIMA", "LSTM", "XGBoost", "Serial (ARFIMA+LSTM)", "Serial (Final)",
              "Parallel Equal", "Parallel InvError", "Parallel SoftMax", "Stacking"]
ARCHITECTURES = ["Serial (Final)", "Parallel Equal", "Parallel InvError", "Parallel SoftMax", "Stacking"]
COLS = ["y_true", "Persistence"] + ALL_MODELS

# ---------------------------------------------------------------- ARFIMA (identical to v2)
P_RANGE = [0, 1, 2]
Q_RANGE = [0, 1, 2]
D_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
BURN_IN = 50


def frac_weights(d, n):
    w = np.empty(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (k - 1 - d) / k
    return w


def frac_diff(x, d):
    w = frac_weights(d, len(x))
    return np.convolve(x, w)[:len(x)], w


def fit_arma(z_fit, p, q):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(z_fit, order=(p, 0, q), trend="c", enforce_stationarity=True, enforce_invertibility=True)
    res = model.fit(disp=False, maxiter=1000)
    return res, bool(res.mle_retvals.get("converged", True))


def fit_arima_integer_only(x_fit, p, d, q):
    """F1: the v1 route. statsmodels ARIMA takes d inside `order` and raises for any non-integer d."""
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(x_fit, order=(p, d, q), trend="c", enforce_stationarity=True, enforce_invertibility=True)
    res = model.fit(method_kwargs={"maxiter": 1000, "disp": False})
    return res, bool(res.mle_retvals.get("converged", True))


def arfima_grid_search(x, n_train, fault=None, burn_in=BURN_IN):
    """v2 grid search. With fault='F1', candidates go through the integer-only ARIMA route and
    every exception is swallowed: the candidate disappears from the log (as in v1).
    `silent_drops` is an instrument for this study only; a researcher running v1 would not see it."""
    log, best, silent_drops = [], None, 0
    for d in D_GRID:
        z, _ = frac_diff(x, d)
        for p in P_RANGE:
            for q in Q_RANGE:
                if fault == "F1":
                    try:
                        res, conv = fit_arima_integer_only(x[burn_in:n_train], p, d, q)
                    except Exception:
                        silent_drops += 1
                        continue
                    aic, status = res.aic, ("success" if conv else "not_converged")
                else:
                    try:
                        res, conv = fit_arma(z[burn_in:n_train], p, q)
                        aic, status = res.aic, ("success" if conv else "not_converged")
                    except Exception as e:
                        res, aic, status = None, np.nan, f"failed: {str(e)[:60]}"
                log.append({"p": p, "d": d, "q": q, "AIC": aic, "Status": status})
                if status == "success" and (best is None or aic < best["AIC"]):
                    best = {"p": p, "d": d, "q": q, "AIC": aic, "res": res,
                            "route": "arima_integer" if fault == "F1" else "fracdiff_sarimax"}
    return best, pd.DataFrame(log), silent_drops


def arfima_one_step(x, best, burn_in=BURN_IN):
    """One-step forecasts with parameters frozen from training (v2 CELL 2). For the F1 route the
    same ARIMA specification is re-filtered with its own parameters (d = 0, so no fractional part)."""
    if best["route"] == "arima_integer":
        from statsmodels.tsa.arima.model import ARIMA
        full = ARIMA(x[burn_in:], order=(best["p"], 0, best["q"]), trend="c",
                     enforce_stationarity=True, enforce_invertibility=True)
        x_hat = np.full(len(x), np.nan)
        x_hat[burn_in:] = full.filter(best["res"].params).predict()
        return x_hat
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    z, w = frac_diff(x, best["d"])
    full = SARIMAX(z[burn_in:], order=(best["p"], 0, best["q"]), trend="c",
                   enforce_stationarity=True, enforce_invertibility=True)
    filt = full.filter(best["res"].params)
    z_hat = np.full(len(x), np.nan)
    z_hat[burn_in:] = filt.predict()
    past_part = np.convolve(x, w)[:len(x)] - x
    return z_hat - past_part


def arfima_static(x, best, n_train, horizon):
    """F2: one static forecast of `horizon` steps from the last training observation.
    z is forecast by the fitted ARMA; x is rebuilt recursively, with forecasts standing in for
    every unknown future value (no new information after training)."""
    _, w = frac_diff(x, best["d"])
    zf = np.asarray(best["res"].forecast(steps=horizon))
    xs = np.empty(n_train + horizon)
    xs[:n_train] = x[:n_train]
    for j in range(horizon):
        t = n_train + j
        xs[t] = zf[j] - np.dot(w[1:t + 1], xs[t - 1::-1])
    return xs[n_train:]


def _arfima_table(D, x_hat, train_mean):
    """Same layout as the v2 prediction CSVs (CELL 7): Date index, y_true, ARFIMA, Persistence, Split."""
    price = D["price"].values
    out = pd.DataFrame({"Date": D["full"].index, "y_true": price, "ARFIMA": np.exp(x_hat + train_mean),
                        "Persistence": np.concatenate([[np.nan], price[:-1]]),
                        "Split": D["split_of"].values})
    out.loc[:BURN_IN - 1, "ARFIMA"] = np.nan
    return out.dropna(subset=["ARFIMA", "Persistence"]).set_index("Date")


def fit_arfima(data, fault=None):
    """Re-estimate ARFIMA for all stocks. fault=None reproduces v2; 'F1' injects the silent failure.
    Returns (tables, models, log) where tables has the same layout as ap.load_arfima()."""
    tables, models, rows = {}, {}, []
    for t in ap.TICKERS:
        D = data[t]
        n_train = int((D["split_of"] == "train").sum())
        logp = np.log(D["price"].values)
        train_mean = logp[:n_train].mean()
        x = logp - train_mean
        best, grid, drops = arfima_grid_search(x, n_train, fault=fault)
        tables[t] = _arfima_table(D, arfima_one_step(x, best), train_mean)
        models[t] = {"best": best, "x": x, "train_mean": train_mean, "n_train": n_train, "grid": grid}
        counts = grid["Status"].value_counts().to_dict()
        rows.append({"Ticker": t, "Fault": fault or "REF", "p": best["p"], "d": best["d"], "q": best["q"],
                     "AIC": float(best["AIC"]), "Fits_logged": len(grid),
                     "Fits_success": counts.get("success", 0),
                     "Fits_not_converged": counts.get("not_converged", 0),
                     "Fits_failed_logged": int(sum(v for k, v in counts.items() if k.startswith("failed"))),
                     "Fits_dropped_silently": drops})
    return tables, models, pd.DataFrame(rows)


def static_arfima_tables(data, ref_models):
    """F2: train rows keep the v2 one-step values; validation = static steps 1..n_val;
    test = static steps 1..n_test placed on the test dates (the v1 misalignment)."""
    tables = {}
    for t in ap.TICKERS:
        D, M = data[t], ref_models[t]
        x, n_train = M["x"], M["n_train"]
        split = D["split_of"].values
        n_val, n_test = int((split == "val").sum()), int((split == "test").sum())
        x_hat = arfima_one_step(x, M["best"])
        x_hat[split == "val"] = arfima_static(x, M["best"], n_train, n_val)
        x_hat[split == "test"] = arfima_static(x, M["best"], n_train, n_test)
        tables[t] = _arfima_table(D, x_hat, M["train_mean"])
    return tables


def compare_arfima(a, b):
    """Largest absolute difference (USD) between two ARFIMA table sets, per stock and split."""
    rows = []
    for t in ap.TICKERS:
        m = a[t][["ARFIMA", "Split"]].join(b[t][["ARFIMA"]], rsuffix="_b", how="outer")
        for s in ["train", "val", "test"]:
            part = m[m.Split == s]
            rows.append({"Ticker": t, "Split": s, "Rows_a": int(part["ARFIMA"].notna().sum()),
                         "Rows_b": int(part["ARFIMA_b"].notna().sum()),
                         "Max_abs_diff_USD": float(np.nanmax(np.abs(part["ARFIMA"] - part["ARFIMA_b"])))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- F4: XGBoost with a leaked target column
def xgb_base(data, base_ref, seed, leak=False):
    """Retrain the standalone XGBoost exactly as ap.train_base (v2), optionally with
    one extra input equal to the target. LSTM columns are copied from the same-seed reference.
    With leak=False this must reproduce the reference XGB column exactly."""
    import xgboost as xgb
    tables, logs = {}, []
    for t in ap.TICKERS:
        D = data[t]
        valid = D["y_next"].notna() & D["X"].notna().all(axis=1)
        Xv, yv, sp = D["X"][valid].copy(), D["y_next"][valid], D["target_split"][valid]
        if leak:
            Xv["leak_next_day_return"] = yv.values
        xm = xgb.XGBRegressor(**ap.xgb_params(seed))
        xm.fit(Xv[sp == "train"], yv[sp == "train"], eval_set=[(Xv[sp == "val"], yv[sp == "val"])], verbose=False)
        xgb_ret = pd.Series(xm.predict(Xv), index=Xv.index)
        origin = pd.Series(D["full"].index, index=D["target_date"].values)
        origin = origin[origin.index.notna()]
        tb = base_ref[t].copy()
        r = origin.loc[tb["Date"].values].values
        tb["XGB"] = (D["price"].loc[r] * np.exp(xgb_ret.loc[r])).values
        tables[t] = tb
        logs.append({"Ticker": t, "Seed": seed, "Leak": leak,
                     "XGB_best_iteration": int(xm.best_iteration),
                     "XGB_trees_kept": int(xm.best_iteration) + 1,
                     "XGB_train_R2_returns": float(r2_score(yv[sp == "train"], xm.predict(Xv[sp == "train"]))),
                     "XGB_test_R2_returns": float(r2_score(yv[sp == "test"], xm.predict(Xv[sp == "test"]))),
                     "Leak_feature_importance": float(xm.feature_importances_[-1]) if leak else np.nan})
    return tables, pd.DataFrame(logs)


# ---------------------------------------------------------------- F3: training-period MinMax units
def minmax_scalers(data):
    """One scaler per stock, fitted on training-period adjusted close prices."""
    return {t: MinMaxScaler().fit(data[t]["price"][data[t]["split_of"] == "train"].values.reshape(-1, 1))
            for t in ap.TICKERS}


def to_minmax(aligned, scalers, cols=("y_true", "Persistence", "LSTM")):
    out = copy.deepcopy(aligned)
    for t in ap.TICKERS:
        for s in ["val", "test"]:
            for c in cols:
                out[t][s][c] = scalers[t].transform(out[t][s][c].values.reshape(-1, 1)).ravel()
    return out


# ---------------------------------------------------------------- assemble one run (reference or one fault)
def assemble(base, arfima, serial, seed, fault=None, scalers=None):
    """Rebuild Parallel and Stacking and the combined test tables. With fault=None this is
    exactly ap.run_reference after training (same functions, same order)."""
    aligned = ap.align_three(base, arfima)
    if fault == "F3":
        aligned = to_minmax(aligned, scalers)
    ens_in = aligned
    if fault == "F5":
        ens_in = copy.deepcopy(aligned)
        for t in ap.TICKERS:
            for s in ["val", "test"]:
                ens_in[t][s]["ARFIMA"] = ens_in[t][s]["LSTM"].values
    par, weights = ap.run_parallel(ens_in)
    stk, coefs = ap.run_stacking(ens_in)
    combined = {}
    for t in ap.TICKERS:
        te = aligned[t]["test"].copy()
        for k, v in par[t]["test"].items():
            te[k] = v
        te["Stacking"] = stk[t]["test"]
        if fault == "F6":
            s2 = te["ARFIMA"] + (te["LSTM"] - te["ARFIMA"])
            te["Stage2_ARFIMA_LSTM"] = s2
            te["Stage3_Serial_Final"] = s2 + (te["XGB"] - s2)
        else:
            s = serial[t][serial[t].Split == "test"][["Date", "Stage2_ARFIMA_LSTM", "Stage3_Serial_Final"]]
            te = te.merge(s, on="Date", how="left")
        te = te.rename(columns=ap.DISPLAY).set_index("Date")
        combined[t] = te[COLS]
        assert len(combined[t]) == 501 and combined[t].notna().all().all(), f"{t}: expected 501 complete test rows"
    w_rows = [{"Ticker": t, "Seed": seed, "Scheme": k, **{f"w_{m}": float(w[i]) for i, m in enumerate(ap.MODELS3)}}
              for t in ap.TICKERS for k, w in weights[t].items()]
    c_rows = [{"Ticker": t, "Seed": seed, **coefs[t]} for t in ap.TICKERS]
    return combined, pd.DataFrame(w_rows), pd.DataFrame(c_rows)


def run_fault(fault, seed, data, ref, fault_arfima=None, scalers=None):
    """One fault, one seed. `ref` holds the same-seed reference parts: base, serial, arfima.
    Returns combined test tables and a dict of mechanism logs."""
    base, serial, arfima, logs = ref["base"], ref["serial"], ref["arfima"], {}
    if fault in ("F1", "F2"):
        arfima = fault_arfima
        serial, logs["serial"] = ap.run_serial(data, arfima, seed)
    if fault == "F4":
        base, logs["xgb"] = xgb_base(data, base, seed, leak=True)
    combined, w, c = assemble(base, arfima, serial, seed, fault=fault, scalers=scalers)
    logs["weights"], logs["stacking"] = w, c
    if fault == "F6":
        logs["equality"] = pd.DataFrame([{
            "Ticker": t, "Seed": seed,
            "Max_abs_diff_Final_vs_XGBoost": float(np.max(np.abs(combined[t]["Serial (Final)"] - combined[t]["XGBoost"]))),
            "Corr_Final_vs_XGBoost": float(np.corrcoef(combined[t]["Serial (Final)"], combined[t]["XGBoost"])[0, 1]),
            "Max_abs_diff_Stage2_vs_LSTM": float(np.max(np.abs(combined[t]["Serial (ARFIMA+LSTM)"] - combined[t]["LSTM"]))),
        } for t in ap.TICKERS])
    if fault == "F3":
        logs["units"] = pd.DataFrame([{
            "Ticker": t, "Seed": seed,
            "y_true_test_min": float(combined[t]["y_true"].min()), "y_true_test_max": float(combined[t]["y_true"].max()),
            "ARFIMA_test_mean": float(combined[t]["ARFIMA"].mean()), "LSTM_test_mean": float(combined[t]["LSTM"].mean()),
            "Ratio_ARFIMA_to_target_mean": float(combined[t]["ARFIMA"].mean() / combined[t]["y_true"].mean()),
        } for t in ap.TICKERS])
    return combined, logs


# ---------------------------------------------------------------- outcomes (protocol Section 4)
def metrics_long(combined, label, seed):
    """MAE ratio, MAE, RMSE, price R2 and returns R2 for every model and stock."""
    rows = []
    for t, df in combined.items():
        pers = mean_absolute_error(df["y_true"], df["Persistence"])
        prev = df["Persistence"].values
        for m in ALL_MODELS + ["Persistence"]:
            mt = ap.compute_metrics(df["y_true"], df[m])
            ok = (df["y_true"].values > 0).all() and (df[m].values > 0).all() and (prev > 0).all()
            r2_ret = r2_score(np.log(df["y_true"].values / prev), np.log(df[m].values / prev)) if ok else np.nan
            rows.append({"Run": label, "Seed": seed, "Ticker": t, "Model": m, "MAE_ratio": mt["MAE"] / pers,
                         "MAE": mt["MAE"], "RMSE": mt["RMSE"], "R2_price": mt["R2"], "R2_returns": r2_ret})
    return pd.DataFrame(rows)


def dm_sets(combined):
    """Primary 9 (Holm), versus persistence 27 (Holm within set), versus ARFIMA 15 (Holm within set);
    the same three sets as v2. Squared loss, Newey-West, HLN."""
    out = []
    sets = {"primary": ap.PRIMARY_PAIRS,
            "vs_persistence": [(m, "Persistence") for m in ALL_MODELS],
            "vs_ARFIMA": [(m, "ARFIMA") for m in ARCHITECTURES]}
    for name, pairs in sets.items():
        r = ap.dm_set(combined, pairs)
        r["Set"] = name
        r["p_Holm"] = ap.holm(r["p_HLN"].fillna(1.0))
        r["Sig_5%"] = r["p_HLN"] < 0.05
        r["Sig_Holm"] = r["p_Holm"] < 0.05
        out.append(r)
    return pd.concat(out, ignore_index=True)


def sig_counts(dm):
    return dm.groupby("Set")["Sig_Holm"].sum().astype(int).to_dict()


def ranking_tau(ratios_fault, ratios_ref, decimals=3):
    """Kendall's tau (tau-b, ties allowed) between two rankings of the nine models by mean MAE ratio.
    Ratios are rounded to three decimals first (protocol rounding rule), so that differences
    smaller than the reported precision cannot reorder models."""
    a = ratios_fault.groupby("Model")["MAE_ratio"].mean().loc[ALL_MODELS].round(decimals)
    b = ratios_ref.groupby("Model")["MAE_ratio"].mean().loc[ALL_MODELS].round(decimals)
    return float(stats.kendalltau(a.values, b.values).statistic)


def max_table_diff(a, b):
    """Largest absolute difference between two combined-table sets (all columns, all stocks)."""
    return max(float(np.max(np.abs(a[t][COLS].values - b[t][COLS].values))) for t in ap.TICKERS)
