"""
article2_diagnostics.py
Part of the code for the fault-injection study (see README).

(1) The seven diagnostic checks C1-C7 with the fail rules frozen in the study protocol,
    Section 6 (frozen 11 Nov 2026). Each check reads a "run record" (what a pipeline saves) and
    returns PASS or FAIL with a short reason. These functions are the core of the open-source tool.
(2) The combined run: all eight faults injected together (Protocol Section 3, "Combined run").

Operational definitions (logged in the deviation log before any check was run on fault output):
  C2  training forecasts of each base component and each baseline; returns R2 on log returns
      against the previous actual price.
  C3  every reported test forecast column; declared unit tags plus the 0.5-2.0 MAE-ratio band.
  C4  "forecast" correlation is measured on implied returns (forecast / previous actual - 1),
      because price-level forecasts of any two sensible models correlate above 0.99999 simply by
      tracking the price. Pairs: the three inputs of Parallel/Stacking, and each Serial stage
      against the standalone model it should differ from.
  C5  Serial Stage 2 against standalone LSTM and Stage 3 against standalone XGBoost; equality =
      largest absolute difference below 0.000001 USD.
  C6  base components (ARFIMA, LSTM, XGBoost) re-forecast by their declared one-step method from
      data cut at day t, with the run's own fitted models; reproduced if within 0.01% of the price.
      Baselines are audited by C7 (their models are not kept).
  C7  "ranking loss differs from test loss" read as: the winner reported for a test is decided by a
      loss other than the test loss.
"""

import copy

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import article2_pipeline as ap
import article2_faults as af

CHECKS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
CHECK_NAMES = {"C1": "Estimation log audit", "C2": "Training-fit sanity", "C3": "Unit audit",
               "C4": "Component identity", "C5": "Stage contribution", "C6": "Forecast-origin audit",
               "C7": "Evaluation-spec audit"}
EXPECTED = {"C1": ["F1"], "C2": ["F4"], "C3": ["F3"], "C4": ["F5", "F6"], "C5": ["F6"],
            "C6": ["F2", "F4"], "C7": ["F7", "F8"]}            # Protocol Section 6, "expected to catch"
C6_DAYS, C6_SEED, C6_TOL = 20, 2026, 1e-4                    # 20 random test days; 0.01% of price

REFERENCE_SPEC = {"hac": True, "hln": True, "n_tests": 9, "correction": "Holm", "winner_from_test_loss": True,
                  "baseline_train_window": "2015-2020", "baseline_horizon": "one-step", "baseline_eval_days": 501}
PROTOCOL_SPEC = dict(REFERENCE_SPEC, research_train_window="2015-2020", research_horizon="one-step",
                     research_eval_days=501)


# ---------------------------------------------------------------- the seven checks
def c1_estimation_log(rec):
    """Fail if any grid candidate is missing from the log or a failure is not reported."""
    reasons = []
    for t, g in rec["arfima_grid_log"].items():
        if g["logged"] < g["expected"]:
            reasons.append(f"{t}: {g['expected'] - g['logged']} of {g['expected']} candidates missing from the log")
        if g.get("unreported_failures", 0) > 0:
            reasons.append(f"{t}: {g['unreported_failures']} unreported failures")
    return bool(reasons), "; ".join(reasons) or "all candidates logged"


def _returns_r2(y, f, prev):
    ok = np.isfinite(y) & np.isfinite(f) & np.isfinite(prev) & (y > 0) & (f > 0) & (prev > 0)
    return r2_score(np.log(y[ok] / prev[ok]), np.log(f[ok] / prev[ok]))


def c2_training_fit(rec):
    """Fail if any component's training returns R2 > 0.20 or training MAE ratio < 0.80."""
    reasons, rows, skipped = [], [], []
    for (t, name), df in rec["train_forecasts"].items():
        y, f, prev = (df[c].values.astype(float) for c in ["y_true", "forecast", "Persistence"])
        ok = np.isfinite(y) & np.isfinite(f) & np.isfinite(prev)
        if ok.sum() < 30:                       # no training forecasts saved (e.g. F8 recursive residual hybrids)
            skipped.append(f"{t} {name}")
            continue
        ratio = np.abs(y[ok] - f[ok]).mean() / np.abs(y[ok] - prev[ok]).mean()
        r2 = _returns_r2(y, f, prev)
        rows.append({"Ticker": t, "Component": name, "Train_MAE_ratio": ratio, "Train_returns_R2": r2})
        if r2 > 0.20 or ratio < 0.80:
            reasons.append(f"{t} {name}: R2 {r2:.3f}, ratio {ratio:.3f}")
    note = f" | not audited (no training forecasts saved): {len(skipped)}" if skipped else ""
    return bool(reasons), ("; ".join(reasons[:8]) + (f" (+{len(reasons) - 8} more)" if len(reasons) > 8 else "")
                           if reasons else "training fit plausible") + note, pd.DataFrame(rows)


def c3_units(rec):
    """Fail if unit tags differ or any test MAE ratio is outside 0.5-2.0."""
    reasons = []
    tags = set(rec["unit_tags"].values())
    if len(tags) > 1:
        reasons.append(f"unit tags differ: {sorted(tags)}")
    for (t, name), ratio in rec["test_ratios"].items():
        if not (0.5 <= ratio <= 2.0):
            reasons.append(f"{t} {name}: ratio {ratio:.3f}")
    return bool(reasons), "; ".join(reasons[:8]) + (f" (+{len(reasons) - 8} more)" if len(reasons) > 8 else "") \
        if reasons else "one unit, ratios in band"


def _implied(f, prev):
    return f / prev - 1


def _identical(a, b, prev_a, prev_b):
    diff = float(np.max(np.abs(a - b)))
    ra, rb = _implied(a, prev_a), _implied(b, prev_b)
    corr = float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else 1.0
    return (corr > 0.99999) or (diff < 1e-6), corr, diff


def c4_component_identity(rec):
    """Fail if any pair of components has implied-return correlation > 0.99999 or max difference < 0.000001."""
    reasons, rows = [], []
    for t, pairs in rec["component_pairs"].items():
        for (na, a, pa), (nb, b, pb) in pairs:
            same, corr, diff = _identical(a, b, pa, pb)
            rows.append({"Ticker": t, "Pair": f"{na} vs {nb}", "Corr_implied_returns": corr, "Max_abs_diff": diff})
            if same:
                reasons.append(f"{t}: {na} = {nb} (corr {corr:.6f}, max diff {diff:.2e})")
    return bool(reasons), "; ".join(reasons) or "no component is a copy", pd.DataFrame(rows)


def c5_stage_contribution(rec):
    """Fail if a Serial stage has no training log, or its output equals the standalone model."""
    reasons = []
    s = rec["serial"]
    if not s["stage2_log"]:
        reasons.append("Stage 2 has no training log")
    if not s["stage3_log"]:
        reasons.append("Stage 3 has no training log")
    for t, (s2, lstm, s3, xgbf) in s["outputs"].items():
        if np.max(np.abs(s2 - lstm)) < 1e-6:
            reasons.append(f"{t}: Stage 2 equals standalone LSTM")
        if np.max(np.abs(s3 - xgbf)) < 1e-6:
            reasons.append(f"{t}: Stage 3 equals standalone XGBoost")
    return bool(reasons), "; ".join(reasons) or "each stage trained and distinct"


def c6_forecast_origin(rec):
    """Fail if, on any of the audited test days, the declared one-step re-forecast from data cut at
    day t does not reproduce the stored forecast (tolerance 0.01% of the price)."""
    a = rec["c6_audit"]
    bad = a[a.Rel_diff > C6_TOL]
    if bad.empty:
        return False, f"{len(a)} re-forecasts reproduced (max rel diff {a.Rel_diff.max():.1e})", a
    summ = bad.groupby(["Ticker", "Component"]).size()
    return True, "; ".join(f"{t} {c}: {n} of {C6_DAYS} days not reproduced" for (t, c), n in summ.items()), a


def c7_evaluation_spec(rec):
    """Fail on: no HAC variance; test count differs from the protocol; no multiplicity correction;
    winner decided by a loss other than the test loss; baselines differ in training window, horizon or days."""
    s, reasons = rec["spec"], []
    if not s["hac"]:
        reasons.append("no HAC variance")
    if s["n_tests"] != PROTOCOL_SPEC["n_tests"]:
        reasons.append(f"{s['n_tests']} tests vs {PROTOCOL_SPEC['n_tests']} in the protocol")
    if s["correction"] in (None, "none"):
        reasons.append("no multiplicity correction")
    if not s["winner_from_test_loss"]:
        reasons.append("winner chosen by a different loss from the test loss")
    for k in ["train_window", "horizon", "eval_days"]:
        if s[f"baseline_{k}"] != PROTOCOL_SPEC[f"research_{k}"]:
            reasons.append(f"baselines differ in {k.replace('_', ' ')} ({s[f'baseline_{k}']} vs {PROTOCOL_SPEC[f'research_{k}']})")
    return bool(reasons), "; ".join(reasons) or "evaluation follows the protocol"


def run_checks(rec):
    out, extra = {}, {}
    out["C1"] = c1_estimation_log(rec)
    f, r, extra["C2"] = c2_training_fit(rec); out["C2"] = (f, r)
    out["C3"] = c3_units(rec)
    f, r, extra["C4"] = c4_component_identity(rec); out["C4"] = (f, r)
    out["C5"] = c5_stage_contribution(rec)
    f, r, extra["C6"] = c6_forecast_origin(rec); out["C6"] = (f, r)
    out["C7"] = c7_evaluation_spec(rec)
    return out, extra


# ---------------------------------------------------------------- C6 helpers: models kept for seed S0
def train_base_capture(data, seed):
    """Same steps as ap.train_base (v2), but also returns the fitted models and scalers so the
    forecasts can be re-computed day by day. NB4 checks the tables equal the cached reference exactly."""
    import tensorflow as tf
    import xgboost as xgb
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    tables, models = {}, {}
    for t in ap.TICKERS:
        D = data[t]
        valid = D["y_next"].notna() & D["X"].notna().all(axis=1)
        train_rows = (D["target_split"] == "train") & valid
        x_scaler = MinMaxScaler().fit(D["X"][train_rows])
        y_scaler = StandardScaler().fit(D["y_next"][train_rows].values.reshape(-1, 1))
        Xs = x_scaler.transform(D["X"].fillna(0))
        seq, idx = ap.make_sequences(Xs, ap.WINDOW)
        rows = D["X"].index[idx]
        ok = valid.loc[rows].values
        seq, rows = seq[ok], rows[ok]
        split = D["target_split"].loc[rows].values
        ys = y_scaler.transform(D["y_next"].loc[rows].values.reshape(-1, 1)).ravel()
        ap.set_all_seeds(seed)
        model = ap.build_lstm(seq.shape[2])
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=ap.PATIENCE, restore_best_weights=True)
        model.fit(seq[split == "train"], ys[split == "train"], validation_data=(seq[split == "val"], ys[split == "val"]),
                  epochs=ap.MAX_EPOCHS, batch_size=ap.BATCH_SIZE, shuffle=False, callbacks=[es], verbose=0)
        lstm_ret = pd.Series(y_scaler.inverse_transform(model.predict(seq, verbose=0)).ravel(), index=rows)
        Xv, yv, sp = D["X"][valid], D["y_next"][valid], D["target_split"][valid]
        xm = xgb.XGBRegressor(**ap.xgb_params(seed))
        xm.fit(Xv[sp == "train"], yv[sp == "train"], eval_set=[(Xv[sp == "val"], yv[sp == "val"])], verbose=False)
        xgb_ret = pd.Series(xm.predict(Xv), index=Xv.index)
        r = lstm_ret.index.intersection(xgb_ret.index)
        p_today = D["price"].loc[r]
        tables[t] = pd.DataFrame({"Date": D["target_date"].loc[r].values, "Split": D["target_split"].loc[r].values,
                                  "y_true": D["price"].shift(-1).loc[r].values, "Persistence": p_today.values,
                                  "LSTM": (p_today * np.exp(lstm_ret.loc[r])).values,
                                  "XGB": (p_today * np.exp(xgb_ret.loc[r])).values})
        models[t] = {"lstm": model, "x_scaler": x_scaler, "y_scaler": y_scaler, "xgb": xm}
    return tables, models


def xgb_leak_capture(data, seed):
    """F4 model kept for C6 (same steps as af.xgb_base with leak=True)."""
    import xgboost as xgb
    models = {}
    for t in ap.TICKERS:
        D = data[t]
        valid = D["y_next"].notna() & D["X"].notna().all(axis=1)
        Xv, yv, sp = D["X"][valid].copy(), D["y_next"][valid], D["target_split"][valid]
        Xv["leak_next_day_return"] = yv.values
        xm = xgb.XGBRegressor(**ap.xgb_params(seed))
        xm.fit(Xv[sp == "train"], yv[sp == "train"], eval_set=[(Xv[sp == "val"], yv[sp == "val"])], verbose=False)
        models[t] = xm
    return models


def c6_audit(data, stored, arfima_models, base_models, xgb_models, leak=False, days=C6_DAYS, seed=C6_SEED):
    """Re-forecast ARFIMA, LSTM and XGBoost for `days` random test days per stock from data cut at the
    forecast origin, by the declared one-step method, and compare with the stored test forecasts."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in ap.TICKERS:
        D = data[t]
        idx = D["full"].index
        test_pos = np.where(D["split_of"].values == "test")[0]
        chosen = np.sort(rng.choice(test_pos, size=days, replace=False))
        M = arfima_models[t]
        bm = base_models[t]
        for pos in chosen:
            o = pos - 1                                        # forecast origin: last day of data allowed
            day = idx[pos]
            p_o = float(D["price"].iloc[o])
            # ARFIMA: frozen parameters, series cut at the origin (the target-day value is a placeholder)
            x_cut = M["x"][:pos + 1].copy()
            x_cut[pos] = x_cut[o]
            ar_hat = float(np.exp(af.arfima_one_step(x_cut, M["best"])[pos] + M["train_mean"]))
            # LSTM: last 60 input rows up to the origin
            Xc = D["X"].iloc[:o + 1]
            win = bm["x_scaler"].transform(Xc.fillna(0).iloc[-ap.WINDOW:])[None, ...]
            l_ret = bm["y_scaler"].inverse_transform(bm["lstm"].predict(win, verbose=0)).ravel()[0]
            l_hat = p_o * np.exp(l_ret)
            # XGBoost: input row of the origin; a leaked next-day column cannot be built from cut data
            xrow = Xc.iloc[[-1]].copy()
            if leak:
                price_cut = D["price"].iloc[:o + 1]
                xrow["leak_next_day_return"] = np.log(price_cut.shift(-1) / price_cut).iloc[-1]   # NaN by construction
            x_hat = p_o * np.exp(float(xgb_models[t].predict(xrow)[0]))
            for comp, val in [("ARFIMA", ar_hat), ("LSTM", l_hat), ("XGBoost", x_hat)]:
                s = float(stored[t].loc[day, comp])
                rows.append({"Ticker": t, "Date": day, "Component": comp, "Stored": s, "Recomputed": val,
                             "Rel_diff": abs(s - val) / p_o})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the combined run (all eight faults)
def assemble_multi(base, arfima, seed, faults, scalers=None, serial=None):
    """af.assemble generalised to several faults at once (F3, F5, F6 can act together).
    Returns combined test tables, weights, coefficients and the ensemble inputs (for C4)."""
    aligned = ap.align_three(base, arfima)
    if "F3" in faults:
        aligned = af.to_minmax(aligned, scalers)
    ens_in = aligned
    if "F5" in faults:
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
        if "F6" in faults:
            s2 = te["ARFIMA"] + (te["LSTM"] - te["ARFIMA"])
            te["Stage2_ARFIMA_LSTM"] = s2
            te["Stage3_Serial_Final"] = s2 + (te["XGB"] - s2)
        else:
            s = serial[t][serial[t].Split == "test"][["Date", "Stage2_ARFIMA_LSTM", "Stage3_Serial_Final"]]
            te = te.merge(s, on="Date", how="left")
        te = te.rename(columns=ap.DISPLAY).set_index("Date")
        combined[t] = te[af.COLS]
        assert len(combined[t]) == 501 and combined[t].notna().all().all(), f"{t}: expected 501 complete test rows"
    w_rows = [{"Ticker": t, "Seed": seed, "Scheme": k, **{f"w_{m}": float(w[i]) for i, m in enumerate(ap.MODELS3)}}
              for t in ap.TICKERS for k, w in weights[t].items()]
    c_rows = [{"Ticker": t, "Seed": seed, **coefs[t]} for t in ap.TICKERS]
    return combined, pd.DataFrame(w_rows), pd.DataFrame(c_rows), ens_in


def run_combined(data, ref_base, arfima_f1f2, seed, scalers):
    """All eight faults: ARFIMA through the v1 route (F1) with a static forecast (F2); XGBoost with the
    leaked column (F4); LSTM, target and persistence in MinMax units (F3); ARFIMA replaced by LSTM in
    Parallel and Stacking (F5); pseudo-residual Serial (F6). F7 and F8 act on testing and baselines."""
    base, xlog = af.xgb_base(data, ref_base, seed, leak=True)
    combined, w, c, ens_in = assemble_multi(base, arfima_f1f2, seed, {"F3", "F5", "F6"}, scalers=scalers)
    return combined, {"xgb": xlog, "weights": w, "stacking": c}, base, ens_in


# ---------------------------------------------------------------- run records
def ens_inputs(base, arfima, faults=(), scalers=None):
    """The validation and test inputs that Parallel and Stacking actually receive (same steps as assemble)."""
    aligned = ap.align_three(base, arfima)
    if "F3" in faults:
        aligned = af.to_minmax(aligned, scalers)
    if "F5" in faults:
        aligned = copy.deepcopy(aligned)
        for t in ap.TICKERS:
            for s in ["val", "test"]:
                aligned[t][s]["ARFIMA"] = aligned[t][s]["LSTM"].values
    return aligned


def ens_pairs(ens_in, serial_test=None):
    """Component pairs for C4: the three ensemble inputs (validation + test) and Serial stages vs standalone."""
    out = {}
    for t in ap.TICKERS:
        df = pd.concat([ens_in[t]["val"], ens_in[t]["test"]])
        prev = df["Persistence"].values
        cols = {"ARFIMA": df["ARFIMA"].values, "LSTM": df["LSTM"].values, "XGBoost": df["XGB"].values}
        pairs = [((a, cols[a], prev), (b, cols[b], prev)) for a, b in [("ARFIMA", "LSTM"), ("ARFIMA", "XGBoost"), ("LSTM", "XGBoost")]]
        if serial_test is not None:
            st = serial_test[t]
            pv = st["Persistence"].values
            pairs += [(("Serial Stage 2", st["Serial (ARFIMA+LSTM)"].values, pv), ("LSTM", st["LSTM"].values, pv)),
                      (("Serial Final", st["Serial (Final)"].values, pv), ("XGBoost", st["XGBoost"].values, pv))]
        out[t] = pairs
    return out


def test_ratios(combined, baselines=None):
    out = {}
    for t, df in combined.items():
        pers = np.abs(df["y_true"] - df["Persistence"]).mean()
        for m in af.ALL_MODELS:
            out[(t, m)] = float(np.abs(df["y_true"] - df[m]).mean() / pers)
    if baselines is not None:
        for t, tab in baselines.items():
            te = tab[tab.Split == "test"]
            pers = np.abs(te["y_true"] - te["Persistence"]).mean()
            for m in [c for c in tab.columns if c not in ("y_true", "Persistence", "Split")]:
                out[(t, f"Baseline {m}")] = float(np.abs(te["y_true"] - te[m]).mean() / pers)
    return out


def train_forecasts(arfima_tables, base_tables, baselines=None, data=None):
    out = {}
    for t in ap.TICKERS:
        a = arfima_tables[t]
        split = a["Split"] if "Split" in a else data[t]["split_of"].reindex(a.index)
        a = a[split.values == "train"]
        out[(t, "ARFIMA")] = pd.DataFrame({"y_true": a["y_true"], "Persistence": a["Persistence"], "forecast": a["ARFIMA"]})
        b = base_tables[t]
        b = b[b.Split == "train"]
        for col, name in [("LSTM", "LSTM"), ("XGB", "XGBoost")]:
            out[(t, name)] = pd.DataFrame({"y_true": b["y_true"].values, "Persistence": b["Persistence"].values,
                                           "forecast": b[col].values})
        if baselines is not None:
            tab = baselines[t]
            tr = tab[tab.Split == "train"]
            for m in [c for c in tab.columns if c not in ("y_true", "Persistence", "Split")]:
                out[(t, f"Baseline {m}")] = pd.DataFrame({"y_true": tr["y_true"].values,
                                                         "Persistence": tr["Persistence"].values,
                                                         "forecast": tr[m].values})
    return out


def detection_matrix(results):
    """results: {run: {check: (fail, reason)}} -> PASS/FAIL table with expected detections marked."""
    rows = []
    for run, res in results.items():
        row = {"Run": run}
        for c in CHECKS:
            fail = res[c][0]
            mark = "FAIL" if fail else "pass"
            if fail and run in EXPECTED[c]:
                mark = "FAIL (expected)"
            elif not fail and run in EXPECTED[c]:
                mark = "pass (MISSED)"
            row[c] = mark
        rows.append(row)
    return pd.DataFrame(rows).set_index("Run")
