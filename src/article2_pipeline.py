"""
article2_pipeline.py
Part of the code for the fault-injection study (see README).

Reference pipeline (v2, the verified version). Every function mirrors the verified v2
notebooks (run 8 Oct 2026) line for line, with one
change only: the random seed is a function argument instead of a fixed constant, so the
reference can be re-run with several seeds.

Fault injection (F1-F8) is not in this module. It lives in article2_faults.py and
article2_tests.py, which import this module unchanged, so no fault code can influence the reference.

Drive layout: everything lives in the separate master folder "My Drive/Article 2".
The study reads and writes only inside this folder. The v2 inputs were copied once
(11 Nov 2026) into a frozen snapshot, and a checksum manifest detects any later change.
  <ROOT>/02_Inputs_v2_snapshot/splits/{TICKER}_{train|val|test}.csv
  <ROOT>/02_Inputs_v2_snapshot/arfima_predictions/{TICKER}_{train|val|test}_predictions.csv
  <ROOT>/02_Inputs_v2_snapshot/combined_test_forecasts/{TICKER}_test_all_models.csv
  <ROOT>/05_Outputs/...   (all results)
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

TICKERS = ["JPM", "NEE", "XOM"]
TARGET = "Adj Close"
MODELS3 = ["ARFIMA", "LSTM", "XGB"]

# Hyperparameters (identical to v2 and to Istikhor et al., 2026)
WINDOW = 60
LSTM_UNITS = (64, 32)
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
GRID_STEP = 0.05

RATIO_COLS = ["Open", "High", "Low", "sma_10", "sma_20", "sma_50", "ema_10", "ema_20",
              "bb_mid_20", "bb_up_20", "bb_low_20"]
KEEP_COLS = ["ret_1", "rv_10", "rsi_14", "ret_lag1", "ret_lag2", "ret_lag5"]

# Display names used in the first article (Istikhor et al., 2026)
DISPLAY = {
    "ARFIMA": "ARFIMA", "LSTM": "LSTM", "XGB": "XGBoost",
    "Stage2_ARFIMA_LSTM": "Serial (ARFIMA+LSTM)", "Stage3_Serial_Final": "Serial (Final)",
    "Parallel_Equal": "Parallel Equal", "Parallel_InvError": "Parallel InvError",
    "Parallel_SoftMax_Grid": "Parallel SoftMax", "Stacking": "Stacking",
}
TABLE4_MODELS = ["XGBoost", "LSTM", "ARFIMA", "Serial (Final)", "Parallel Equal",
                 "Parallel InvError", "Parallel SoftMax", "Stacking"]
PRIMARY_PAIRS = [("LSTM", "Parallel InvError"), ("LSTM", "Stacking"),
                 ("Parallel InvError", "Stacking")]


# ---------------------------------------------------------------- paths and seeds
def paths(root):
    root = Path(root)
    inputs = root / "02_Inputs_v2_snapshot"
    return {
        "ROOT": root,
        "INPUTS": inputs,
        "SPLITS": inputs / "splits",
        "ARFIMA": inputs / "arfima_predictions",
        "COMBINED": inputs / "combined_test_forecasts",
        "OUT": root / "05_Outputs",
    }


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def check_input_snapshot(P):
    """First run: write a checksum manifest of the 21 snapshot files. Later runs: stop if any file changed."""
    files = sorted(p for p in P["INPUTS"].rglob("*.csv"))
    assert len(files) == 21, f"Expected 21 input files in the snapshot, found {len(files)}"
    current = {str(p.relative_to(P["INPUTS"])): _sha256(p) for p in files}
    manifest = P["INPUTS"] / "inputs_manifest.json"
    if not manifest.exists():
        with open(manifest, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
        return "manifest created"
    with open(manifest) as f:
        saved = json.load(f)
    changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
    assert not changed, f"Input snapshot changed since the manifest was written: {changed}"
    return "unchanged"


def xgb_params(seed):
    return dict(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                early_stopping_rounds=50, eval_metric="rmse")


def set_all_seeds(seed):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ---------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            "SMAPE": np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100,
            "R2": r2_score(y_true, y_pred)}


def mae_ratio_table(combined):
    """combined: {ticker: DataFrame with y_true, Persistence and model columns (display names)}."""
    rows = {}
    for t, df in combined.items():
        pers = mean_absolute_error(df["y_true"], df["Persistence"])
        rows[t] = {m: mean_absolute_error(df["y_true"], df[m]) / pers
                   for m in df.columns if m not in ("y_true", "Persistence")}
    out = pd.DataFrame(rows)
    out["Mean"] = out.mean(axis=1)
    return out


# ---------------------------------------------------------------- Diebold-Mariano (identical to v2)
def nw_lag(n):
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def dm_test(e_a, e_b, loss="squared", lag=None, h=1):
    e_a, e_b = np.asarray(e_a, float), np.asarray(e_b, float)
    d = e_a ** 2 - e_b ** 2 if loss == "squared" else np.abs(e_a) - np.abs(e_b)
    n = len(d)
    lag = nw_lag(n) if lag is None else lag
    dbar = d.mean()
    dc = d - dbar
    gamma0 = np.dot(dc, dc) / n
    if gamma0 == 0:
        return {"n": n, "lag": lag, "mean_d": dbar, "DM": np.nan, "p_DM": np.nan,
                "DM_HLN": np.nan, "p_HLN": np.nan, "note": "identical forecasts"}
    lr_var = gamma0 + 2 * sum((1 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n for k in range(1, lag + 1))
    if lr_var <= 0:
        return {"n": n, "lag": lag, "mean_d": dbar, "DM": np.nan, "p_DM": np.nan,
                "DM_HLN": np.nan, "p_HLN": np.nan, "note": "non-positive variance"}
    dm = dbar / np.sqrt(lr_var / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"n": n, "lag": lag, "mean_d": dbar, "DM": dm, "p_DM": 2 * stats.norm.sf(abs(dm)),
            "DM_HLN": hln, "p_HLN": 2 * stats.t.sf(abs(hln), df=n - 1), "note": ""}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def dm_set(combined, pairs, loss="squared"):
    rows = []
    for t, df in combined.items():
        for a, b in pairs:
            r = dm_test(df["y_true"] - df[a], df["y_true"] - df[b], loss=loss)
            better = "" if np.isnan(r["DM"]) else (b if r["mean_d"] > 0 else a)
            rows.append({"Ticker": t, "Model_A": a, "Model_B": b, "Loss": loss, **r, "Better": better})
    return pd.DataFrame(rows)


def primary_dm(combined, loss="squared"):
    res = dm_set(combined, PRIMARY_PAIRS, loss=loss)
    res["p_Holm"] = holm(res["p_HLN"].fillna(1.0))
    res["Sig_5%"] = res["p_HLN"] < 0.05
    res["Sig_Holm"] = res["p_Holm"] < 0.05
    return res


# ---------------------------------------------------------------- data (identical to v2)
def build_inputs(df):
    px = df[TARGET]
    X = pd.DataFrame(index=df.index)
    for c in RATIO_COLS:
        X[c + "_rel"] = df[c] / px - 1
    X["atr_14_rel"] = df["atr_14"] / px
    X["log_volume"] = np.log(df["Volume"].clip(lower=1))
    for c in KEEP_COLS:
        X[c] = df[c]
    return X


def load_data(P):
    data = {}
    for t in TICKERS:
        parts = {n: pd.read_csv(P["SPLITS"] / f"{t}_{n}.csv", index_col="Date", parse_dates=True)
                 for n in ["train", "val", "test"]}
        full = pd.concat([parts["train"], parts["val"], parts["test"]])
        X = build_inputs(full)
        price = full[TARGET]
        y_next = np.log(price.shift(-1) / price)
        target_date = pd.Series(full.index, index=full.index).shift(-1)
        split_of = pd.Series("train", index=full.index)
        split_of[full.index.isin(parts["val"].index)] = "val"
        split_of[full.index.isin(parts["test"].index)] = "test"
        target_split = target_date.map(split_of)
        data[t] = {"full": full, "X": X, "price": price, "y_next": y_next,
                   "target_date": target_date, "target_split": target_split, "split_of": split_of}
    return data


def load_arfima(P):
    """v2 ARFIMA one-step forecasts (deterministic; not retrained here)."""
    return {t: pd.concat([pd.read_csv(P["ARFIMA"] / f"{t}_{s}_predictions.csv", parse_dates=["Date"])
                          for s in ["train", "val", "test"]]).set_index("Date")
            for t in TICKERS}


# ---------------------------------------------------------------- base models (identical to v2)
def make_sequences(Xs, window):
    n = len(Xs)
    idx = np.arange(window - 1, n)
    seq = np.stack([Xs[i - window + 1:i + 1] for i in idx])
    return seq, idx


def build_lstm(n_features):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW, n_features)),
        tf.keras.layers.LSTM(LSTM_UNITS[0], return_sequences=True),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.LSTM(LSTM_UNITS[1]),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


def train_base(data, seed):
    import tensorflow as tf
    import xgboost as xgb
    tables, logs = {}, []
    for t in TICKERS:
        D = data[t]
        valid = D["y_next"].notna() & D["X"].notna().all(axis=1)

        # LSTM
        train_rows = (D["target_split"] == "train") & valid
        x_scaler = MinMaxScaler().fit(D["X"][train_rows])
        y_scaler = StandardScaler().fit(D["y_next"][train_rows].values.reshape(-1, 1))
        Xs = x_scaler.transform(D["X"].fillna(0))
        seq, idx = make_sequences(Xs, WINDOW)
        rows = D["X"].index[idx]
        ok = valid.loc[rows].values
        seq, rows = seq[ok], rows[ok]
        split = D["target_split"].loc[rows].values
        ys = y_scaler.transform(D["y_next"].loc[rows].values.reshape(-1, 1)).ravel()
        set_all_seeds(seed)
        model = build_lstm(seq.shape[2])
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
        hist = model.fit(seq[split == "train"], ys[split == "train"],
                         validation_data=(seq[split == "val"], ys[split == "val"]),
                         epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=False, callbacks=[es], verbose=0)
        lstm_ret = pd.Series(y_scaler.inverse_transform(model.predict(seq, verbose=0)).ravel(), index=rows)

        # XGBoost
        Xv, yv, sp = D["X"][valid], D["y_next"][valid], D["target_split"][valid]
        xm = xgb.XGBRegressor(**xgb_params(seed))
        xm.fit(Xv[sp == "train"], yv[sp == "train"], eval_set=[(Xv[sp == "val"], yv[sp == "val"])], verbose=False)
        xgb_ret = pd.Series(xm.predict(Xv), index=Xv.index)
        train_r2 = r2_score(yv[sp == "train"], xm.predict(Xv[sp == "train"]))

        # USD table (CELL 7)
        r = lstm_ret.index.intersection(xgb_ret.index)
        p_today = D["price"].loc[r]
        tables[t] = pd.DataFrame({
            "Date": D["target_date"].loc[r].values,
            "Split": D["target_split"].loc[r].values,
            "y_true": D["price"].shift(-1).loc[r].values,
            "Persistence": p_today.values,
            "LSTM": (p_today * np.exp(lstm_ret.loc[r])).values,
            "XGB": (p_today * np.exp(xgb_ret.loc[r])).values,
        })
        logs.append({"Ticker": t, "Seed": seed,
                     "LSTM_epochs_run": len(hist.history["loss"]),
                     "LSTM_best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
                     "XGB_best_iteration": int(xm.best_iteration),
                     "XGB_train_R2_returns": float(train_r2)})
    return tables, pd.DataFrame(logs)


# ---------------------------------------------------------------- serial (identical to v2)
def run_serial(data, arfima, seed):
    import tensorflow as tf
    import xgboost as xgb
    out, logs = {}, []
    for t in TICKERS:
        D = data[t]
        df = pd.DataFrame(index=D["full"].index)
        df["price"] = D["price"]
        df["arfima"] = arfima[t]["ARFIMA"].reindex(df.index)
        df["r1"] = np.log(df["price"]) - np.log(df["arfima"])
        df["split"] = D["split_of"]
        X = D["X"]

        # Stage 2: LSTM on 60-day sequences of past r1
        r1 = df["r1"]
        target = r1.shift(-1)
        tsplit = df["split"].shift(-1)
        fdate = pd.Series(df.index, index=df.index).shift(-1)
        scaler = StandardScaler().fit(r1[(df["split"] == "train") & r1.notna()].values.reshape(-1, 1))
        r1s = pd.Series(np.nan, index=df.index)
        r1s[r1.notna()] = scaler.transform(r1[r1.notna()].values.reshape(-1, 1)).ravel()
        vals = r1s.values
        rows = [i for i in range(len(df)) if i >= WINDOW - 1
                and not np.isnan(vals[i - WINDOW + 1:i + 1]).any() and not np.isnan(target.iloc[i])]
        seq = np.stack([vals[i - WINDOW + 1:i + 1] for i in rows])[..., None]
        ys = scaler.transform(target.iloc[rows].values.reshape(-1, 1)).ravel()
        sp = tsplit.iloc[rows].values
        set_all_seeds(seed)
        model = build_lstm(1)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
        h = model.fit(seq[sp == "train"], ys[sp == "train"], validation_data=(seq[sp == "val"], ys[sp == "val"]),
                      epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=False, callbacks=[es], verbose=0)
        pred = scaler.inverse_transform(model.predict(seq, verbose=0)).ravel()
        s2 = pd.Series(np.nan, index=df.index)
        s2[fdate.iloc[rows].values] = pred
        df["r1_hat"] = s2

        # Stage 3: XGBoost on Stage 2 residual
        df["stage2_log"] = np.log(df["arfima"]) + df["r1_hat"]
        df["r2"] = np.log(df["price"]) - df["stage2_log"]
        target = df["r2"].shift(-1)
        ok = target.notna() & X.notna().all(axis=1)
        Xo, yo, so = X[ok], target[ok], tsplit[ok]
        xm = xgb.XGBRegressor(**xgb_params(seed))
        xm.fit(Xo[so == "train"], yo[so == "train"], eval_set=[(Xo[so == "val"], yo[so == "val"])], verbose=False)
        s3 = pd.Series(np.nan, index=df.index)
        s3[fdate[ok].values] = xm.predict(Xo)
        df["r2_hat"] = s3

        res = pd.DataFrame({
            "Date": df.index, "Split": df["split"].values, "y_true": df["price"].values,
            "Persistence": df["price"].shift(1).values,
            "Stage1_ARFIMA": df["arfima"].values,
            "Stage2_ARFIMA_LSTM": np.exp(np.log(df["arfima"]) + df["r1_hat"]).values,
            "Stage3_Serial_Final": np.exp(np.log(df["arfima"]) + df["r1_hat"] + df["r2_hat"]).values,
        })
        out[t] = res
        logs.append({"Ticker": t, "Seed": seed, "Stage2_epochs_run": len(h.history["loss"]),
                     "Stage2_best_epoch": int(np.argmin(h.history["val_loss"])) + 1,
                     "Stage3_best_iteration": int(xm.best_iteration)})
    return out, pd.DataFrame(logs)


# ---------------------------------------------------------------- parallel and stacking (identical to v2)
def simplex_grid(n_steps):
    g = [(a, l, n_steps - a - l) for a in range(n_steps + 1) for l in range(n_steps + 1 - a)]
    return np.array(g, dtype=float) / n_steps


GRID = simplex_grid(int(round(1 / GRID_STEP)))


def align_three(base_tables, arfima):
    """Merge ARFIMA with LSTM/XGB on Date for val and test; stop on any actual/persistence mismatch."""
    out = {}
    for t in TICKERS:
        out[t] = {}
        a_all = arfima[t].reset_index()
        b_all = base_tables[t]
        for split in ["val", "test"]:
            a = a_all[a_all["Date"].isin(b_all.loc[b_all.Split == split, "Date"])]
            b = b_all[b_all.Split == split]
            m = a[["Date", "y_true", "Persistence", "ARFIMA"]].merge(
                b[["Date", "y_true", "Persistence", "LSTM", "XGB"]], on="Date", how="inner", suffixes=("", "_b"))
            assert float(np.max(np.abs(m["y_true"] - m["y_true_b"]))) < 1e-6, f"{t} {split}: actual mismatch"
            assert float(np.nanmax(np.abs(m["Persistence"] - m["Persistence_b"]))) < 1e-6, f"{t} {split}: persistence mismatch"
            out[t][split] = m.drop(columns=["y_true_b", "Persistence_b"]).dropna().sort_values("Date").reset_index(drop=True)
    return out


def run_parallel(aligned):
    weights = {}
    for t in TICKERS:
        v = aligned[t]["val"]
        P, y = v[MODELS3].values, v["y_true"].values
        val_mae = np.array([mean_absolute_error(y, v[m]) for m in MODELS3])
        inv = 1 / val_mae
        rmse = np.sqrt(((P @ GRID.T - y[:, None]) ** 2).mean(axis=0))
        weights[t] = {"Equal": np.full(3, 1 / 3), "InvError": inv / inv.sum(),
                      "SoftMax_Grid": GRID[int(np.argmin(rmse))]}
    preds = {t: {s: aligned[t][s][MODELS3].values for s in ["val", "test"]} for t in TICKERS}
    out = {t: {s: {f"Parallel_{k}": preds[t][s] @ w for k, w in weights[t].items()} for s in ["val", "test"]}
           for t in TICKERS}
    return out, weights


def run_stacking(aligned):
    out, coefs = {}, {}
    for t in TICKERS:
        v = aligned[t]["val"]
        meta = LinearRegression(fit_intercept=True, positive=True).fit(v[MODELS3].values, v["y_true"].values)
        coefs[t] = {"b0": float(meta.intercept_), **{f"b_{m}": float(meta.coef_[i]) for i, m in enumerate(MODELS3)}}
        out[t] = {s: meta.predict(aligned[t][s][MODELS3].values) for s in ["val", "test"]}
    return out, coefs


# ---------------------------------------------------------------- one full reference run
def run_reference(P, data, arfima, seed):
    """Retrain LSTM and XGBoost (base and serial) with `seed`; rebuild all architectures.
    Returns combined test tables (display names, same layout as the v2 combined tables), plus logs."""
    base, base_log = train_base(data, seed)
    serial, serial_log = run_serial(data, arfima, seed)
    aligned = align_three(base, arfima)
    par, weights = run_parallel(aligned)
    stk, coefs = run_stacking(aligned)
    combined = {}
    for t in TICKERS:
        te = aligned[t]["test"].copy()
        for k, v in par[t]["test"].items():
            te[k] = v
        te["Stacking"] = stk[t]["test"]
        s = serial[t][serial[t].Split == "test"][["Date", "Stage2_ARFIMA_LSTM", "Stage3_Serial_Final"]]
        te = te.merge(s, on="Date", how="left")
        te = te.rename(columns=DISPLAY).set_index("Date")
        cols = ["y_true", "Persistence", "ARFIMA", "LSTM", "XGBoost", "Serial (ARFIMA+LSTM)", "Serial (Final)",
                "Parallel Equal", "Parallel InvError", "Parallel SoftMax", "Stacking"]
        combined[t] = te[cols]
        assert len(combined[t]) == 501 and combined[t].notna().all().all(), f"{t}: expected 501 complete test rows"
    w_rows = [{"Ticker": t, "Seed": seed, "Scheme": k, **{f"w_{m}": float(w[i]) for i, m in enumerate(MODELS3)}}
              for t in TICKERS for k, w in weights[t].items()]
    c_rows = [{"Ticker": t, "Seed": seed, **coefs[t]} for t in TICKERS]
    logs = {"base": base_log, "serial": serial_log, "weights": pd.DataFrame(w_rows), "stacking": pd.DataFrame(c_rows)}
    return combined, logs
