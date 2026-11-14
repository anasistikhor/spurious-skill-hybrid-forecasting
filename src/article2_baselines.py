"""
article2_baselines.py
Part of the code for the fault-injection study (see README).

The seven literature baselines, rebuilt line for line from the v2 baselines (run 8 Oct 2026)
with the seed as an argument, and the F8 version (non-identical baselines),
as frozen in the study protocol, Section 3.

Reference (identical conditions, v2): train 2015-2020, early stopping on 2021-2022, one-step
forecasts on the 501 test days, the 19 leak-free inputs known at the close of day t.

F8 (the v1 route): three changes together, as Section 3 states.
  1. Training window 2015-2022. Early stopping uses the last 10% of that window (chronological).
  2. Recursive 501-step forecasts from the last day of 2022: ARIMA and ARFIMA forecast the whole
     test path at once; residual models feed their own forecasts back in; return models build the
     price path by compounding their predicted returns from the last 2022 price. No model is
     re-anchored to an actual test-period price.
  3. Same-day indicators: the inputs used to forecast day t+1 are the indicators of day t+1,
     which contain the price being forecast. This applies to the models that use the 19 inputs
     (XGBoost, Attention-CNN-LSTM, XGBoost -> LSTM and the XGBoost/attention parts of AttCLX).
The evaluation days stay the same 501 test days.
"""

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import article2_pipeline as ap

BASELINES = ["ARIMA", "ARIMA -> LSTM", "ARIMA -> XGBoost", "XGBoost -> LSTM", "ARFIMA -> LSTM",
             "Attention-CNN-LSTM", "AttCLX"]
P5_LSTM_EPOCHS, P5_ATT_EPOCHS = 50, 100
ARIMA_P, ARIMA_D, ARIMA_Q = range(6), [0, 1], range(6)
BURN_IN = 50
F8_STOP_SHARE = 0.10          # F8: early stopping on the last 10% of the 2015-2022 window

# v2 test MAE (USD) from the v2 baselines, seed 42 (reproduction check for seed S0)
V2_BASELINE_MAE = pd.DataFrame({
    "JPM": [1.5971, 1.5808, 1.5870, 1.5532, 1.6100, 1.5655, 1.5538],
    "NEE": [0.7974, 0.8042, 0.7970, 0.7782, 0.7809, 0.7754, 0.7782],
    "XOM": [1.1215, 1.1225, 1.1223, 1.0949, 1.1281, 1.0880, 1.0988]}, index=BASELINES)


def p5_xgb(seed):
    return dict(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=seed,
                early_stopping_rounds=50, eval_metric="rmse")


# ---------------------------------------------------------------- shared pieces (identical to v2)
def next_day(series, full_index):
    pos = full_index.get_indexer(series.index) + 1
    keep = pos < len(full_index)
    return pd.Series(series.values[keep], index=full_index[pos[keep]])


def fit_net(model, Xtr, ytr, Xva, yva, epochs):
    import tensorflow as tf
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=ap.PATIENCE, restore_best_weights=True)
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=ap.BATCH_SIZE,
                  shuffle=False, callbacks=[es], verbose=0)
    return {"Epochs_run": len(h.history["loss"]), "Best_epoch": int(np.argmin(h.history["val_loss"])) + 1}


def p5_lstm(window, n_features):
    import tensorflow as tf
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(window, n_features)),
                             tf.keras.layers.LSTM(64), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    return m


def att_cnn_lstm(window, n_features):
    import tensorflow as tf

    class AttentionPooling(tf.keras.layers.Layer):
        def build(self, input_shape):
            self.W = self.add_weight(name="att_w", shape=(input_shape[-1], 1), initializer="random_normal")
            self.b = self.add_weight(name="att_b", shape=(input_shape[1], 1), initializer="zeros")

        def call(self, x):
            e = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
            a = tf.nn.softmax(e, axis=1)
            return tf.reduce_sum(x * a, axis=1)

    inp = tf.keras.layers.Input(shape=(window, n_features))
    x = tf.keras.layers.Conv1D(64, 3, activation="relu")(inp)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = AttentionPooling()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    m = tf.keras.Model(inp, tf.keras.layers.Dense(1)(x))
    m.compile(optimizer="adam", loss="mse")
    return m


def baseline_data(data, arfima):
    """v2 layout from the study inputs."""
    out = {}
    for t in ap.TICKERS:
        D = data[t]
        out[t] = {"price": D["price"], "logp": np.log(D["price"]), "X": D["X"], "split": D["split_of"],
                  "arfima_log": np.log(arfima[t]["ARFIMA"].reindex(D["full"].index))}
    return out


# ---------------------------------------------------------------- ARIMA (seed independent; cached by the notebook)
def arima_grid(logp_train):
    from statsmodels.tsa.arima.model import ARIMA
    log, best, order = [], None, None
    for p, d, q in itertools.product(ARIMA_P, ARIMA_D, ARIMA_Q):
        try:
            res = ARIMA(logp_train, order=(p, d, q)).fit()
            log.append({"p": p, "d": d, "q": q, "AIC": res.aic, "Status": "ok"})
            if best is None or res.aic < best.aic:
                best, order = res, (p, d, q)
        except Exception as e:
            log.append({"p": p, "d": d, "q": q, "AIC": np.nan, "Status": str(e)[:60]})
    return best, order, pd.DataFrame(log)


def arima_reference(bd):
    """v2 CELL 4: estimate on 2015-2020, one-step forecasts with frozen parameters, 50-day burn-in."""
    out, logs = {}, []
    for t in ap.TICKERS:
        D = bd[t]
        t0 = time.time()
        best, order, g = arima_grid(D["logp"][D["split"] == "train"].values)
        f = pd.Series(best.apply(D["logp"].values).predict(), index=D["logp"].index)
        f.iloc[:BURN_IN] = np.nan
        out[t] = f
        logs.append({"Ticker": t, "Route": "reference", "Order": str(order), "AIC": float(best.aic),
                     "Failed_fits": int((g.Status != "ok").sum()), "Seconds": round(time.time() - t0, 1)})
    return out, pd.DataFrame(logs)


def arima_f8(bd):
    """F8: estimate on 2015-2022; in-sample one-step values for training; one static 501-step path for test."""
    out, logs = {}, []
    for t in ap.TICKERS:
        D = bd[t]
        fit_rows = D["split"].isin(["train", "val"]).values
        t0 = time.time()
        best, order, g = arima_grid(D["logp"][fit_rows].values)
        f = pd.Series(np.nan, index=D["logp"].index)
        f[fit_rows] = best.predict()
        f.iloc[:BURN_IN] = np.nan
        f[D["split"] == "test"] = np.asarray(best.forecast(steps=int((D["split"] == "test").sum())))
        out[t] = f
        logs.append({"Ticker": t, "Route": "F8", "Order": str(order), "AIC": float(best.aic),
                     "Failed_fits": int((g.Status != "ok").sum()), "Seconds": round(time.time() - t0, 1)})
    return out, pd.DataFrame(logs)


def arfima_f8(data):
    """F8 for ARFIMA -> LSTM: the v2 ARFIMA grid re-estimated on 2015-2022; in-sample one-step values for
    training; one static 501-step path for test (same functions as article2_faults)."""
    import article2_faults as af
    out, logs = {}, []
    for t in ap.TICKERS:
        D = data[t]
        n_fit = int(D["split_of"].isin(["train", "val"]).sum())
        logp = np.log(D["price"].values)
        mean = logp[:n_fit].mean()
        x = logp - mean
        best, grid, _ = af.arfima_grid_search(x, n_fit)
        x_hat = af.arfima_one_step(x, best)
        test = (D["split_of"] == "test").values
        x_hat[test] = af.arfima_static(x, best, n_fit, int(test.sum()))
        f = pd.Series(x_hat + mean, index=D["full"].index)
        f.iloc[:af.BURN_IN] = np.nan
        out[t] = f
        logs.append({"Ticker": t, "Route": "F8", "p": best["p"], "d": best["d"], "q": best["q"], "AIC": float(best["AIC"])})
    return out, pd.DataFrame(logs)


# ---------------------------------------------------------------- reference baselines (v2)
def residual_lstm(D, base_log, seed, label, log):
    import tensorflow as tf
    r = D["logp"] - base_log
    target, tsplit = r.shift(-1), D["split"].shift(-1)
    fdate = pd.Series(r.index, index=r.index).shift(-1)
    sc = StandardScaler().fit(r[(D["split"] == "train") & r.notna()].values.reshape(-1, 1))
    rs = pd.Series(np.nan, index=r.index)
    rs[r.notna()] = sc.transform(r[r.notna()].values.reshape(-1, 1)).ravel()
    v = rs.values
    rows = [i for i in range(ap.WINDOW - 1, len(v)) if not np.isnan(v[i - ap.WINDOW + 1:i + 1]).any()
            and not np.isnan(target.iloc[i])]
    seq = np.stack([v[i - ap.WINDOW + 1:i + 1] for i in rows])[..., None]
    ys = sc.transform(target.iloc[rows].values.reshape(-1, 1)).ravel()
    sp = tsplit.iloc[rows].values
    tf.keras.utils.set_random_seed(seed)
    m = p5_lstm(ap.WINDOW, 1)
    info = fit_net(m, seq[sp == "train"], ys[sp == "train"], seq[sp == "val"], ys[sp == "val"], P5_LSTM_EPOCHS)
    out = pd.Series(sc.inverse_transform(m.predict(seq, verbose=0)).ravel(), index=fdate.iloc[rows].values)
    log.append({"Model": label, "Train_samples": int((sp == "train").sum()), **info})
    return out.reindex(r.index)


def residual_xgb(D, base_log, seed, label, log, n_lags=5):
    import xgboost as xgb
    r = D["logp"] - base_log
    feats = pd.concat({f"lag_{k}": r.shift(k) for k in range(n_lags)}, axis=1)
    target, tsplit = r.shift(-1), D["split"].shift(-1)
    fdate = pd.Series(r.index, index=r.index).shift(-1)
    ok = feats.notna().all(axis=1) & target.notna()
    F, y, s = feats[ok], target[ok], tsplit[ok]
    m = xgb.XGBRegressor(**p5_xgb(seed))
    m.fit(F[s == "train"], y[s == "train"], eval_set=[(F[s == "val"], y[s == "val"])], verbose=False)
    out = pd.Series(m.predict(F), index=fdate[ok].values)
    log.append({"Model": label, "Train_samples": int((s == "train").sum()), "Best_iteration": int(m.best_iteration)})
    return out.reindex(r.index)


def return_models(D, seed, log):
    """v2 CELL 5: XGBoost (P5 spec) and Attention-CNN-LSTM on the 19 inputs, one-step, log-price forecasts."""
    import tensorflow as tf
    import xgboost as xgb
    y_next = np.log(D["price"].shift(-1) / D["price"])
    tsplit = D["split"].shift(-1)
    valid = y_next.notna() & D["X"].notna().all(axis=1)
    Xv, yv, sv = D["X"][valid], y_next[valid], tsplit[valid]
    m = xgb.XGBRegressor(**p5_xgb(seed))
    m.fit(Xv[sv == "train"], yv[sv == "train"], eval_set=[(Xv[sv == "val"], yv[sv == "val"])], verbose=False)
    r_hat = pd.Series(m.predict(Xv), index=Xv.index)
    xgb_log = next_day(D["logp"].loc[r_hat.index] + r_hat, D["logp"].index).reindex(D["logp"].index)
    log.append({"Model": "XGBoost (P5 spec)", "Train_samples": int((sv == "train").sum()), "Best_iteration": int(m.best_iteration)})

    train_rows = (tsplit == "train") & valid
    xs = MinMaxScaler().fit(D["X"][train_rows])
    ysc = StandardScaler().fit(y_next[train_rows].values.reshape(-1, 1))
    Xs = xs.transform(D["X"].fillna(0))
    idx = np.arange(ap.WINDOW - 1, len(Xs))
    seq = np.stack([Xs[i - ap.WINDOW + 1:i + 1] for i in idx])
    rows = D["X"].index[idx]
    ok = valid.loc[rows].values
    seq, rows = seq[ok], rows[ok]
    sp = tsplit.loc[rows].values
    ys = ysc.transform(y_next.loc[rows].values.reshape(-1, 1)).ravel()
    tf.keras.utils.set_random_seed(seed)
    net = att_cnn_lstm(ap.WINDOW, seq.shape[2])
    info = fit_net(net, seq[sp == "train"], ys[sp == "train"], seq[sp == "val"], ys[sp == "val"], P5_ATT_EPOCHS)
    r_att = pd.Series(ysc.inverse_transform(net.predict(seq, verbose=0)).ravel(), index=rows)
    att_log = next_day(D["logp"].loc[rows] + r_att, D["logp"].index).reindex(D["logp"].index)
    log.append({"Model": "Attention-CNN-LSTM", "Train_samples": int((sp == "train").sum()), **info})
    return xgb_log, att_log


def assemble_table(D, logs_dict):
    """v2 CELL 7: USD table with y_true, Persistence, Split and the seven baselines."""
    tab = pd.DataFrame({"y_true": D["price"], "Persistence": D["price"].shift(1), "Split": D["split"]})
    for name in BASELINES[:-1]:
        tab[name] = np.exp(logs_dict[name])
    tab["AttCLX"] = (np.exp(logs_dict["ARIMA"]) + np.exp(logs_dict["Attention-CNN-LSTM"]) + np.exp(logs_dict["_xgb"])) / 3
    return tab


def run_reference_baselines(bd, arima_ref, seed):
    tables, log = {}, []
    for t in ap.TICKERS:
        D, L = bd[t], []
        a = arima_ref[t]
        xgb_log, att_log = return_models(D, seed, L)
        logs = {"ARIMA": a, "Attention-CNN-LSTM": att_log, "_xgb": xgb_log,
                "ARIMA -> LSTM": a + residual_lstm(D, a, seed, "ARIMA -> LSTM (residual LSTM)", L),
                "ARIMA -> XGBoost": a + residual_xgb(D, a, seed, "ARIMA -> XGBoost (residual XGB)", L),
                "XGBoost -> LSTM": xgb_log + residual_lstm(D, xgb_log, seed, "XGBoost -> LSTM (residual LSTM)", L),
                "ARFIMA -> LSTM": D["arfima_log"] + residual_lstm(D, D["arfima_log"], seed, "ARFIMA -> LSTM (residual LSTM)", L)}
        tables[t] = assemble_table(D, logs)
        log += [{"Ticker": t, "Seed": seed, "Route": "reference", **r} for r in L]
    return tables, pd.DataFrame(log)


# ---------------------------------------------------------------- F8 baselines
def _fit_stop_split(n_rows):
    """Chronological split of the 2015-2022 training rows: last 10% for early stopping."""
    cut = int(round(n_rows * (1 - F8_STOP_SHARE)))
    fit = np.zeros(n_rows, dtype=bool)
    fit[:cut] = True
    return fit, ~fit


def _recursive_predict(step_fn, history, steps):
    """Feed each forecast back as the newest input. history: the last values known at the end of 2022."""
    h = list(history)
    out = []
    for _ in range(steps):
        nxt = float(step_fn(np.asarray(h)))
        out.append(nxt)
        h.append(nxt)
    return np.array(out)


def residual_lstm_f8(D, base_log, seed, label, log):
    import tensorflow as tf
    r = D["logp"] - base_log
    target = r.shift(-1)
    tsplit = D["split"].shift(-1)
    in_window = tsplit.isin(["train", "val"])
    sc = StandardScaler().fit(r[D["split"].isin(["train", "val"]) & r.notna()].values.reshape(-1, 1))
    rs = pd.Series(np.nan, index=r.index)
    rs[r.notna()] = sc.transform(r[r.notna()].values.reshape(-1, 1)).ravel()
    v = rs.values
    rows = [i for i in range(ap.WINDOW - 1, len(v)) if in_window.iloc[i]
            and not np.isnan(v[i - ap.WINDOW + 1:i + 1]).any() and not np.isnan(target.iloc[i])]
    seq = np.stack([v[i - ap.WINDOW + 1:i + 1] for i in rows])[..., None]
    ys = sc.transform(target.iloc[rows].values.reshape(-1, 1)).ravel()
    fit, stop = _fit_stop_split(len(rows))
    tf.keras.utils.set_random_seed(seed)
    m = p5_lstm(ap.WINDOW, 1)
    info = fit_net(m, seq[fit], ys[fit], seq[stop], ys[stop], P5_LSTM_EPOCHS)
    test = (D["split"] == "test").values
    last_train = int(np.where(~test)[0][-1])
    hist = v[last_train - ap.WINDOW + 1:last_train + 1]
    infer = tf.function(lambda x: m(x, training=False))          # compiled once; same weights, much faster
    step = lambda h: float(infer(tf.constant(h[-ap.WINDOW:].reshape(1, ap.WINDOW, 1), dtype=tf.float32))[0, 0])
    path = sc.inverse_transform(_recursive_predict(step, hist, int(test.sum())).reshape(-1, 1)).ravel()
    out = pd.Series(np.nan, index=r.index)
    out[test] = path
    log.append({"Model": label, "Train_samples": int(fit.sum()), **info})
    return out


def residual_xgb_f8(D, base_log, seed, label, log, n_lags=5):
    import xgboost as xgb
    r = D["logp"] - base_log
    feats = pd.concat({f"lag_{k}": r.shift(k) for k in range(n_lags)}, axis=1)
    target, tsplit = r.shift(-1), D["split"].shift(-1)
    ok = feats.notna().all(axis=1) & target.notna() & tsplit.isin(["train", "val"])
    F, y = feats[ok], target[ok]
    fit, stop = _fit_stop_split(len(F))
    m = xgb.XGBRegressor(**p5_xgb(seed))
    m.fit(F[fit], y[fit], eval_set=[(F[stop], y[stop])], verbose=False)
    test = (D["split"] == "test").values
    last_train = int(np.where(~test)[0][-1])
    hist = r.values[last_train - n_lags + 1:last_train + 1]
    cols = list(feats.columns)
    step = lambda h: m.predict(pd.DataFrame([h[-1:-n_lags - 1:-1]], columns=cols))[0]
    out = pd.Series(np.nan, index=r.index)
    out[test] = _recursive_predict(step, hist, int(test.sum()))
    log.append({"Model": label, "Train_samples": int(fit.sum()), "Best_iteration": int(m.best_iteration)})
    return out


def return_models_f8(D, seed, log):
    """XGBoost (P5 spec) and Attention-CNN-LSTM trained on 2015-2022 with same-day inputs (row t holds the
    indicators of day t+1). Training rows get one-step values; the test path compounds predicted returns
    from the last 2022 price."""
    import tensorflow as tf
    import xgboost as xgb
    y_next = np.log(D["price"].shift(-1) / D["price"])
    tsplit = D["split"].shift(-1)
    X_same = D["X"].shift(-1)                                   # the leak: inputs of the day being forecast
    valid = y_next.notna() & X_same.notna().all(axis=1)
    in_window = tsplit.isin(["train", "val"])
    test = (D["split"] == "test").values
    last_train = int(np.where(~test)[0][-1])
    p0 = D["logp"].iloc[last_train]

    def to_log(r_hat):
        """In-sample: one-step log forecasts. Test: compounded path from the last 2022 price."""
        f = next_day(D["logp"].loc[r_hat.index] + r_hat, D["logp"].index).reindex(D["logp"].index)
        origins = D["logp"].index[last_train:len(test) - 1]
        f[test] = p0 + np.cumsum(r_hat.reindex(origins).values)
        return f

    Xv, yv = X_same[valid & in_window], y_next[valid & in_window]
    fit, stop = _fit_stop_split(len(Xv))
    m = xgb.XGBRegressor(**p5_xgb(seed))
    m.fit(Xv[fit], yv[fit], eval_set=[(Xv[stop], yv[stop])], verbose=False)
    r_hat = pd.Series(m.predict(X_same[valid]), index=X_same[valid].index)
    xgb_log = to_log(r_hat)
    log.append({"Model": "XGBoost (P5 spec)", "Train_samples": int(fit.sum()), "Best_iteration": int(m.best_iteration)})

    train_rows = in_window & valid
    xs = MinMaxScaler().fit(X_same[train_rows])
    ysc = StandardScaler().fit(y_next[train_rows].values.reshape(-1, 1))
    Xs = xs.transform(X_same.fillna(0))
    idx = np.arange(ap.WINDOW - 1, len(Xs))
    seq = np.stack([Xs[i - ap.WINDOW + 1:i + 1] for i in idx])
    rows = D["X"].index[idx]
    ok = valid.loc[rows].values
    seq, rows = seq[ok], rows[ok]
    inw = in_window.loc[rows].values
    ys = ysc.transform(y_next.loc[rows].values.reshape(-1, 1)).ravel()
    fit, stop = _fit_stop_split(int(inw.sum()))
    tf.keras.utils.set_random_seed(seed)
    net = att_cnn_lstm(ap.WINDOW, seq.shape[2])
    info = fit_net(net, seq[inw][fit], ys[inw][fit], seq[inw][stop], ys[inw][stop], P5_ATT_EPOCHS)
    r_att = pd.Series(ysc.inverse_transform(net.predict(seq, verbose=0)).ravel(), index=rows)
    att_log = to_log(r_att)
    log.append({"Model": "Attention-CNN-LSTM", "Train_samples": int(fit.sum()), **info})
    return xgb_log, att_log


def run_f8_baselines(bd, arima8, arfima8, seed):
    tables, log = {}, []
    for t in ap.TICKERS:
        D, L = bd[t], []
        a = arima8[t]
        xgb_log, att_log = return_models_f8(D, seed, L)
        logs = {"ARIMA": a, "Attention-CNN-LSTM": att_log, "_xgb": xgb_log,
                "ARIMA -> LSTM": a + residual_lstm_f8(D, a, seed, "ARIMA -> LSTM (residual LSTM)", L),
                "ARIMA -> XGBoost": a + residual_xgb_f8(D, a, seed, "ARIMA -> XGBoost (residual XGB)", L),
                "XGBoost -> LSTM": xgb_log + residual_lstm_f8(D, xgb_log, seed, "XGBoost -> LSTM (residual LSTM)", L),
                "ARFIMA -> LSTM": arfima8[t] + residual_lstm_f8(D, arfima8[t], seed, "ARFIMA -> LSTM (residual LSTM)", L)}
        tables[t] = assemble_table(D, logs)
        log += [{"Ticker": t, "Seed": seed, "Route": "F8", **r} for r in L]
    return tables, pd.DataFrame(log)


# ---------------------------------------------------------------- scoring and ranking
def baseline_metrics(tables, label, seed):
    rows = []
    for t, tab in tables.items():
        te = tab[tab.Split == "test"]
        assert len(te) == 501 and te[BASELINES].notna().all().all(), f"{t} {label}: expected 501 complete test rows"
        pers = ap.mean_absolute_error(te["y_true"], te["Persistence"])
        for m in BASELINES:
            mae = ap.mean_absolute_error(te["y_true"], te[m])
            rows.append({"Run": label, "Seed": seed, "Ticker": t, "Model": m, "Type": "Baseline",
                         "MAE": mae, "MAE_ratio": mae / pers})
    return pd.DataFrame(rows)


def research_metrics(combined, seed):
    rows = []
    for t, df in combined.items():
        pers = ap.mean_absolute_error(df["y_true"], df["Persistence"])
        for m in ["ARFIMA", "Serial (Final)", "Parallel Equal", "Parallel InvError", "Parallel SoftMax", "Stacking"]:
            mae = ap.mean_absolute_error(df["y_true"], df[m])
            rows.append({"Run": "research", "Seed": seed, "Ticker": t, "Model": m, "Type": "This research",
                         "MAE": mae, "MAE_ratio": mae / pers})
    return pd.DataFrame(rows)
