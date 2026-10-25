"""Separate weekly Brooklyn pothole-REPORT volume forecast (never a count of physical defects or repairs).

Target y_t: unique Brooklyn Street Condition/Pothole reports (with Brooklyn evidence) created in [t, t+7) local
(Stage 3 label_weeks.located_brooklyn_reports). Weeks whose labels are not 'observable' (unlocated-report weeks,
incomplete final week) are excluded as training targets and from evaluation.

As-of features at origin t (Monday 00:00):
  recent    reports created in [c-7d, c), [c-14d, c-7d), [c-28d, c) with c = t - 2 days (311 lag)
  seasonal  same-week-last-year volume [t-364d, t-357d) and the mean of the 3 weeks around it; week-of-year sin/cos
  weather   NOAA observations dated <= t - 5 days: freeze-thaw-day proxy (14/30 d), freeze days (30 d), precipitation
            (14/30 d), snow (30 d), mean TMIN (7 d)
Models: Poisson GLM (log link) on standardised log1p features; baselines: last-7-days naive, seasonal naive, 4-week mean.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = {
    "recent": ["log_r7", "log_r7_prev", "log_r28"],
    "recent_seasonal": ["log_r7", "log_r7_prev", "log_r28", "log_ly", "log_ly3", "woy_sin", "woy_cos"],
    "recent_seasonal_weather": ["log_r7", "log_r7_prev", "log_r28", "log_ly", "log_ly3", "woy_sin", "woy_cos",
                                "ft_14", "ft_30", "fz_30", "prcp_14", "prcp_30", "snow_30", "tmin_7"],
}
BASELINES = ["naive_last7", "seasonal_naive", "mean_4wk"]


def build_frame(report_days: pd.Series, weeks: pd.DataFrame, noaa: pd.DataFrame) -> pd.DataFrame:
    """report_days: local creation dates (one per Brooklyn report); weeks: Stage 3 label_weeks."""
    d = np.sort(pd.to_datetime(report_days).values.astype("datetime64[D]"))

    def count(a, b):  # reports with a <= date < b
        return int(np.searchsorted(d, np.datetime64(b), "left") - np.searchsorted(d, np.datetime64(a), "left"))

    nd = noaa.assign(day=pd.to_datetime(noaa["date"]).dt.date).set_index("day").sort_index()
    rows = []
    for _, w in weeks.sort_values("origin").iterrows():
        t = pd.Timestamp(w["origin"]).date()
        c = t - dt.timedelta(days=2)
        ly = t - dt.timedelta(days=364)
        r = {"origin": t, "y": int(w["located_brooklyn_reports"]), "week_status": str(w["week_status"]),
             "r7": count(c - dt.timedelta(days=7), c), "r7_prev": count(c - dt.timedelta(days=14), c - dt.timedelta(days=7)),
             "r28": count(c - dt.timedelta(days=28), c), "ly": count(ly, ly + dt.timedelta(days=7)),
             "ly3": np.mean([count(ly + dt.timedelta(days=7 * j), ly + dt.timedelta(days=7 * j + 7)) for j in (-1, 0, 1)])}
        woy = t.isocalendar()[1]
        r["woy_sin"], r["woy_cos"] = np.sin(2 * np.pi * woy / 52.18), np.cos(2 * np.pi * woy / 52.18)
        last = t - dt.timedelta(days=5)
        wx = nd.loc[:last]

        def win(days):
            return wx.loc[last - dt.timedelta(days=days - 1):]
        w14, w30, w7 = win(14), win(30), win(7)
        r["ft_14"] = float(((w14["tmin_value"] < 0) & (w14["tmax_value"] > 0)).sum())
        r["ft_30"] = float(((w30["tmin_value"] < 0) & (w30["tmax_value"] > 0)).sum())
        r["fz_30"] = float((w30["tmin_value"] < 0).sum())
        r["prcp_14"], r["prcp_30"] = float(w14["prcp_value"].sum()), float(w30["prcp_value"].sum())
        r["snow_30"], r["tmin_7"] = float(w30["snow_value"].sum()), float(w7["tmin_value"].mean())
        r["weather_last_day"] = str(wx.index.max()) if len(wx) else None
        rows.append(r)
    f = pd.DataFrame(rows)
    for c in ("r7", "r7_prev", "r28", "ly", "ly3"):
        f[f"log_{c}"] = np.log1p(f[c])
    return f


def baseline_forecast(f: pd.DataFrame, name: str) -> np.ndarray:
    if name == "naive_last7":
        return f["r7"].to_numpy(float)
    if name == "seasonal_naive":
        return f["ly"].to_numpy(float)
    if name == "mean_4wk":
        return f["r28"].to_numpy(float) / 4
    raise ValueError(name)


def fit_glm(train: pd.DataFrame, feats: list[str], alpha: float) -> Pipeline:
    m = Pipeline([("sc", StandardScaler()), ("glm", PoissonRegressor(alpha=alpha, max_iter=3000))])
    return m.fit(train[feats], train["y"])


def interval(model, train: pd.DataFrame, feats: list[str], q: tuple[float, float], last_n: int) -> tuple[float, float]:
    """Empirical multiplicative interval from the most recent in-sample residual ratios (y+1)/(mu+1)."""
    tr = train.tail(last_n)
    ratio = (tr["y"] + 1) / (model.predict(tr[feats]) + 1)
    return float(np.quantile(ratio, q[0])), float(np.quantile(ratio, q[1]))


def errors(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    return {"weeks": int(len(y)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "mape_pct": float((np.abs(e) / np.maximum(y, 1)).mean() * 100), "bias_mean_pred_minus_actual": float(e.mean()),
            "mean_actual": float(y.mean())}
