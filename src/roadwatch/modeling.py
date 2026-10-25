"""Models, deterministic negative sampling, Platt calibration and the walk-forward runner.

Walk-forward protocol (one fit per retraining date r, a Monday):
  last matured origin m = r - 14 days        (label horizon m+7d plus the 2-day 311 lag ends by r)
  calibration/early-stopping holdout = the 13 matured weeks (m - 12w .. m)
  fit window = train_start <= origin < m - 12w (optionally excluding an origin range)
  predictions for origins in [r, next r)
Hence no prediction ever uses a label whose horizon had not ended (plus lag) before r, and calibration never sees
evaluation weeks. Test-period outcomes are never passed to model selection (see guard_selection_data).
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import CATEGORICAL

HOLDOUT_WEEKS = 13
TEST_START = dt.date(2025, 7, 7)


class SelectionLeakError(Exception):
    pass


def guard_selection_data(origins, test_start: dt.date = TEST_START) -> None:
    """Model selection may only see origins before the walk-forward test period."""
    o = pd.to_datetime(pd.Series(origins)).dt.date
    if len(o) and o.max() >= test_start:
        raise SelectionLeakError(f"selection data contains origin {o.max()} >= test start {test_start}")


def det_uniform(block_id: np.ndarray, origin: pd.Series) -> np.ndarray:
    """Deterministic pseudo-uniform [0,1) per (block, origin) - splitmix64 hash; identical across runs and variants."""
    days = (pd.to_datetime(pd.Series(origin)).values.astype("datetime64[D]").astype(np.int64)).astype(np.uint64)
    x = (np.asarray(block_id).astype(np.uint64) * np.uint64(1_000_003)) ^ days
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def negative_sample(y: np.ndarray, u: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep every positive and negatives with u < rate; weight negatives 1/rate so the weighted loss is unbiased."""
    keep = (y == 1) | (u < rate)
    w = np.where(y == 1, 1.0, 1.0 / rate)
    return keep, w


class Platt:
    """Monotone logistic recalibration on a transformed score; preserves ranking."""

    def __init__(self, transform: str = "logit"):
        self.transform, self.lr, self.const = transform, None, None

    def _z(self, s):
        s = np.asarray(s, float)
        if self.transform == "logit":
            s = np.clip(s, 1e-7, 1 - 1e-7)
            return np.log(s / (1 - s)).reshape(-1, 1)
        return np.log(np.clip(s, 0, None) + 1e-4).reshape(-1, 1)

    def fit(self, s, y):
        y = np.asarray(y)
        if np.unique(np.asarray(s)).size < 2 or np.unique(y).size < 2:
            self.const = float(y.mean())
            return self
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(s), y)
        return self

    def predict(self, s):
        if self.const is not None:
            return np.full(len(s), self.const)
        return self.lr.predict_proba(self._z(s))[:, 1]


# ---------------- scorers ----------------
@dataclass
class ModelSpec:
    name: str                 # B0_global | B1_hist_rate | B2_recency | LR | LGBM
    params: dict = field(default_factory=dict)


def baseline_score(spec: ModelSpec, X: pd.DataFrame, fit_prevalence: float) -> np.ndarray:
    if spec.name == "B0_global":
        return np.full(len(X), fit_prevalence)
    if spec.name == "B1_hist_rate":
        a = spec.params["prior_weeks"]
        n = X["obs_weeks_52"].to_numpy()
        k = np.nan_to_num(X["pos_week_rate_52"].to_numpy()) * n
        p0 = (pd.Series(k).groupby(X["origin"].to_numpy()).transform("sum") /
              pd.Series(n).groupby(X["origin"].to_numpy()).transform("sum")).to_numpy()
        return (k + a * np.nan_to_num(p0)) / (n + a) + 1e-7 * X["rep_frac_365d"].to_numpy()
    if spec.name == "B2_recency":
        return X[f"rep_decay_hl{spec.params['half_life']}d"].to_numpy() + 1e-9 * X["rep_frac_365d"].to_numpy()
    raise ValueError(spec.name)


def signed_log1p(a):
    """Module-level (picklable) variance-stabilising transform for count-like features."""
    return np.sign(a) * np.log1p(np.abs(a))


def _lr_pipeline(features: list[str], C: float) -> Pipeline:
    num = [f for f in features if f not in CATEGORICAL]
    cat = [f for f in features if f in CATEGORICAL]
    signed_log = FunctionTransformer(signed_log1p, feature_names_out="one-to-one")
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("log", signed_log),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs"))])


@dataclass
class FitResult:
    model: object
    calibrator: Platt
    info: dict


def fit_model(spec: ModelSpec, features: list[str], fit: pd.DataFrame, hold: pd.DataFrame, ycol: str) -> FitResult:
    t0 = time.time()
    y, w = fit[ycol].to_numpy().astype(int), fit["_w"].to_numpy()
    prev = float(np.average(y, weights=w))
    info = {"fit_rows": len(fit), "fit_positives": int(y.sum()), "fit_weighted_prevalence": prev,
            "hold_rows": len(hold), "hold_positives": int(hold[ycol].sum())}
    if spec.name.startswith("B"):
        raw_h = baseline_score(spec, hold, prev)
        cal = Platt("log" if spec.name != "B0_global" else "logit").fit(raw_h, hold[ycol].to_numpy().astype(int))
        info["seconds"] = round(time.time() - t0, 2)
        return FitResult({"prevalence": prev}, cal, info)
    if spec.name == "LR":
        m = _lr_pipeline(features, spec.params["C"])
        m.fit(fit[features], y, lr__sample_weight=w)
        raw_h = m.predict_proba(hold[features])[:, 1]
    elif spec.name == "LGBM":
        p = {"objective": "binary", "learning_rate": 0.05, "num_leaves": spec.params["num_leaves"],
             "min_data_in_leaf": spec.params["min_data_in_leaf"], "feature_fraction": 0.8, "bagging_fraction": 0.8,
             "bagging_freq": 1, "lambda_l2": 1.0, "metric": "average_precision", "verbose": -1, "seed": 7,
             "deterministic": True, "force_row_wise": True, "num_threads": 8}
        cats = [f for f in features if f in CATEGORICAL]
        dtr = lgb.Dataset(fit[features], y, weight=w, categorical_feature=cats, free_raw_data=True)
        dho = lgb.Dataset(hold[features], hold[ycol].to_numpy().astype(int), categorical_feature=cats, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=spec.params.get("max_rounds", 1500), valid_sets=[dho],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        info["best_iteration"] = int(m.best_iteration)
        raw_h = m.predict(hold[features], num_iteration=m.best_iteration)
    else:
        raise ValueError(spec.name)
    cal = Platt("logit").fit(raw_h, hold[ycol].to_numpy().astype(int))
    info["seconds"] = round(time.time() - t0, 2)
    return FitResult(m, cal, info)


def predict_raw(spec: ModelSpec, fr: FitResult, X: pd.DataFrame, features: list[str]) -> np.ndarray:
    if spec.name.startswith("B"):
        return baseline_score(spec, X, fr.model["prevalence"])
    if spec.name == "LR":
        return fr.model.predict_proba(X[features])[:, 1]
    return fr.model.predict(X[features], num_iteration=fr.model.best_iteration)


def retrain_schedule(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """First Monday of each month from start; each retrain predicts origins in [r, next r) up to end (inclusive)."""
    rs, d = [], start
    while d <= end:
        rs.append(d)
        y, m = (d.year + (d.month == 12), d.month % 12 + 1)
        nxt = dt.date(y, m, 1)
        d = nxt + dt.timedelta(days=(7 - nxt.weekday()) % 7)
    return [(r, (rs[i + 1] - dt.timedelta(days=1)) if i + 1 < len(rs) else end) for i, r in enumerate(rs)]


def windows_for(r: dt.date) -> dict:
    m = r - dt.timedelta(days=14)
    hold_from = m - dt.timedelta(weeks=HOLDOUT_WEEKS - 1)
    return {"retrain_date": r, "last_matured_origin": m, "holdout_from": hold_from, "fit_before": hold_from}
