"""Evaluation of block-week predictions.

Inputs: predictions with block_id, origin, score (raw ranking score), prob (calibrated), label (1/0/NaN=unknown).
Unknown labels are excluded from every metric denominator; they are never treated as negatives.

Capacity scenarios (experimental, not NYC budgets): each week the top ceil(share * eligible blocks) blocks by score are
'inspected'. Ties are broken by a deterministic per-(block, origin) hash, never by row order.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve  # noqa: E402

from .modeling import det_uniform

CAPACITIES = (0.005, 0.01, 0.02)


def topk_select(pred: pd.DataFrame, share: float, n_blocks: int) -> pd.DataFrame:
    k = math.ceil(share * n_blocks)
    p = pred.assign(_tb=det_uniform(pred["block_id"].to_numpy(), pred["origin"]))
    p = p.sort_values(["origin", "score", "_tb"], ascending=[True, False, True], kind="stable")
    return p.groupby("origin", sort=False).head(k)


def capacity_metrics(pred: pd.DataFrame, reports: pd.DataFrame | None, n_blocks: int, shares=CAPACITIES) -> dict:
    lab = pred[pred["label"].notna()]
    pos_total = int((lab["label"] == 1).sum())
    out = {}
    rep_tot = reports.groupby("origin")["report_id"].nunique().sum() if reports is not None and len(reports) else 0
    for s in shares:
        sel = topk_select(pred, s, n_blocks)
        sl = sel[sel["label"].notna()]
        hits = int((sl["label"] == 1).sum())
        d = {"k_per_week": math.ceil(s * n_blocks), "selected": len(sel), "selected_labelled": len(sl), "hits": hits,
             "precision": hits / len(sl) if len(sl) else float("nan"),
             "recall": hits / pos_total if pos_total else float("nan")}
        if reports is not None and len(reports):
            got =reports.merge(sel[["block_id", "origin"]], on=["block_id", "origin"])[["origin", "report_id"]].drop_duplicates()
            d["reports_captured"] = len(got)
            d["reports_total"] = int(rep_tot)
            d["report_capture_rate"] = len(got) / rep_tot if rep_tot else float("nan")
        out[f"top_{s * 100:g}pct"] = d
    return out


def reliability(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = np.unique(np.quantile(prob, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, prob, side="right") - 1, 0, len(q) - 2)
    df = pd.DataFrame({"b": idx, "p": prob, "y": y}).groupby("b").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    return df.reset_index(drop=True)


def core_metrics(pred: pd.DataFrame) -> dict:
    lab = pred[pred["label"].notna()]
    y = lab["label"].to_numpy().astype(int)
    out = {"block_weeks": len(pred), "labelled": len(lab), "unknown": int(pred["label"].isna().sum()),
           "positives": int(y.sum()), "prevalence": float(y.mean()) if len(y) else float("nan"),
           "weeks": int(pred["origin"].nunique())}
    if len(y) and 0 < y.sum() < len(y):
        out["average_precision"] = float(average_precision_score(y, lab["score"].to_numpy()))
        out["ap_lift_over_prevalence"] = out["average_precision"] / out["prevalence"]
        p = lab["prob"].to_numpy()
        out["brier"] = float(brier_score_loss(y, p))
        out["brier_reference_prevalence"] = float(brier_score_loss(y, np.full(len(y), y.mean())))
        out["mean_prob"] = float(p.mean())
        rel = reliability(p, y)
        out["calibration_error_weighted"] = float((rel["n"] * (rel["mean_pred"] - rel["observed"]).abs()).sum() / rel["n"].sum())
        out["calibration_in_the_large_ratio"] = float(p.mean() / y.mean())
    return out


def evaluate(pred: pd.DataFrame, reports: pd.DataFrame | None, n_blocks: int) -> dict:
    return {**core_metrics(pred), "capacity": capacity_metrics(pred, reports, n_blocks)}


def by_group(pred: pd.DataFrame, reports: pd.DataFrame | None, n_blocks: int, key: pd.Series, min_pos: int = 20) -> dict:
    out = {}
    for g, idx in pred.groupby(key.to_numpy()).groups.items():
        sub = pred.loc[idx]
        rep = reports[reports["origin"].isin(set(sub["origin"]))] if reports is not None else None
        m = core_metrics(sub)
        if m["positives"] >= min_pos:
            m["capacity"] = capacity_metrics(sub, rep, n_blocks, shares=(0.01,))
        out[str(g)] = m
    return out


def plot_pr(curves: dict[str, pd.DataFrame], title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, lab in curves.items():
        p, r, _ = precision_recall_curve(lab["label"].astype(int), lab["score"])
        ap = average_precision_score(lab["label"].astype(int), lab["score"])
        ax.plot(r, p, lw=1.4, label=f"{name} (AP {ap:.3f})")
    prev = next(iter(curves.values()))["label"].mean()
    ax.axhline(prev, color="grey", ls=":", lw=1, label=f"prevalence {prev:.3%}")
    ax.set_xlabel("Recall"), ax.set_ylabel("Precision"), ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1), ax.set_ylim(0, min(1, max(0.05, ax.get_ylim()[1])))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_reliability(curves: dict[str, pd.DataFrame], title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    mx = 0
    for name, lab in curves.items():
        rel = reliability(lab["prob"].to_numpy(), lab["label"].to_numpy().astype(int), bins=15)
        ax.plot(rel["mean_pred"], rel["observed"], marker="o", ms=3, lw=1.2, label=name)
        mx = max(mx, rel["mean_pred"].max(), rel["observed"].max())
    ax.plot([0, mx], [0, mx], color="grey", ls=":", lw=1, label="perfect calibration")
    ax.set_xlabel("Mean predicted probability (quantile bins)"), ax.set_ylabel("Observed frequency")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
