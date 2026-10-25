"""Capacity-constrained weekly inspection selection (decision engine).

Policy: top-K by model score per forecast week (Stage 4 conventions):
  K = ceil(capacity_share * eligible_blocks)   (or an explicit integer K)
  ties broken by the deterministic splitmix64 hash of (block_id, origin), ascending - never by row order.
Optional ZIP diversification: walk the same ranking but skip a block once its ZIP has used its quota
  quota_z = max(1, ceil(m * K * eligible_share_z)); quotas always sum to >= K when m >= 1.

Selection reads ONLY block_id, origin, score (and ZIP for diversification). Labels, report outcomes and test metrics are
never inputs, so historical decisions cannot use future outcomes. One inspection slot = one block; a block can be
selected at most once per week; blocks without a valid finite score are excluded and reported, never imputed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .modeling import det_uniform

SELECTION_INPUTS = ("block_id", "origin", "score")


class SelectionError(Exception):
    pass


def capacity_k(capacity: float | int, eligible_blocks: int) -> int:
    if isinstance(capacity, (int, np.integer)) and not isinstance(capacity, bool):
        if capacity < 0:
            raise SelectionError("capacity must be >= 0")
        return int(capacity)
    if not 0 <= capacity <= 1:
        raise SelectionError(f"capacity share must be in [0, 1], got {capacity}")
    return math.ceil(capacity * eligible_blocks)


@dataclass
class WeekSelection:
    origin: object
    k: int
    selected: pd.DataFrame                  # block_id, origin, score, rank (1 = highest priority), zip (optional)
    excluded: pd.DataFrame                  # block_id, origin, reason
    shortfall: int = 0                      # slots left empty because fewer valid blocks than K
    diversification: dict = field(default_factory=dict)


def _validate(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in SELECTION_INPUTS if c not in scores.columns]
    if missing:
        raise SelectionError(f"missing columns {missing}")
    if scores["origin"].nunique() > 1:
        raise SelectionError("select_week expects a single forecast origin")
    dup = scores["block_id"].duplicated(keep=False)
    if dup.any():
        raise SelectionError(f"{int(scores.loc[dup, 'block_id'].nunique())} block(s) appear more than once for this week")
    s = pd.to_numeric(scores["score"], errors="coerce")
    bad = ~np.isfinite(s.to_numpy(dtype=float))
    excluded = scores.loc[bad, ["block_id", "origin"]].assign(reason="invalid_or_missing_score")
    return scores.loc[~bad].assign(score=s[~bad]), excluded


def rank_week(scores: pd.DataFrame) -> pd.DataFrame:
    """Full deterministic ranking of valid blocks for one week (rank 1 = highest score)."""
    valid, _ = _validate(scores[list(SELECTION_INPUTS) + [c for c in scores.columns if c == "zip"]])
    v = valid.assign(_tb=det_uniform(valid["block_id"].to_numpy(), valid["origin"]))
    v = v.sort_values(["score", "_tb"], ascending=[False, True], kind="stable").reset_index(drop=True)
    v["rank"] = np.arange(1, len(v) + 1)
    return v.drop(columns="_tb")


def select_week(scores: pd.DataFrame, capacity: float | int, eligible_blocks: int,
                zip_share: pd.Series | None = None, max_multiple: float | None = None,
                nodes: pd.DataFrame | None = None) -> WeekSelection:
    """`scores` holds one origin; extra columns (e.g. labels) are ignored. For diversification pass `zip` in scores and
    `zip_share` (share of eligible blocks per ZIP). With `nodes` (index block_id; columns node_from, node_to) the
    parameter-free NON-ADJACENT rule applies: walk the ranking and skip a block that shares a centerline node with an
    already-selected block, so one intersection is not covered by several slots."""
    cols = list(SELECTION_INPUTS) + (["zip"] if "zip" in scores.columns else [])
    _, excluded = _validate(scores[cols])
    ranked = rank_week(scores[cols])
    k = capacity_k(capacity, eligible_blocks)
    origin = scores["origin"].iloc[0] if len(scores) else None
    div = {}
    if nodes is not None and max_multiple is not None:
        raise SelectionError("combine either the ZIP cap or the non-adjacent rule, not both")
    if nodes is not None:
        nf = nodes["node_from"].reindex(ranked["block_id"]).to_numpy()
        nt = nodes["node_to"].reindex(ranked["block_id"]).to_numpy()
        used, take = set(), []
        for i in range(len(ranked)):
            if len(take) == k:
                break
            if nf[i] in used or nt[i] in used:
                continue
            take.append(i)
            used.update((nf[i], nt[i]))
        sel = ranked.iloc[take].assign(rank=np.arange(1, len(take) + 1))
        div = {"rule": "non_adjacent", "displaced_from_unconstrained_top_k":
               int((~ranked.head(k)["block_id"].isin(sel["block_id"])).sum())}
    elif max_multiple is None:
        sel = ranked.head(k)
    else:
        if zip_share is None or "zip" not in ranked.columns:
            raise SelectionError("diversification needs a zip column and zip_share")
        quota = {z: max(1, math.ceil(max_multiple * k * s)) for z, s in zip_share.items()}
        used: dict = {}
        take = []
        for i, z in enumerate(ranked["zip"].to_numpy()):
            if len(take) == k:
                break
            if used.get(z, 0) < quota.get(z, 1):
                used[z] = used.get(z, 0) + 1
                take.append(i)
        sel = ranked.iloc[take]
        skipped = int((ranked.head(k)["block_id"].isin(sel["block_id"]) == False).sum())  # noqa: E712
        div = {"max_multiple": max_multiple, "displaced_from_unconstrained_top_k": skipped,
               "zips_at_quota": int(sum(used.get(z, 0) >= q for z, q in quota.items()))}
        sel = sel.assign(rank=np.arange(1, len(sel) + 1))
    if sel["block_id"].duplicated().any():          # defensive: cannot happen after _validate
        raise SelectionError("duplicate block selected")
    return WeekSelection(origin, k, sel.reset_index(drop=True), excluded, max(0, k - len(sel)), div)


def select_all(pred: pd.DataFrame, capacity: float | int, eligible_blocks: int, zip_of: pd.Series | None = None,
               max_multiple: float | None = None, nodes: pd.DataFrame | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run select_week for every origin. Returns (selected, excluded, per-week summary)."""
    p = pred[list(SELECTION_INPUTS)].copy()
    zip_share = None
    if max_multiple is not None or zip_of is not None:
        p["zip"] = zip_of.reindex(p["block_id"]).to_numpy()
        zip_share = zip_of.value_counts(normalize=True)
    sels, excl, summ = [], [], []
    for o, g in p.groupby("origin", sort=True):
        ws = select_week(g, capacity, eligible_blocks, zip_share, max_multiple, nodes)
        sels.append(ws.selected)
        excl.append(ws.excluded)
        summ.append({"origin": o, "k": ws.k, "selected": len(ws.selected), "excluded": len(ws.excluded),
                     "shortfall": ws.shortfall, **{f"div_{a}": b for a, b in ws.diversification.items()}})
    return (pd.concat(sels, ignore_index=True), pd.concat(excl, ignore_index=True) if excl else pd.DataFrame(),
            pd.DataFrame(summ))


# ---------------- outcome evaluation (labels used ONLY here, after selection) ----------------
def weekly_outcomes(selected: pd.DataFrame, labels: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Per week: hits among labelled selections, positives, and UNIQUE report ids captured.
    `labels`: block_id, origin, label (NaN = unknown). `reports`: block_id, origin, report_id - a report may appear on
    several candidate blocks; it is captured once if any of its blocks is selected."""
    lab = labels[["block_id", "origin", "label"]]
    s = selected[["block_id", "origin"]].merge(lab, on=["block_id", "origin"], how="left")
    sel_w = s.groupby("origin").agg(selected=("block_id", "size"), labelled_selected=("label", "count"),
                                    hits=("label", lambda x: int((x == 1).sum())))
    pos_w = lab.groupby("origin").agg(labelled=("label", "count"), positives=("label", lambda x: int((x == 1).sum())))
    rep_tot = reports.groupby("origin")["report_id"].nunique().rename("reports_total")
    cap = reports.merge(selected[["block_id", "origin"]], on=["block_id", "origin"])
    rep_cap = cap.drop_duplicates(["origin", "report_id"]).groupby("origin").size().rename("reports_captured")
    out = sel_w.join(pos_w, how="outer").join(rep_tot, how="left").join(rep_cap, how="left").fillna(0)
    return out.astype({c: int for c in out.columns}).reset_index()


CAPTURE_DEFINITIONS = {
    "any_candidate": "captured if ANY candidate block of the report is inspected (optimistic for ambiguous reports)",
    "assigned_block": "captured if the report's single assigned block is inspected (assignment of ambiguous reports is a "
                      "hash pick, so this is an unbiased estimate of fractional capture)",
    "fractional": "credit = share of the report's candidate blocks inspected (expected capture if the true location is "
                  "uniform over candidates)",
    "intersection_node": "unique reports: assigned block; ambiguous INTERSECTION reports whose candidates all meet at one "
                         "node: any candidate (assumes inspecting a block covers its end intersection); all other ambiguous "
                         "reports (e.g. blockface alternatives along a street): fractional",
}


def report_frame(assignments: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """One row per included report: origin, report_id, candidates, assigned block, ambiguity and shared-node flag."""
    a = assignments[assignments["included_in_labels"]].copy()
    a["candidates"] = a["candidate_block_ids"].map(lambda c: [int(x) for x in c])
    nf, nt = nodes["node_from"].to_dict(), nodes["node_to"].to_dict()

    def shared(c):
        s = None
        for b in c:
            e = {nf.get(b), nt.get(b)}
            s = e if s is None else s & e
        return bool(s) and len(c) > 1
    a["ambiguous"] = a["match_status"] == "ambiguous"
    # coverage assumption applies only to genuine intersection reports meeting at one node, not to blockface alternatives
    a["shared_node"] = a["candidates"].map(shared) & a["method"].fillna("").str.startswith("intersection")
    return a[["origin", "report_id", "candidates", "assigned_block_id", "ambiguous", "shared_node", "method"]].reset_index(drop=True)


def capture_by_definition(selected: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Per-week captured credit under each definition (labels/outcomes used only AFTER selection).
    `reports` = report_frame(...) restricted to the evaluated weeks; weeks with no capture keep explicit zeros."""
    sel = selected.groupby("origin")["block_id"].apply(lambda s: set(int(x) for x in s)).to_dict()
    rows = []
    for o, cands, asg, amb, sh in zip(reports["origin"], reports["candidates"], reports["assigned_block_id"],
                                      reports["ambiguous"], reports["shared_node"]):
        S = sel.get(o, set())
        hit = sum(1 for c in cands if c in S)
        frac = hit / len(cands) if cands else 0.0
        anyc = float(hit > 0)
        asgc = float(int(asg) in S) if pd.notna(asg) else 0.0
        node = (anyc if sh else frac) if amb else asgc
        rows.append((o, anyc, asgc, frac, node))
    df = pd.DataFrame(rows, columns=["origin"] + list(CAPTURE_DEFINITIONS))
    out = df.groupby("origin").sum()
    out["reports_total"] = df.groupby("origin").size()
    weeks = sorted(set(reports["origin"]) | set(sel))
    return out.reindex(weeks, fill_value=0.0).rename_axis("origin").reset_index()


def summarize(w: pd.DataFrame) -> dict:
    t = w[["selected", "labelled_selected", "hits", "positives", "reports_total", "reports_captured"]].sum()
    return {"weeks": int(len(w)), "inspections": int(t["selected"]), "labelled_selected": int(t["labelled_selected"]),
            "hits": int(t["hits"]), "positives": int(t["positives"]),
            "precision": t["hits"] / t["labelled_selected"] if t["labelled_selected"] else float("nan"),
            "recall": t["hits"] / t["positives"] if t["positives"] else float("nan"),
            "reports_total": int(t["reports_total"]), "reports_captured": int(t["reports_captured"]),
            "report_capture_rate": t["reports_captured"] / t["reports_total"] if t["reports_total"] else float("nan"),
            "reports_per_100_inspections": 100 * t["reports_captured"] / t["selected"] if t["selected"] else float("nan")}


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, rng, n: int, num_b=None, den_b=None) -> list[float]:
    """95% CI of sum(num)/sum(den) over resampled weeks; with num_b/den_b, CI of the paired difference A - B."""
    idx = rng.integers(0, len(num), (n, len(num)))
    a = num[idx].sum(1) / den[idx].sum(1)
    if num_b is not None:
        a = a - num_b[idx].sum(1) / den_b[idx].sum(1)
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


def concentration(counts: pd.Series) -> dict:
    s = counts[counts > 0]
    share = s / s.sum()
    return {"units_with_any": int(len(s)), "hhi": float((share ** 2).sum()), "top5_share": float(share.nlargest(5).sum())}
