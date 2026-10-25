"""Pre-registered multimodal experiments (see docs/final/multimodal_research_plan.md).

Reuses the production fitting code unchanged (stage4.run_fit / modeling.fit_model, frozen hyper-parameters, same negative
sampling, holdout and Platt calibration) and the Stage 5 decision engine (inspection.select_all, capture_by_definition).
Extra feature groups are merged onto the production block-week features by (block_id, origin) / block_id. Writes only under
data/processed/multimodal and outputs/multimodal.

CLI:  python -I -X utf8 -m roadwatch.multimodal.experiment validation|test|report [--sets F0,F3,...] [--model lgbm|catboost]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import inspection as insp
from ..config import Paths
from ..features import FEATURE_SETS
from ..modeling import TEST_START, ModelSpec, retrain_schedule
from ..stage4 import TRAIN, VAL, Data, last_origin, run_fit
from ..storage import read_json, sha256_file, utc_iso
from . import infra_features as inf

D = dt.date
VARIANT = "primary"                                     # strict single-block target
BASE_SET = "history_static"
CAPS = (0.005, 0.01, 0.02)
STRESS = (D(2026, 3, 2), D(2026, 3, 30))
TERRAIN = ["terrain_elev_mean_ft", "terrain_elev_std_ft", "terrain_slope_mean", "terrain_slope_max", "terrain_rel_elev_300ft_ft",
           "terrain_low_point_rel_ft", "terrain_cover"]
SETS = {  # fixed in advance (research plan section 4); a set is skipped (not faked) if its source is unavailable
    "F0": [], "F1": ["terrain"], "F2": ["displacement"], "F3": ["infra"], "F4": ["permits"], "F5": ["terrain", "infra"],
}
GROUP_FEATURES = {"terrain": TERRAIN, "infra": inf.feature_names(), "displacement": [], "permits": []}


def out_dirs(paths: Paths):
    a, b = paths.processed / "multimodal", paths.root / "outputs" / "multimodal"
    for d in (a / "predictions", b):
        d.mkdir(parents=True, exist_ok=True)
    return a, b


# ------------------------------------------------------------------ data
class Extras:
    """Experimental feature tables. Dynamic (block x origin) and static (block) groups, float32."""

    def __init__(self, paths: Paths):
        a, _ = out_dirs(paths)
        self.dyn = pd.read_parquet(a / "infra_features.parquet") if (a / "infra_features.parquet").exists() else None
        if self.dyn is not None:
            self.dyn["origin"] = pd.to_datetime(self.dyn["origin"])
        self.static = pd.read_parquet(a / "terrain_features.parquet") if (a / "terrain_features.parquet").exists() else None

    def available(self, group: str) -> bool:
        return {"infra": self.dyn is not None, "terrain": self.static is not None}.get(group, False)


class MMData(Data):
    """Production Data plus experimental columns merged by key (no row is added or dropped)."""

    def __init__(self, paths, base_features, rate, extras: Extras, groups: list[str]):
        super().__init__(paths, base_features, rate)
        self.extras, self.groups = extras, groups

    def _merge(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        if "infra" in self.groups:
            o = pd.to_datetime(df["origin"])
            sub = self.extras.dyn[(self.extras.dyn["origin"] >= o.min()) & (self.extras.dyn["origin"] <= o.max())]
            df = df.assign(_o=o).merge(sub.rename(columns={"origin": "_o"}), on=["block_id", "_o"], how="left").drop(columns="_o")
        if "terrain" in self.groups:
            df = df.merge(self.extras.static, on="block_id", how="left")
        assert len(df) == n, "merge changed the number of rows"
        return df

    def sample(self, variant, a, b):
        df = super().sample(variant, a, b)
        key = f"{variant}|{a}|{b}"
        if "_merged" not in df.attrs:
            df = self._merge(df)
            df.attrs["_merged"] = True
            self.cache[key] = df
        return df

    def rows(self, variant, a, b):
        return self._merge(super().rows(variant, a, b))


def feature_list(groups: list[str]) -> list[str]:
    return list(FEATURE_SETS[BASE_SET]) + [f for g in groups for f in GROUP_FEATURES[g]]


# ------------------------------------------------------------------ fitting
def make_spec(model: str, sel: dict) -> ModelSpec:
    params = sel["chosen"][VARIANT]["LGBM"]["params"]                 # frozen production hyper-parameters
    return ModelSpec("LGBM", dict(params))


def fit_period(paths: Paths, sel: dict, sched, set_id: str, groups: list[str], extras: Extras, model: str = "lgbm") -> tuple[pd.DataFrame, list[dict]]:
    feats = feature_list(groups)
    data = MMData(paths, list(FEATURE_SETS[BASE_SET]), sel["chosen"]["sampling_rate"], extras, groups)
    parts, infos = [], []
    for r, p_to in sched:
        spec = make_spec(model, sel)
        if model == "catboost":
            pred, info = _catboost_fit(data, feats, r, p_to, spec)
        else:
            pred, fr = run_fit(spec, feats, VARIANT, data, r, TRAIN[0], (r, p_to))
            info = fr.info
        t0 = time.perf_counter()
        parts.append(pred)
        infos.append({**info, "retrain_date": str(r), "n_features": len(feats)})
    return pd.concat(parts, ignore_index=True), infos


def _catboost_fit(data: MMData, feats: list[str], r: D, p_to: D, spec: ModelSpec):
    """CatBoost on the same fit/holdout/prediction rows and the same Platt calibration as production LightGBM."""
    from catboost import CatBoostClassifier, Pool
    from ..features import CATEGORICAL
    from ..modeling import Platt, windows_for
    w = windows_for(r)
    ycol = f"y_{VARIANT}"
    fit = data.sample(VARIANT, TRAIN[0], w["fit_before"] - dt.timedelta(days=1))
    fit = fit[(fit["origin"] >= TRAIN[0]) & (fit["origin"] < w["fit_before"])]
    hold = data.rows(VARIANT, w["holdout_from"], w["last_matured_origin"])
    hold = hold[hold[ycol].notna()]
    pr = data.rows(VARIANT, r, p_to)
    cats = [f for f in feats if f in CATEGORICAL]
    for df in (fit, hold, pr):
        for c in cats:
            df[c] = df[c].astype("Int64").astype("string").fillna("NA")
    t0 = time.perf_counter()
    m = CatBoostClassifier(iterations=1500, learning_rate=0.08, depth=6, loss_function="Logloss", eval_metric="PRAUC", random_seed=7,
                           od_type="Iter", od_wait=50, verbose=False, thread_count=8, cat_features=cats, allow_writing_files=False)
    m.fit(Pool(fit[feats], fit[ycol].astype(int), weight=fit["_w"], cat_features=cats),
          eval_set=Pool(hold[feats], hold[ycol].astype(int), cat_features=cats), use_best_model=True)
    raw_h = m.predict_proba(hold[feats])[:, 1]
    cal = Platt("logit").fit(raw_h, hold[ycol].to_numpy().astype(int))
    secs = time.perf_counter() - t0
    raw = m.predict_proba(pr[feats])[:, 1]
    pred = pd.DataFrame({"block_id": pr["block_id"].to_numpy(), "origin": pr["origin"].to_numpy(), "score": raw, "prob": cal.predict(raw),
                         "label": pr[ycol].to_numpy(), "retrain_date": r})
    return pred, {"seconds": round(secs, 2), "best_iteration": int(m.get_best_iteration()), "fit_rows": len(fit)}


# ------------------------------------------------------------------ evaluation
def _reports(paths: Paths):
    from ..stage5 import block_nodes
    a = pd.read_parquet(paths.processed / "stage3" / "report_assignments.parquet",
                        columns=["origin", "report_id", "candidate_block_ids", "assigned_block_id", "match_status", "method", "included_in_labels"])
    a["origin"] = pd.to_datetime(a["origin"]).dt.date
    return a, block_nodes(paths)


def evaluate(paths: Paths, pred: pd.DataFrame, a_assign, nodes, window: tuple[D, D], eligible: int = 23448) -> dict:
    """Per-capacity weekly vectors (for paired bootstraps) and pooled metrics on the same observable block-weeks."""
    from sklearn.metrics import average_precision_score, brier_score_loss
    from ..stage5 import reports as reports_for
    p = pred.copy()
    p["origin"] = pd.to_datetime(p["origin"]).dt.date
    p = p[(p["origin"] >= window[0]) & (p["origin"] <= window[1])]
    lab = p[p["label"].notna()]
    rf = insp.report_frame(a_assign[(a_assign["origin"] >= window[0]) & (a_assign["origin"] <= window[1])], nodes)
    rp = reports_for(paths, VARIANT)
    out = {"n_weeks": int(p["origin"].nunique()), "rows": int(len(lab)), "prevalence": float(lab["label"].mean()),
           "average_precision": float(average_precision_score(lab["label"].astype(int), lab["score"])),
           "brier": float(brier_score_loss(lab["label"].astype(int), lab["prob"])),
           "calibration_in_the_large": float(lab["prob"].sum() / max(lab["label"].sum(), 1)), "caps": {}}
    for cap in CAPS:
        sel, _, _ = insp.select_all(p, cap, eligible)
        w = insp.weekly_outcomes(sel, lab[["block_id", "origin", "label"]], rp[(rp["origin"] >= window[0]) & (rp["origin"] <= window[1])]).set_index("origin")
        cd = insp.capture_by_definition(sel, rf).set_index("origin")
        out["caps"][cap] = {"weekly": pd.concat([w[["selected", "labelled_selected", "hits"]], cd], axis=1).sort_index(), "selection": sel}
    return out


def paired_ci(num_a, den_a, num_b, den_b, n=1000, seed=11) -> list[float]:
    """95% percentile CI of (A - B) ratio-of-sums, resampling WEEKS jointly (paired)."""
    rng = np.random.default_rng(seed)
    k = len(num_a)
    idx = rng.integers(0, k, size=(n, k))
    d = num_a[idx].sum(1) / den_a[idx].sum(1) - num_b[idx].sum(1) / den_b[idx].sum(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def summarize(ev: dict, ref: dict | None = None) -> list[dict]:
    rows = []
    for cap, c in ev["caps"].items():
        w = c["weekly"]
        r = {"capacity": cap, "precision": float(w["hits"].sum() / w["labelled_selected"].sum()),
             "recall": float(w["hits"].sum() / max(ev_positives(ev), 1)),
             "capture_any": float(w["any_candidate"].sum() / w["reports_total"].sum()),
             "capture_fractional": float(w["fractional"].sum() / w["reports_total"].sum()),
             "capture_assigned": float(w["assigned_block"].sum() / w["reports_total"].sum()),
             "capture_intersection_node": float(w["intersection_node"].sum() / w["reports_total"].sum()),
             "reports_total": int(w["reports_total"].sum())}
        if ref is not None:
            rw = ref["caps"][cap]["weekly"]
            assert list(rw.index) == list(w.index)
            r["d_precision_ci95"] = paired_ci(w["hits"].to_numpy(float), w["labelled_selected"].to_numpy(float), rw["hits"].to_numpy(float), rw["labelled_selected"].to_numpy(float))
            r["d_precision"] = r["precision"] - float(rw["hits"].sum() / rw["labelled_selected"].sum())
            r["d_capture_fractional_ci95"] = paired_ci(w["fractional"].to_numpy(float), w["reports_total"].to_numpy(float), rw["fractional"].to_numpy(float), rw["reports_total"].to_numpy(float))
            r["d_capture_fractional"] = r["capture_fractional"] - float(rw["fractional"].sum() / rw["reports_total"].sum())
            a, b = c["selection"], ref["caps"][cap]["selection"]
            sa = set(zip(a["block_id"], a["origin"]))
            sb = set(zip(b["block_id"], b["origin"]))
            r["list_overlap_with_baseline"] = len(sa & sb) / len(sa)
        rows.append(r)
    return rows


def ev_positives(ev: dict) -> int:
    return int(round(ev["prevalence"] * ev["rows"]))


def report(paths: Paths, stage: str, model: str, sets: list[str], ref_id: str = "F0") -> dict:
    """Compare candidate predictions with the F0 predictions of the SAME run on identical weeks. Writes JSON + CSV."""
    a_dir, o_dir = out_dirs(paths)
    window = VAL if stage == "validation" else (TEST_START, last_origin(paths))
    a_assign, nodes = _reports(paths)
    evs, preds = {}, {}
    for sid in sets:
        f = a_dir / "predictions" / f"{stage}_{model}_{sid}.parquet"
        if f.exists():
            preds[sid] = pd.read_parquet(f)
            evs[sid] = evaluate(paths, preds[sid], a_assign, nodes, window)
    ref = evs[ref_id]
    out = {"stage": stage, "model": model, "window": [str(window[0]), str(window[1])], "weeks": ref["n_weeks"], "prevalence": ref["prevalence"], "sets": {}}
    rows = []
    for sid, ev in evs.items():
        s = summarize(ev, None if sid == ref_id else ref)
        out["sets"][sid] = {"average_precision": ev["average_precision"], "brier": ev["brier"], "calibration_in_the_large": ev["calibration_in_the_large"], "capacities": s}
        for r in s:
            rows.append({"set": sid, "average_precision": ev["average_precision"], "brier": ev["brier"], **{k: v for k, v in r.items()}})
    # March 2026 surge and by-month precision at 1% (test only)
    if stage == "test":
        mar = {}
        for sid, ev in evs.items():
            w = ev["caps"][0.01]["weekly"]
            m = w[(w.index >= STRESS[0]) & (w.index <= STRESS[1])]
            mar[sid] = {"weeks": len(m), "precision": float(m["hits"].sum() / m["labelled_selected"].sum()), "capture_fractional": float(m["fractional"].sum() / m["reports_total"].sum())}
        out["march_2026"] = mar
    monthly = []
    for sid, ev in evs.items():
        w = ev["caps"][0.01]["weekly"].copy()
        w["month"] = pd.to_datetime(pd.Series(w.index, index=w.index)).dt.to_period("M").astype(str)
        g = w.groupby("month").agg(h=("hits", "sum"), l=("labelled_selected", "sum"), f=("fractional", "sum"), r=("reports_total", "sum"))
        for mth, x in g.iterrows():
            monthly.append({"set": sid, "month": mth, "precision_1pct": x.h / x.l, "capture_fractional_1pct": x.f / x.r})
    # timing
    fits = json.loads((o_dir / f"fits_{stage}_{model}.json").read_text(encoding="utf-8")) if (o_dir / f"fits_{stage}_{model}.json").exists() else {}
    out["fit_seconds_total"] = {sid: v["seconds_total"] for sid, v in fits.items()}
    (o_dir / f"comparison_{stage}_{model}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    pd.DataFrame(rows).to_csv(o_dir / f"comparison_{stage}_{model}.csv", index=False)
    pd.DataFrame(monthly).to_csv(o_dir / f"monthly_{stage}_{model}.csv", index=False)
    return out


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["validation", "test", "report-validation", "report-test"])
    ap.add_argument("--sets", default="F0,F1,F3,F5")
    ap.add_argument("--model", default="lgbm", choices=["lgbm", "catboost"])
    args = ap.parse_args(argv)
    paths = Paths()
    if args.stage.startswith("report-"):
        r = report(paths, args.stage.split("-")[1], args.model, args.sets.split(","))
        print(json.dumps({k: v for k, v in r.items() if k != "sets"}, default=float))
        return 0
    a_dir, o_dir = out_dirs(paths)
    sel = read_json(paths.processed / "stage4" / "selection.json")
    extras = Extras(paths)
    sched = retrain_schedule(*VAL) if args.stage == "validation" else retrain_schedule(TEST_START, last_origin(paths))
    results = {}
    for sid in args.sets.split(","):
        groups = SETS[sid]
        missing = [g for g in groups if not extras.available(g)]
        if missing:
            print(f"{sid}: skipped, source unavailable: {missing}")
            continue
        f = a_dir / "predictions" / f"{args.stage}_{args.model}_{sid}.parquet"
        t0 = time.perf_counter()
        pred, infos = fit_period(paths, sel, sched, sid, groups, extras, args.model)
        pred.to_parquet(f, index=False)
        results[sid] = {"file": str(f), "predictions_sha256": sha256_file(f), "fits": infos, "seconds_total": round(time.perf_counter() - t0, 1),
                        "features": feature_list(groups), "generated_utc": utc_iso()}
        print(sid, args.stage, args.model, "done", results[sid]["seconds_total"], "s")
    (o_dir / f"fits_{args.stage}_{args.model}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
