"""Stage 4 orchestration: features -> validation-only selection (frozen) -> walk-forward test -> evaluation.

Steps (python -I -m roadwatch stage4 <step>):
  features  build as-of feature tables (one Parquet per year, labels for every variant as y_<variant>, NaN = unknown)
  select    all model/hyper-parameter/sampling choices on train+validation ONLY; writes selection.json (frozen)
  test      monthly walk-forward over the test period using the frozen selection; persists predictions + models
  evaluate  metrics, plots, sensitivities, stress tests -> outputs/stage4/
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import platform
import time
from importlib import metadata

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from . import evaluation as ev
from .conditional_features import EXPERIMENTAL, ConditionalFeatureBuilder, experimental_definitions
from .config import Paths
from .features import CATEGORICAL, DIAGNOSTIC_ONLY, FEATURE_SETS, FeatureBuilder, feature_definitions
from .modeling import (TEST_START, ModelSpec, det_uniform, fit_model, guard_selection_data, negative_sample,
                       predict_raw, retrain_schedule, windows_for)
from .stage3_diagnostics import _md_table
from .storage import read_json, sha256_file, utc_iso, write_json

log = logging.getLogger(__name__)
D = dt.date
TRAIN = (D(2021, 1, 4), D(2024, 12, 30))
VAL = (D(2025, 1, 6), D(2025, 6, 30))
MAIN_VARIANTS = ["primary", "alt_tiebreak", "multi_block", "confident_only"]
ALL_VARIANTS = MAIN_VARIANTS + ["no_supplementary"]
OPERATIONAL = "multi_block"
SAMPLING_RATES = (0.05, 0.10, 0.25)
LGBM_GRID = [{"num_leaves": 15, "min_data_in_leaf": 200}, {"num_leaves": 15, "min_data_in_leaf": 2000},
             {"num_leaves": 63, "min_data_in_leaf": 200}, {"num_leaves": 63, "min_data_in_leaf": 2000}]
LR_GRID = [0.01, 0.1, 1.0]
B1_GRID = [2, 10, 50]
B2_GRID = [7, 30, 90]
STRESS = (D(2026, 3, 2), D(2026, 3, 30))
NOAA_PRE = (D(2025, 7, 7), D(2025, 8, 25))     # every weather window entirely ISD ('W') data
NOAA_POST = (D(2025, 11, 3), D(2026, 10, 12))  # every 60-day weather window entirely CF6 ('1') data


def dirs(paths: Paths):
    base = paths.processed / "stage4"
    rep = paths.root / "outputs" / "stage4"
    for d in (base / "features", base / "predictions", base / "models", rep / "plots"):
        d.mkdir(parents=True, exist_ok=True)
    return base, rep


def last_origin(paths: Paths) -> D:
    return D.fromisoformat(read_json(paths.processed / "stage3" / "populations.json")["recommended_primary"]["origin_to"])


def mondays(a: D, b: D) -> list[D]:
    out, t = [], a
    while t <= b:
        out.append(t)
        t += dt.timedelta(days=7)
    return out


# ============================== features ==============================
def step_features(paths: Paths) -> dict:
    base, rep = dirs(paths)
    s3 = paths.processed / "stage3"
    blocks = gpd.read_parquet(s3 / "blocks.parquet")
    fb = FeatureBuilder(pd.read_parquet(s3 / "report_assignments.parquet"), pd.read_parquet(s3 / "label_weeks.parquet"),
                        blocks, pd.read_parquet(paths.processed / "noaa.parquet"))
    cond = ConditionalFeatureBuilder(blocks, gpd.read_parquet(paths.processed / "pavement.parquet"),
                                     gpd.read_parquet(paths.processed / "workorders.parquet"),
                                     gpd.read_parquet(paths.processed / "resurfacing_inhouse.parquet"))
    end = last_origin(paths)
    origins = mondays(TRAIN[0], end)
    out_sha, rows = {}, 0
    for year in sorted({o.year for o in origins}):
        oy = [o for o in origins if o.year == year]
        X = fb.build(oy).merge(cond.build(oy), on=["block_id", "origin"], how="left")
        for v in ALL_VARIANTS:
            lab = pq.read_table(s3 / f"labels_{v}.parquet", columns=["block_id", "origin", "label"],
                                filters=[("origin", ">=", oy[0]), ("origin", "<=", oy[-1])]).to_pandas()
            lab = lab.rename(columns={"label": f"y_{v}"})
            lab[f"y_{v}"] = lab[f"y_{v}"].astype("float32")          # NaN = unknown (never 0)
            X = X.merge(lab, on=["block_id", "origin"], how="left")
        X = X.sort_values(["origin", "block_id"]).reset_index(drop=True)
        f = base / "features" / f"features_{year}.parquet"
        tmp = f.with_name(f.name + ".tmp")
        X.to_parquet(tmp, index=False, row_group_size=len(fb.block_ids) * 4)
        os.replace(tmp, f)
        out_sha[f.name] = sha256_file(f)
        rows += len(X)
        log.info("features %d: %d rows", year, len(X))
        del X
    defs = [d.__dict__ for d in feature_definitions() + experimental_definitions()]
    write_json(rep / "feature_definitions.json", defs)
    md = ["# Stage 4 feature definitions", "", "| Feature | Group | Window | Units | Source | Availability | Definition |", "|---|---|---|---|---|---|---|"]
    md += [f"| {d['name']} | {d['group']} | {d['window']} | {d['units']} | {d['source']} | {d['availability']} | {d['definition']} |" for d in defs]
    (rep / "feature_definitions.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    man = {"generated_utc": utc_iso(), "origins": [str(origins[0]), str(origins[-1])], "n_origins": len(origins),
           "rows": rows, "files_sha256": out_sha, "conditional_mapping": cond.mapping_stats,
           "feature_sets": FEATURE_SETS, "experimental": EXPERIMENTAL, "diagnostic_only": DIAGNOSTIC_ONLY,
           "inputs": {k: read_json(paths.processed / f"{k}.meta.json")["snapshot_id"] for k in
                      ("sr311", "sr311_unassigned", "centerline", "noaa", "pavement", "workorders", "resurfacing_inhouse")},
           "stage3_manifest_sha256": sha256_file(s3 / "stage3_manifest.json")}
    write_json(base / "features_manifest.json", man)
    regime_diagnostics(paths)
    return man


def load(paths: Paths, columns: list[str], a: D, b: D) -> pd.DataFrame:
    base = paths.processed / "stage4" / "features"
    ds = pads.dataset(sorted(str(p) for p in base.glob("features_*.parquet")), format="parquet")
    filt = (pads.field("origin") >= pa.scalar(a, pa.date32())) & (pads.field("origin") <= pa.scalar(b, pa.date32()))
    return ds.to_table(columns=list(dict.fromkeys(columns)), filter=filt).to_pandas()


def regime_diagnostics(paths: Paths) -> None:
    _, rep = dirs(paths)
    a = pd.read_parquet(paths.processed / "stage3" / "report_assignments.parquet")
    a = a[(a["source"] == "primary") | a["included_in_labels"]]
    a["year"] = pd.to_datetime(a["created_date_local"]).dt.year
    g = a.groupby("year")
    tab = pd.DataFrame({
        "reports": g.size(),
        "unique_pct": g["match_status"].apply(lambda s: (s == "matched").mean() * 100).round(1),
        "ambiguous_pct": g["match_status"].apply(lambda s: (s == "ambiguous").mean() * 100).round(1),
        "unmatched_or_out_pct": g["match_status"].apply(lambda s: s.isin(["unmatched", "out_of_universe"]).mean() * 100).round(2),
        "no_coordinates_pct": g["x"].apply(lambda s: s.isna().mean() * 100).round(1),
        "address_pct": g["address_type"].apply(lambda s: (s == "ADDRESS").mean() * 100).round(1),
        "blockface_pct": g["address_type"].apply(lambda s: (s == "BLOCKFACE").mean() * 100).round(1),
        "intersection_pct": g["address_type"].apply(lambda s: (s == "INTERSECTION").mean() * 100).round(1)})
    cols = ["origin", "rep_frac_365d", "no_report_in_365d", "pos_week_rate_52", "history_window_compromised",
            "freeze_thaw_days_30d", "prcp_mm_30d", "tmin_mean_7d"] + [f"y_{v}" for v in MAIN_VARIANTS]
    fs = []
    for y in range(2021, last_origin(paths).year + 1):
        x = load(paths, cols, D(y, 1, 1), D(y, 12, 31))
        r = {"year": y, "mean_rep_frac_365d": x["rep_frac_365d"].mean(), "share_no_report_365d": x["no_report_in_365d"].mean(),
             "mean_pos_week_rate_52": x["pos_week_rate_52"].mean(), "share_history_compromised": x["history_window_compromised"].mean()}
        wk = x.drop_duplicates("origin")
        r |= {"mean_weekly_freeze_thaw_days_30d": wk["freeze_thaw_days_30d"].mean(), "mean_prcp_mm_30d": wk["prcp_mm_30d"].mean()}
        for v in MAIN_VARIANTS:
            r[f"prevalence_{v}_pct"] = x[f"y_{v}"].mean() * 100
            r[f"unknown_{v}_pct"] = x[f"y_{v}"].isna().mean() * 100
        fs.append(r)
        del x
    feat = pd.DataFrame(fs).set_index("year").astype("float64").round(4)
    nm = pd.read_parquet(paths.processed / "noaa.parquet")
    nm["month"] = nm["date"].dt.month
    nm["period"] = np.where(nm["date"] < "2025-08-27", "ISD_W (to 2025-08-26)", "CF6_1 (from 2025-08-27)")
    same_months = nm[nm["month"].isin([9, 10])].assign(y=nm["date"].dt.year).groupby("y")[["tmax_value", "tmin_value", "prcp_value"]].mean().round(2)
    out = {"reports_by_year": tab.reset_index().to_dict("records"), "features_and_labels_by_year": feat.reset_index().to_dict("records"),
           "noaa_sep_oct_means_by_year": same_months.reset_index().to_dict("records"),
           "notes": ["Unique-match share falls after 2021 because BLOCKFACE reports lost coordinates (Stage 3, risk B7).",
                     "2026 prevalence includes the March 2026 surge.",
                     "NOAA source changes from ISD ('W') to NWS CF6 ('1') on 2025-08-27; Sep-Oct means shown for a like-for-like season check."]}
    write_json(rep / "regime_diagnostics.json", out)
    md = ["# Stage 4 - historical regime diagnostics (real data)", "", "## Reports by year (primary + evidence-backed supplementary)", "",
          _md_table(tab.reset_index()), "", "## Features and labels by year (forecast origins)", "",
          _md_table(feat.reset_index()), "", "## NOAA September-October means by year (source change 2025-08-27)", "",
          _md_table(same_months.reset_index()), "", *[f"- {n}" for n in out["notes"]]]
    (rep / "regime_diagnostics.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ============================== data access for modelling ==============================
class Data:
    """Deterministic negative-sampled fit sets (cached per variant) and full rows for holdout/prediction weeks."""

    def __init__(self, paths: Paths, features: list[str], rate: float):
        self.paths, self.features, self.rate = paths, features, rate
        self.cache: dict[str, pd.DataFrame] = {}

    def sample(self, variant: str, a: D, b: D) -> pd.DataFrame:
        key = f"{variant}|{a}|{b}"
        if key not in self.cache:
            parts = []
            for y in range(a.year, b.year + 1):
                x = load(self.paths, ["block_id", "origin", f"y_{variant}"] + self.features, max(a, D(y, 1, 1)), min(b, D(y, 12, 31)))
                x = x[x[f"y_{variant}"].notna()]
                keep, w = negative_sample(x[f"y_{variant}"].to_numpy(), det_uniform(x["block_id"].to_numpy(), x["origin"]), self.rate)
                parts.append(x[keep].assign(_w=w[keep]))
                del x
            self.cache.clear()
            self.cache[key] = pd.concat(parts, ignore_index=True)
        return self.cache[key]

    def rows(self, variant: str, a: D, b: D) -> pd.DataFrame:
        return load(self.paths, ["block_id", "origin", f"y_{variant}"] + self.features, a, b)


def run_fit(spec: ModelSpec, features, variant, data: Data, r: D, train_start: D, predict: tuple[D, D],
            exclude_to: D | None = None, hold_rows: pd.DataFrame | None = None, pred_rows: pd.DataFrame | None = None):
    w = windows_for(r)
    ycol = f"y_{variant}"
    fit = data.sample(variant, train_start, w["fit_before"] - dt.timedelta(days=1))
    fit = fit[(fit["origin"] >= train_start) & (fit["origin"] < w["fit_before"])]
    if exclude_to is not None:
        fit = fit[fit["origin"] > exclude_to]
    hold = hold_rows if hold_rows is not None else data.rows(variant, w["holdout_from"], w["last_matured_origin"])
    hold = hold[hold[ycol].notna()]
    assert hold["origin"].max() <= w["last_matured_origin"] < r and fit["origin"].max() < w["holdout_from"]
    fr = fit_model(spec, features, fit, hold, ycol)
    pr = pred_rows if pred_rows is not None else data.rows(variant, *predict)
    raw = predict_raw(spec, fr, pr, features)
    out = pd.DataFrame({"block_id": pr["block_id"].to_numpy(), "origin": pr["origin"].to_numpy(), "score": raw,
                        "prob": fr.calibrator.predict(raw), "label": pr[ycol].to_numpy(), "retrain_date": r})
    fr.info |= {k: str(v) for k, v in w.items()} | {"train_start": str(train_start), "excluded_to": str(exclude_to)}
    return out, fr


def reports_for(paths: Paths, variant: str) -> pd.DataFrame:
    r = pd.read_parquet(paths.processed / "stage3" / f"label_reports_{variant}.parquet", columns=["block_id", "origin", "report_id"])
    r["origin"] = pd.to_datetime(r["origin"]).dt.date
    return r


# ============================== selection (validation only) ==============================
def step_select(paths: Paths) -> dict:
    base, rep = dirs(paths)
    n_blocks = 23_448
    core = FEATURE_SETS["core"]
    allf = core + EXPERIMENTAL
    results, t0 = [], time.time()

    def val_eval(pred, variant):
        guard_selection_data(pred["origin"])
        rp = reports_for(paths, variant)
        return ev.evaluate(pred, rp[rp["origin"].between(*VAL)], n_blocks)

    def record(variant, spec, fset, rate, pred, fr, extra=None):
        m = val_eval(pred, variant)
        results.append({"variant": variant, "model": spec.name, "params": spec.params, "feature_set": fset, "rate": rate,
                        "ap": m.get("average_precision"), "p_at_1pct": m["capacity"]["top_1pct"]["precision"],
                        "brier": m.get("brier"), "prevalence": m["prevalence"], "fit_seconds": fr.info.get("seconds"),
                        "best_iteration": fr.info.get("best_iteration"), "fit_rows": fr.info["fit_rows"], **(extra or {})})
        log.info("val %-15s %-13s %-18s %s rate=%s AP=%.4f P@1%%=%.4f (%.0fs)", variant, spec.name, fset, spec.params, rate,
                 m.get("average_precision", float("nan")), results[-1]["p_at_1pct"], fr.info.get("seconds", 0))
        return m

    r = VAL[0]
    # 1. sampling-rate benchmark (fixed rule: smallest rate whose AP >= 99% of the largest-rate AP)
    bench_spec = ModelSpec("LGBM", {"num_leaves": 31, "min_data_in_leaf": 500})
    bench = {}
    for v in (OPERATIONAL, "primary"):
        for rate in SAMPLING_RATES:
            data = Data(paths, allf, rate)
            pred, fr = run_fit(bench_spec, core, v, data, r, TRAIN[0], VAL)
            m = record(v, bench_spec, "core", rate, pred, fr, {"stage": "sampling_benchmark"})
            bench[(v, rate)] = m["average_precision"]
    best = {v: max(bench[(v, x)] for x in SAMPLING_RATES) for v in (OPERATIONAL, "primary")}
    rate = min(x for x in SAMPLING_RATES if all(bench[(v, x)] >= 0.99 * best[v] for v in best))
    data = Data(paths, allf, rate)
    chosen = {"sampling_rate": rate}
    # 2. per-variant grids
    for v in MAIN_VARIANTS:
        hold = data.rows(v, windows_for(r)["holdout_from"], windows_for(r)["last_matured_origin"])
        prows = data.rows(v, *VAL)
        cand = {}
        for name, grid, key in (("B1_hist_rate", B1_GRID, "prior_weeks"), ("B2_recency", B2_GRID, "half_life")):
            for g in grid:
                spec = ModelSpec(name, {key: g})
                pred, fr = run_fit(spec, core, v, data, r, TRAIN[0], VAL, hold_rows=hold, pred_rows=prows)
                cand.setdefault(name, []).append((record(v, spec, "core", rate, pred, fr, {"stage": "grid"})["average_precision"], spec.params))
        spec = ModelSpec("B0_global", {})
        pred, fr = run_fit(spec, core, v, data, r, TRAIN[0], VAL, hold_rows=hold, pred_rows=prows)
        record(v, spec, "core", rate, pred, fr, {"stage": "grid"})
        # Selection round 2 (revision made on validation evidence, before any test prediction existed): the feature
        # set is crossed with hyper-parameters for the learned models, because round 1 showed week-level weather
        # features degrading LightGBM (outputs/stage4/validation_results_v1_core_only.csv).
        for fset, feats in FEATURE_SETS.items():
            for C in LR_GRID:
                spec = ModelSpec("LR", {"C": C})
                pred, fr = run_fit(spec, feats, v, data, r, TRAIN[0], VAL, hold_rows=hold, pred_rows=prows)
                ap = record(v, spec, fset, rate, pred, fr, {"stage": "grid"})["average_precision"]
                cand.setdefault("LR", []).append((ap, {"params": spec.params, "feature_set": fset}))
            for g in LGBM_GRID:
                spec = ModelSpec("LGBM", dict(g))
                pred, fr = run_fit(spec, feats, v, data, r, TRAIN[0], VAL, hold_rows=hold, pred_rows=prows)
                ap = record(v, spec, fset, rate, pred, fr, {"stage": "grid"})["average_precision"]
                cand.setdefault("LGBM", []).append((ap, {"params": spec.params, "feature_set": fset}))
        chosen[v] = {}
        for k, lst in cand.items():
            best_ = max(lst, key=lambda z: z[0])[1]
            chosen[v][k] = best_ if k in ("LR", "LGBM") else {"params": best_, "feature_set": None}
        chosen[v]["B0_global"] = {"params": {}, "feature_set": None}
        del hold, prows
    # 3. validation ablations + reduced history for the selected LightGBM (reported, not used for re-selection)
    for v in (OPERATIONAL, "primary"):
        c = chosen[v]["LGBM"]
        spec = ModelSpec("LGBM", c["params"])
        sel_feats = FEATURE_SETS[c["feature_set"]]
        pred, fr = run_fit(spec, sel_feats + EXPERIMENTAL, v, data, r, TRAIN[0], VAL)
        record(v, spec, f"{c['feature_set']}_plus_experimental", rate, pred, fr, {"stage": "ablation"})
        for label, start, excl in (("train_from_2021-05-03", D(2021, 5, 3), None),
                                   ("exclude_origins_to_2022-05-03", TRAIN[0], D(2022, 5, 3))):
            pred, fr = run_fit(spec, sel_feats, v, data, r, start, VAL, exclude_to=excl)
            record(v, spec, c["feature_set"], rate, pred, fr, {"stage": "reduced_history", "history": label})
    res = pd.DataFrame(results)
    res.to_csv(rep / "validation_results.csv", index=False)
    selection = {"frozen_utc": utc_iso(), "selection_data": {"train": [str(x) for x in TRAIN], "validation": [str(x) for x in VAL],
                                                             "test_start": str(TEST_START)},
                 "rule": "per variant and model family: max validation average precision over hyper-parameters x feature "
                         "sets (round 2); sampling rate: smallest rate with AP >= 99% of the best tested rate for both "
                         "multi_block and primary",
                 "revision_note": "Round 1 (selection_v1_core_only.json) pre-registered the 'core' feature set; validation "
                                  "showed week-level weather features degrading LightGBM, so round 2 selects the feature "
                                  "set on validation as well. No test-period prediction existed at either round.",
                 "chosen": chosen, "feature_sets": FEATURE_SETS, "operational_target": OPERATIONAL,
                 "elapsed_seconds": round(time.time() - t0)}
    write_json(base / "selection.json", selection)
    selection["selection_sha256"] = sha256_file(base / "selection.json")
    return selection


# ============================== walk-forward test ==============================
def experiments(sel: dict) -> list[dict]:
    """Walk-forward experiments derived ONLY from the frozen validation selection."""
    ch = sel["chosen"]
    ex = []

    def add(eid, v, mname, c, fset, group, start=TRAIN[0], excl=None, extra_feats=()):
        feats = (FEATURE_SETS[fset] if fset else []) + list(extra_feats)
        ex.append({"id": eid, "variant": v, "model": mname, "params": c["params"], "features": feats,
                   "feature_set": (fset or "n/a") + ("_plus_experimental" if extra_feats else ""),
                   "train_start": str(start), "exclude_to": excl, "group": group})

    for v in MAIN_VARIANTS:
        for mname in ("B0_global", "B1_hist_rate", "B2_recency", "LR", "LGBM"):
            c = ch[v][mname]
            add(f"{v}__{mname}", v, mname, c, c["feature_set"] or ("history" if mname.startswith("B") else None), "main")
    op = ch[OPERATIONAL]["LGBM"]
    add("no_supplementary__LGBM", "no_supplementary", "LGBM", ch["primary"]["LGBM"], ch["primary"]["LGBM"]["feature_set"],
        "sensitivity_supplementary")
    for fset in FEATURE_SETS:
        if fset != op["feature_set"]:
            add(f"{OPERATIONAL}__LGBM__abl_{fset}", OPERATIONAL, "LGBM", op, fset, "ablation")
    add(f"{OPERATIONAL}__LGBM__abl_plus_experimental", OPERATIONAL, "LGBM", op, op["feature_set"], "ablation_experimental",
        extra_feats=EXPERIMENTAL)
    add(f"{OPERATIONAL}__LGBM__from_2021-05-03", OPERATIONAL, "LGBM", op, op["feature_set"], "reduced_history", start=D(2021, 5, 3))
    add(f"{OPERATIONAL}__LGBM__excl_to_2022-05-03", OPERATIONAL, "LGBM", op, op["feature_set"], "reduced_history",
        excl="2022-05-03")
    return ex


def step_test(paths: Paths, only: list[str] | None = None) -> dict:
    base, rep = dirs(paths)
    sel = read_json(base / "selection.json")
    sel_sha = sha256_file(base / "selection.json")
    end = last_origin(paths)
    sched = retrain_schedule(TEST_START, end)
    exps = [e for e in experiments(sel) if not only or e["id"] in only]
    versions = {p: metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow")}
    by_variant: dict[str, list[dict]] = {}
    for e in exps:
        by_variant.setdefault(e["variant"], []).append(e)
    done = {}
    for v, group in by_variant.items():
        feats_union = sorted({f for e in group for f in e["features"]}, key=lambda f: (f not in FEATURE_SETS["core"], f))
        data = Data(paths, feats_union, sel["chosen"]["sampling_rate"])
        preds = {e["id"]: [] for e in group}
        infos = {e["id"]: [] for e in group}
        todo = [e for e in group if not (base / "predictions" / f"{e['id']}.parquet").exists()]
        if not todo:
            continue
        for r, p_to in sched:
            w = windows_for(r)
            hold = data.rows(v, w["holdout_from"], w["last_matured_origin"])
            prows = data.rows(v, r, p_to)
            for e in todo:
                spec = ModelSpec(e["model"], dict(e["params"]))
                excl = D.fromisoformat(e["exclude_to"]) if e["exclude_to"] else None
                pred, fr = run_fit(spec, e["features"], v, data, r, D.fromisoformat(e["train_start"]), (r, p_to),
                                   exclude_to=excl, hold_rows=hold, pred_rows=prows)
                preds[e["id"]].append(pred)
                infos[e["id"]].append(fr.info)
                mdir = base / "models" / e["id"]
                mdir.mkdir(parents=True, exist_ok=True)
                if e["model"] == "LGBM":
                    fr.model.save_model(str(mdir / f"{r}.txt"), num_iteration=fr.model.best_iteration)
                elif e["model"] == "LR":
                    joblib.dump(fr.model, mdir / f"{r}.joblib")
                joblib.dump(fr.calibrator, mdir / f"{r}_calibrator.joblib")
                log.info("test %-48s retrain %s -> %s..%s rows=%d (%ss)", e["id"], r, r, p_to, len(pred), fr.info.get("seconds"))
            del hold, prows
        for e in todo:
            out = pd.concat(preds[e["id"]], ignore_index=True)
            f = base / "predictions" / f"{e['id']}.parquet"
            out.to_parquet(f, index=False)
            meta = {**{k: e[k] for k in e if k != "features"}, "feature_order": e["features"], "retrain_schedule": [[str(a), str(b)] for a, b in sched],
                    "fits": infos[e["id"]], "selection_sha256": sel_sha, "package_versions": versions,
                    "python": platform.python_version(), "predictions_sha256": sha256_file(f),
                    "features_manifest_sha256": sha256_file(base / "features_manifest.json"), "generated_utc": utc_iso()}
            write_json(base / "models" / e["id"] / "metadata.json", meta)
            done[e["id"]] = len(out)
        data.cache.clear()
    return done


# ============================== evaluation ==============================
def _pred(paths, eid):
    p = pd.read_parquet(paths.processed / "stage4" / "predictions" / f"{eid}.parquet")
    p["origin"] = pd.to_datetime(p["origin"]).dt.date
    return p


def _flat(m: dict) -> dict:
    out = {k: m.get(k) for k in ("weeks", "labelled", "unknown", "positives", "prevalence", "average_precision",
                                 "ap_lift_over_prevalence", "brier", "brier_reference_prevalence", "calibration_in_the_large_ratio",
                                 "calibration_error_weighted")}
    for k, c in m.get("capacity", {}).items():
        out[f"{k}_precision"] = c["precision"]
        out[f"{k}_recall"] = c["recall"]
        out[f"{k}_hits"] = c["hits"]
        if "report_capture_rate" in c:
            out[f"{k}_report_capture"] = c["report_capture_rate"]
    return out


def step_evaluate(paths: Paths) -> dict:
    base, rep = dirs(paths)
    sel = read_json(base / "selection.json")
    n_blocks = 23_448
    exps = {e["id"]: e for e in experiments(sel)}
    blocks = pd.read_parquet(paths.processed / "stage3" / "blocks.parquet", columns=["block_id", "l_zip"]).set_index("block_id")
    summary, monthly, rows = {}, {}, []
    for eid, e in exps.items():
        f = base / "predictions" / f"{eid}.parquet"
        if not f.exists():
            log.warning("missing predictions for %s", eid)
            continue
        p = _pred(paths, eid)
        rp = reports_for(paths, e["variant"])
        rp = rp[rp["origin"].between(TEST_START, p["origin"].max())]
        m = ev.evaluate(p, rp, n_blocks)
        stress = p[p["origin"].between(*STRESS)]
        nonstress = p[~p["origin"].between(*STRESS)]
        res = {"overall": m, "test_excluding_march_2026": ev.evaluate(nonstress, rp[~rp["origin"].between(*STRESS)], n_blocks),
               "march_2026_stress": ev.evaluate(stress, rp[rp["origin"].between(*STRESS)], n_blocks),
               "noaa_pre_2025-08-27": ev.evaluate(p[p["origin"].between(*NOAA_PRE)], rp[rp["origin"].between(*NOAA_PRE)], n_blocks),
               "noaa_post_2025-08-27": ev.evaluate(p[p["origin"].between(*NOAA_POST)], rp[rp["origin"].between(*NOAA_POST)], n_blocks)}
        mon = pd.to_datetime(p["origin"]).dt.to_period("M").astype(str)
        res["by_month"] = ev.by_group(p, rp, n_blocks, mon, min_pos=1)
        if e["group"] == "main" and e["model"] in ("LGBM", "B1_hist_rate"):
            z = blocks.reindex(p["block_id"])["l_zip"].fillna("none").to_numpy()
            res["by_zip"] = ev.by_group(p, rp, n_blocks, pd.Series(z), min_pos=40)
        summary[eid] = res
        for period in ("overall", "test_excluding_march_2026", "march_2026_stress", "noaa_pre_2025-08-27", "noaa_post_2025-08-27"):
            rows.append({"experiment": eid, "variant": e["variant"], "model": e["model"], "feature_set": e["feature_set"],
                         "group": e["group"], "period": period, **_flat(res[period])})
        monthly[eid] = res["by_month"]
        log.info("eval %-50s AP=%.4f P@1%%=%.4f", eid, m.get("average_precision", float("nan")), m["capacity"]["top_1pct"]["precision"])
    tab = pd.DataFrame(rows)
    # lift vs historical-rate baseline (same variant, period, blocks, weeks)
    b1 = tab[tab["model"] == "B1_hist_rate"].set_index(["variant", "period"])
    for s in ("top_0.5pct", "top_1pct", "top_2pct"):
        tab[f"{s}_lift_vs_B1"] = [r[f"{s}_precision"] / b1.loc[(r["variant"], r["period"]), f"{s}_precision"]
                                  if (r["variant"], r["period"]) in b1.index else np.nan for _, r in tab.iterrows()]
    tab["ap_lift_vs_B1"] = [r["average_precision"] / b1.loc[(r["variant"], r["period"]), "average_precision"]
                            if (r["variant"], r["period"]) in b1.index else np.nan for _, r in tab.iterrows()]
    tab.to_csv(rep / "test_results.csv", index=False)
    write_json(rep / "test_results_detail.json", summary)
    # cross-variant: every main LGBM evaluated on the common high-confidence subset (confident_only labels, labelled in all)
    common = None
    preds = {v: _pred(paths, f"{v}__LGBM") for v in MAIN_VARIANTS if (base / "predictions" / f"{v}__LGBM.parquet").exists()}
    if len(preds) == len(MAIN_VARIANTS):
        lab = preds["confident_only"][["block_id", "origin", "label"]].rename(columns={"label": "y_conf"})
        mask = None
        for v, p in preds.items():
            mk = p.set_index(["block_id", "origin"])["label"].notna()
            mask = mk if mask is None else mask & mk.reindex(mask.index)
        keep = mask[mask].index
        cross = []
        for v, p in preds.items():
            q = p.set_index(["block_id", "origin"]).loc[keep].reset_index().merge(lab, on=["block_id", "origin"])
            q["label"] = q["y_conf"]
            m = ev.evaluate(q, None, n_blocks)
            cross.append({"model_trained_on": v, **_flat(m)})
        common = pd.DataFrame(cross)
        common.to_csv(rep / "cross_variant_common_subset.csv", index=False)
    # plots
    for v in MAIN_VARIANTS:
        curves = {}
        for mname in ("B1_hist_rate", "B2_recency", "LR", "LGBM"):
            eid = f"{v}__{mname}"
            if eid in summary:
                p = _pred(paths, eid)
                curves[mname] = p[p["label"].notna()]
        if curves:
            ev.plot_pr(curves, f"Test PR curves - target {v}", rep / "plots" / f"pr_{v}.png")
            ev.plot_reliability(curves, f"Test reliability - target {v} (Platt on 13-week holdout)", rep / "plots" / f"reliability_{v}.png")
    op = f"{OPERATIONAL}__LGBM"
    if op in summary:
        p = _pred(paths, op)
        lab = p[p["label"].notna()]
        st = lab[lab["origin"].between(*STRESS)]
        ev.plot_reliability({"test excl. Mar 2026": lab[~lab["origin"].between(*STRESS)], "Mar 2026 stress": st},
                            "Reliability: March 2026 surge vs rest (multi_block LGBM)", rep / "plots" / "reliability_march2026.png")
    importance(paths, sel)
    write_json(rep / "evaluation_manifest.json", {"generated_utc": utc_iso(), "selection_sha256": sha256_file(base / "selection.json"),
                                                   "experiments": list(summary)})
    return {"experiments": len(summary)}


COMPARISONS = [  # (label, experiment A, experiment B) - paired on identical block-weeks and target
    ("multi_block: LGBM vs B1 historical rate", "multi_block__LGBM", "multi_block__B1_hist_rate"),
    ("multi_block: LGBM vs B2 recency", "multi_block__LGBM", "multi_block__B2_recency"),
    ("multi_block: LGBM vs LR", "multi_block__LGBM", "multi_block__LR"),
    ("primary: LGBM vs B1 historical rate", "primary__LGBM", "primary__B1_hist_rate"),
    ("primary: LGBM vs B2 recency", "primary__LGBM", "primary__B2_recency"),
    ("primary: LGBM vs LR", "primary__LGBM", "primary__LR"),
    ("confident_only: LGBM vs B2 recency", "confident_only__LGBM", "confident_only__B2_recency"),
    ("ablation: history_static (selected) vs history", "multi_block__LGBM", "multi_block__LGBM__abl_history"),
    ("ablation: history_static (selected) vs history_weather", "multi_block__LGBM", "multi_block__LGBM__abl_history_weather"),
    ("ablation: history_static (selected) vs core", "multi_block__LGBM", "multi_block__LGBM__abl_core"),
    ("ablation: selected vs + experimental (assumption-dependent)", "multi_block__LGBM", "multi_block__LGBM__abl_plus_experimental"),
]


def _weekly_topk(p: pd.DataFrame, share: float) -> pd.DataFrame:
    sel = ev.topk_select(p, share, 23_448)
    sel = sel[sel["label"].notna()]
    return sel.groupby("origin").agg(hits=("label", "sum"), n=("label", "size"))


def step_uncertainty(paths: Paths, n_boot: int = 1000, n_boot_ap: int = 200, seed: int = 11) -> dict:
    """Paired bootstrap over test WEEKS (the unit of temporal dependence) for precision@1% and average precision."""
    from sklearn.metrics import average_precision_score
    base, rep = dirs(paths)
    rng = np.random.default_rng(seed)
    out = []
    for label, a, b in COMPARISONS:
        pa, pb = _pred(paths, a), _pred(paths, b)
        wa, wb = _weekly_topk(pa, 0.01), _weekly_topk(pb, 0.01)
        weeks = np.array(sorted(set(wa.index) & set(wb.index)))
        A, B = wa.loc[weeks], wb.loc[weeks]
        idx = rng.integers(0, len(weeks), (n_boot, len(weeks)))
        pa_b = A["hits"].to_numpy()[idx].sum(1) / A["n"].to_numpy()[idx].sum(1)
        pb_b = B["hits"].to_numpy()[idx].sum(1) / B["n"].to_numpy()[idx].sum(1)
        diff = pa_b - pb_b
        la, lb = pa[pa["label"].notna()], pb[pb["label"].notna()]
        groups_a = {w: g for w, g in la.groupby("origin")}
        groups_b = {w: g for w, g in lb.groupby("origin")}
        ap_d = []
        for _ in range(n_boot_ap):
            ws = rng.choice(weeks, len(weeks))
            ga = pd.concat([groups_a[w] for w in ws])
            gb = pd.concat([groups_b[w] for w in ws])
            ap_d.append(average_precision_score(ga["label"], ga["score"]) - average_precision_score(gb["label"], gb["score"]))
        ap_d = np.array(ap_d)
        out.append({"comparison": label, "a": a, "b": b, "weeks": len(weeks),
                    "p_at_1pct_a": float(A["hits"].sum() / A["n"].sum()), "p_at_1pct_b": float(B["hits"].sum() / B["n"].sum()),
                    "p_at_1pct_diff_ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
                    "p_at_1pct_share_boot_a_better": float((diff > 0).mean()),
                    "ap_diff_ci95": [float(np.quantile(ap_d, 0.025)), float(np.quantile(ap_d, 0.975))],
                    "ap_share_boot_a_better": float((ap_d > 0).mean())})
        log.info("bootstrap %-60s dP@1%% CI %s dAP CI %s", label, np.round(out[-1]["p_at_1pct_diff_ci95"], 4), np.round(out[-1]["ap_diff_ci95"], 4))
    write_json(rep / "bootstrap_comparisons.json", {"method": f"paired bootstrap over test weeks; {n_boot} resamples (P@1%), "
                                                              f"{n_boot_ap} (AP); seed {seed}", "results": out})
    return {"comparisons": len(out)}


def importance(paths: Paths, sel: dict) -> None:
    """Gain importance and mean |SHAP| (LightGBM pred_contrib) of the final-retrain operational and primary models."""
    import lightgbm as lgb
    base, rep = dirs(paths)
    out = {}
    for v in (OPERATIONAL, "primary"):
        mdir = base / "models" / f"{v}__LGBM"
        files = sorted(mdir.glob("*.txt"))
        if not files:
            continue
        b = lgb.Booster(model_file=str(files[-1]))
        feats = b.feature_name()
        r = D.fromisoformat(files[-1].stem)
        x = load(paths, feats + ["origin"], r, r + dt.timedelta(days=28))
        contrib = b.predict(x[feats], pred_contrib=True)
        shap = np.abs(contrib[:, :-1]).mean(axis=0)
        gain = b.feature_importance("gain")
        df = pd.DataFrame({"feature": feats, "mean_abs_shap": shap, "gain": gain / gain.sum()}).sort_values("mean_abs_shap", ascending=False)
        out[v] = {"model_file": str(files[-1].relative_to(paths.root)), "rows_explained": len(x), "table": df.round(6).to_dict("records")}
    write_json(rep / "feature_importance.json", out)
