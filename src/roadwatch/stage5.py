"""Stage 5: inspection prioritisation (decision engine evaluation, geography, worklists) and the separate weekly
report-volume forecast. Reuses the persisted Stage-4 walk-forward predictions; nothing is retrained or retuned on test
outcomes. Labels are joined only AFTER selection, for evaluation.

Steps (python -I -m roadwatch stage5 <step>): compare | geography | worklists | volume | all
Single week:  python -I -m roadwatch stage5 worklist --week 2026-10-12 --capacity 0.01 [--policy ID] [--diversify]
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
import shapely
from scipy.stats import spearmanr

from . import inspection as insp
from . import volume as vol
from .config import Paths
from .conditional_features import map_to_blocks
from .stage3_diagnostics import _md_table
from .stage4 import load as load_features
from .storage import read_json, sha256_file, utc_iso, write_json

log = logging.getLogger(__name__)
D = dt.date
STRESS = (D(2026, 3, 2), D(2026, 3, 30))
TEST = (D(2025, 7, 7), D(2026, 10, 12))
FEATURE_LABELS = {
    "rep_decay_hl90d": "recent report history (90-day decay)", "rep_decay_hl30d": "recent report history (30-day decay)",
    "rep_decay_hl7d": "very recent reports (7-day decay)", "nbr_frac_365d": "reports on adjacent blocks (365 d)",
    "nbr_frac_30d": "reports on adjacent blocks (30 d)", "cell_frac_365d": "reports in surrounding 2,000 ft cell (365 d)",
    "cell_frac_90d": "reports in surrounding 2,000 ft cell (90 d)", "rep_any_365d": "reports naming the block (365 d)",
    "rep_frac_365d": "reports attributed to the block (365 d)", "rep_unique_365d": "uniquely located reports (365 d)",
    "pos_week_rate_52": "share of past weeks with a report", "days_since_last_report": "recency of the last report",
    "length_m": "block length", "woy_sin": "time of year", "woy_cos": "time of year", "bk_reports_28d": "Brooklyn-wide recent volume",
    "bk_reports_7d": "Brooklyn-wide recent volume", "n_adjacent_blocks": "number of adjoining blocks",
    "snow_priority_code": "snow-priority class", "street_width_ft": "street width", "total_lanes": "number of lanes",
    "travel_lanes": "travel lanes", "truck_route": "truck route"}


def cfg_load(paths: Paths) -> dict:
    return read_json(paths.root / "config" / "stage5.json")


def dirs(paths: Paths):
    base, rep = paths.processed / "stage5", paths.root / "outputs" / "stage5"
    for d in (base / "selections", base / "worklists", rep / "worklists", rep / "plots"):
        d.mkdir(parents=True, exist_ok=True)
    return base, rep


def zip_of_blocks(paths: Paths) -> pd.Series:
    b = pd.read_parquet(paths.processed / "stage3" / "blocks.parquet", columns=["block_id", "l_zip", "r_zip"])
    z = b["l_zip"].fillna(b["r_zip"]).fillna("unknown")
    return pd.Series(z.to_numpy(), index=b["block_id"].to_numpy(), name="zip")


def predictions(paths: Paths, policy: str) -> pd.DataFrame:
    p = pd.read_parquet(paths.processed / "stage4" / "predictions" / f"{policy}.parquet")
    p["origin"] = pd.to_datetime(p["origin"]).dt.date
    return p


def reports(paths: Paths, variant: str, period: tuple = TEST) -> pd.DataFrame:
    r = pd.read_parquet(paths.processed / "stage3" / f"label_reports_{variant}.parquet", columns=["block_id", "origin", "report_id"])
    r["origin"] = pd.to_datetime(r["origin"]).dt.date
    return r[(r["origin"] >= period[0]) & (r["origin"] <= period[1])]


def block_nodes(paths: Paths) -> pd.DataFrame:
    b = pd.read_parquet(paths.processed / "stage3" / "blocks.parquet", columns=["block_id", "node_from", "node_to"])
    return b.set_index("block_id")


def policy_rule(paths: Paths) -> dict:
    f = paths.processed / "stage5" / "policy_design.json"
    return read_json(f) if f.exists() else {"non_adjacent": False, "status": "not yet decided"}


# ============================== validation-period predictions + policy design ==============================
def step_validation_predictions(paths: Paths) -> dict:
    """Walk-forward predictions for the VALIDATION period with the frozen Stage-4 selection (no retuning), so that
    selection-policy choices are made without test outcomes."""
    from .modeling import TEST_START, ModelSpec, retrain_schedule
    from .stage4 import TRAIN, VAL, Data, experiments, run_fit
    cfg = cfg_load(paths)
    base, _ = dirs(paths)
    out_dir = base / "validation_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = read_json(paths.processed / "stage4" / "selection.json")
    exps = {e["id"]: e for e in experiments(sel)}
    sched = retrain_schedule(*VAL)
    assert sched[-1][1] < TEST_START
    done = {}
    for eid in cfg["policy_design"]["validation_policies"]:
        e = exps[eid]
        data = Data(paths, e["features"], sel["chosen"]["sampling_rate"])
        parts = []
        for r, p_to in sched:
            pred, fr = run_fit(ModelSpec(e["model"], dict(e["params"])), e["features"], e["variant"], data, r, TRAIN[0], (r, p_to))
            parts.append(pred)
            log.info("validation walk-forward %s retrain %s -> %s (%d rows)", eid, r, p_to, len(pred))
        out = pd.concat(parts, ignore_index=True)
        assert out["origin"].max() < TEST_START
        out.to_parquet(out_dir / f"{eid}.parquet", index=False)
        done[eid] = len(out)
    return done


def _capture(sel: pd.DataFrame, rp: pd.DataFrame) -> pd.DataFrame:
    """Per-week unique reports captured / total. Indexed by EVERY report week in sorted order (zero-capture weeks
    included), so paired bootstraps of two lists stay aligned (Stage-7 fix of a latent misalignment)."""
    got = rp.merge(sel[["block_id", "origin"]], on=["block_id", "origin"]).drop_duplicates(["origin", "report_id"])
    tot = rp.drop_duplicates(["origin", "report_id"]).groupby("origin").size().rename("total")
    cap = got.groupby("origin").size().rename("captured").reindex(tot.index, fill_value=0)
    return pd.concat([cap, tot], axis=1).sort_index()


def step_policy_design(paths: Paths) -> dict:
    """Apply the pre-registered rule (config policy_design) on validation weeks only, then freeze the decision."""
    from .stage4 import VAL
    cfg = cfg_load(paths)
    base, rep = dirs(paths)
    E, caps, primary = cfg["eligible_blocks"], cfg["capacities"], cfg["primary_policy"]
    nodes = block_nodes(paths)
    rp_any, rp_asg = reports(paths, "multi_block", VAL), reports(paths, "primary", VAL)
    rows, cap_any = [], {}
    for pol in cfg["policy_design"]["validation_policies"]:
        p = pd.read_parquet(base / "validation_predictions" / f"{pol}.parquet")
        p["origin"] = pd.to_datetime(p["origin"]).dt.date
        for cap in caps:
            for rule in ("top_k", "non_adjacent"):
                sel, _, _ = insp.select_all(p, cap, E, nodes=nodes if rule == "non_adjacent" else None)
                a, b = _capture(sel, rp_any), _capture(sel, rp_asg)
                cap_any[(pol, cap, rule)] = a["captured"].sum() / a["total"].sum()
                rows.append({"policy": pol, "capacity": cap, "rule": rule, "weeks": int(a.shape[0]),
                             "capture_any_candidate": cap_any[(pol, cap, rule)],
                             "capture_assigned_block": b["captured"].sum() / b["total"].sum()})
    tab = pd.DataFrame(rows)
    tab.to_csv(rep / "policy_design_validation.csv", index=False)
    adopt = all(cap_any[(primary, c, "non_adjacent")] > cap_any[(primary, c, "top_k")] for c in caps)
    strict_better = all(cap_any[("primary__LGBM", c, "top_k")] > cap_any[(primary, c, "top_k")] for c in caps)
    decision = {"decided_utc": utc_iso(), "data": f"validation weeks {VAL[0]}..{VAL[1]} only",
                "rule": cfg["policy_design"]["non_adjacent_rule"], "non_adjacent": bool(adopt),
                "strict_list_captures_more_at_every_capacity_on_validation": bool(strict_better),
                "primary_ranking_model": primary, "validation_table": tab.round(5).to_dict("records")}
    write_json(base / "policy_design.json", decision)
    return {"non_adjacent": adopt, "strict_better_on_validation": strict_better}


def step_policy_test(paths: Paths) -> dict:
    """Report the decided policy (and the alternatives, clearly labelled) on the TEST weeks."""
    cfg = cfg_load(paths)
    base, rep = dirs(paths)
    E, caps = cfg["eligible_blocks"], cfg["capacities"]
    nodes = block_nodes(paths)
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    nb = cfg["bootstrap"]["resamples"]
    rp_any, rp_asg = reports(paths, "multi_block"), reports(paths, "primary")
    res = {}
    rows = []
    for pol in cfg["policy_design"]["validation_policies"]:
        p = predictions(paths, pol)
        for cap in caps:
            for rule in ("top_k", "non_adjacent"):
                sel, _, _ = insp.select_all(p, cap, E, nodes=nodes if rule == "non_adjacent" else None)
                if rule == "non_adjacent":
                    sel.to_parquet(base / "selections" / f"{pol}__{_cap_name(cap)}__nonadjacent.parquet", index=False)
                res[(pol, cap, rule)] = (_capture(sel, rp_any), _capture(sel, rp_asg))
    for (pol, cap, rule), (a, b) in res.items():
        ref_a, ref_b = res[(cfg["primary_policy"], cap, "top_k")]
        row = {"policy": pol, "capacity": cap, "rule": rule,
               "capture_any_candidate": a["captured"].sum() / a["total"].sum(),
               "capture_assigned_block": b["captured"].sum() / b["total"].sum()}
        if (pol, rule) != (cfg["primary_policy"], "top_k"):
            for nm, x, y in (("any", a, ref_a), ("assigned", b, ref_b)):
                row[f"diff_vs_primary_topk_{nm}_ci95"] = insp.bootstrap_ratio(
                    x["captured"].to_numpy(float), x["total"].to_numpy(float), rng, nb,
                    y["captured"].to_numpy(float), y["total"].to_numpy(float))
        rows.append(row)
    pd.DataFrame(rows).to_csv(rep / "policy_design_test.csv", index=False)
    # full metrics of the decided policy (labels joined only after selection)
    pol, rule = cfg["primary_policy"], policy_rule(paths)
    p = predictions(paths, pol)
    summ = {}
    for cap in caps:
        f = base / "selections" / f"{pol}__{_cap_name(cap)}{'__nonadjacent' if rule.get('non_adjacent') else ''}.parquet"
        w = insp.weekly_outcomes(pd.read_parquet(f), p, rp_any)
        w.to_parquet(base / "selections" / f"{f.stem}__weekly_outcomes.parquet", index=False)
        for period, m in (("overall", np.ones(len(w), bool)), ("excluding_march_2026", ~w["origin"].between(*STRESS).to_numpy()),
                          ("march_2026_stress", w["origin"].between(*STRESS).to_numpy())):
            s = insp.summarize(w[m])
            s["precision_ci95"] = insp.bootstrap_ratio(w[m]["hits"].to_numpy(float), w[m]["labelled_selected"].to_numpy(float), rng, nb)
            s["capture_ci95"] = insp.bootstrap_ratio(w[m]["reports_captured"].to_numpy(float), w[m]["reports_total"].to_numpy(float), rng, nb)
            summ[f"{_cap_name(cap)}|{period}"] = s
    write_json(rep / "decided_policy_test_summary.json", {"policy": pol, "non_adjacent": rule.get("non_adjacent"),
                                                          "outcome": "multi_block labels; unique reports via any candidate block",
                                                          "results": summ})
    return {"rows": len(rows)}


def _cap_name(c) -> str:
    if isinstance(c, (int, np.integer)) and not isinstance(c, bool):
        return f"K{int(c)}"                 # explicit slot count (manually configured capacity)
    return f"{c * 100:g}pct"


# ============================== policy comparison ==============================
def step_compare(paths: Paths) -> dict:
    cfg = cfg_load(paths)
    base, rep = dirs(paths)
    E, caps = cfg["eligible_blocks"], cfg["capacities"]
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    nb = cfg["bootstrap"]["resamples"]
    zips = zip_of_blocks(paths)
    s4 = pd.read_csv(paths.root / "outputs" / "stage4" / "test_results.csv")
    ap = s4[s4["period"] == "overall"].set_index("experiment")["average_precision"]
    weekly, rows, monthly, excl_rows = {}, [], [], []
    rep_cache = {}
    for pol, meta in cfg["policies"].items():
        v = meta["target"]
        p = predictions(paths, pol)
        rep_cache.setdefault(v, reports(paths, v))
        for cap in caps:
            sel, ex, summ = insp.select_all(p, cap, E)
            assert (summ["selected"] == summ["k"]).all() and not sel.duplicated(["block_id", "origin"]).any()
            sel.to_parquet(base / "selections" / f"{pol}__{_cap_name(cap)}.parquet", index=False)
            w = insp.weekly_outcomes(sel, p, rep_cache[v])
            weekly[(pol, cap)] = w
            excl_rows.append({"policy": pol, "capacity": cap, "excluded_block_weeks": len(ex)})
            for period, mask in (("overall", np.ones(len(w), bool)),
                                 ("excluding_march_2026", ~w["origin"].between(*STRESS).to_numpy()),
                                 ("march_2026_stress", w["origin"].between(*STRESS).to_numpy())):
                ww = w[mask]
                s = insp.summarize(ww)
                s["precision_ci95"] = insp.bootstrap_ratio(ww["hits"].to_numpy(float), ww["labelled_selected"].to_numpy(float), rng, nb)
                s["capture_ci95"] = insp.bootstrap_ratio(ww["reports_captured"].to_numpy(float), ww["reports_total"].to_numpy(float), rng, nb)
                rows.append({"policy": pol, "target": v, "role": meta["role"], "capacity": cap, "period": period,
                             "k_per_week": int(w["selected"].iloc[0]), **s,
                             "average_precision": ap.get(pol) if period == "overall" else None})
            mon = w.assign(month=pd.to_datetime(w["origin"]).dt.to_period("M").astype(str)).groupby("month").sum(numeric_only=True)
            for m_, r_ in mon.iterrows():
                monthly.append({"policy": pol, "target": v, "capacity": cap, "month": m_,
                                "precision": r_["hits"] / r_["labelled_selected"], "recall": r_["hits"] / r_["positives"],
                                "reports_captured": int(r_["reports_captured"]), "reports_total": int(r_["reports_total"]),
                                "capture_rate": r_["reports_captured"] / r_["reports_total"]})
            log.info("compare %-28s cap %-5s precision=%.4f capture=%.4f", pol, cap, rows[-3]["precision"], rows[-3]["report_capture_rate"])
    tab = pd.DataFrame(rows)
    # lift vs the historical-rate baseline of the SAME target, paired CIs on identical weeks
    for i, r in tab.iterrows():
        ref = cfg["lift_reference"][r["target"]]
        b = tab[(tab["policy"] == ref) & (tab["capacity"] == r["capacity"]) & (tab["period"] == r["period"])].iloc[0]
        tab.loc[i, "precision_lift_vs_B1"] = r["precision"] / b["precision"]
        tab.loc[i, "capture_lift_vs_B1"] = r["report_capture_rate"] / b["report_capture_rate"]
        if r["policy"] != ref:
            wa, wb = weekly[(r["policy"], r["capacity"])], weekly[(ref, r["capacity"])]
            m = {"overall": np.ones(len(wa), bool), "excluding_march_2026": ~wa["origin"].between(*STRESS).to_numpy(),
                 "march_2026_stress": wa["origin"].between(*STRESS).to_numpy()}[r["period"]]
            wa, wb = wa[m], wb[m]
            assert (wa["origin"].to_numpy() == wb["origin"].to_numpy()).all()
            tab.at[i, "precision_diff_vs_B1_ci95"] = json.dumps(np.round(insp.bootstrap_ratio(
                wa["hits"].to_numpy(float), wa["labelled_selected"].to_numpy(float), rng, nb,
                wb["hits"].to_numpy(float), wb["labelled_selected"].to_numpy(float)), 4).tolist())
            tab.at[i, "capture_diff_vs_B1_ci95"] = json.dumps(np.round(insp.bootstrap_ratio(
                wa["reports_captured"].to_numpy(float), wa["reports_total"].to_numpy(float), rng, nb,
                wb["reports_captured"].to_numpy(float), wb["reports_total"].to_numpy(float)), 4).tolist())
    tab.to_csv(rep / "policy_comparison.csv", index=False)
    pd.DataFrame(monthly).to_csv(rep / "policy_monthly.csv", index=False)
    pd.DataFrame(excl_rows).to_csv(rep / "excluded_blocks.csv", index=False)
    for (pol, cap), w in weekly.items():
        w.assign(policy=pol, capacity=cap).to_parquet(base / "selections" / f"{pol}__{_cap_name(cap)}__weekly_outcomes.parquet", index=False)

    # common-outcome comparison: lists from different targets scored on the SAME report outcome definitions
    common = []
    outcome_defs = {"any_candidate_block (multi_block reports)": rep_cache["multi_block"],
                    "assigned_block (primary reports)": rep_cache["primary"]}
    for pol in ("multi_block__LGBM", "primary__LGBM", "multi_block__LR", "multi_block__B1_hist_rate", "primary__B1_hist_rate"):
        for cap in caps:
            sel = pd.read_parquet(base / "selections" / f"{pol}__{_cap_name(cap)}.parquet")
            for oname, rp in outcome_defs.items():
                got = rp.merge(sel[["block_id", "origin"]], on=["block_id", "origin"]).drop_duplicates(["origin", "report_id"])
                tot = rp.drop_duplicates(["origin", "report_id"])
                common.append({"list_from_policy": pol, "capacity": cap, "outcome_definition": oname,
                               "reports_captured": len(got), "reports_total": len(tot), "capture_rate": len(got) / len(tot)})
    pd.DataFrame(common).to_csv(rep / "common_outcome_comparison.csv", index=False)

    # optional ZIP diversification of the primary policy (pre-specified multiple; not tuned on outcomes)
    m = cfg["diversification"]["max_multiple_of_block_share"]
    pol = cfg["primary_policy"]
    p = predictions(paths, pol)
    div = []
    for cap in caps:
        sel, _, summ = insp.select_all(p, cap, E, zip_of=zips, max_multiple=m)
        sel.to_parquet(base / "selections" / f"{pol}__{_cap_name(cap)}__zipcap{m:g}.parquet", index=False)
        w = insp.weekly_outcomes(sel, p, rep_cache["multi_block"])
        un = weekly[(pol, cap)]
        s, su = insp.summarize(w), insp.summarize(un)
        sel_u = pd.read_parquet(base / "selections" / f"{pol}__{_cap_name(cap)}.parquet")
        cz_u = insp.concentration(zips.reindex(sel_u["block_id"]).value_counts())
        cz_d = insp.concentration(zips.reindex(sel["block_id"]).value_counts())
        div.append({"capacity": cap, "policy": f"{pol} + ZIP cap {m:g}x block share",
                    "precision": s["precision"], "precision_unconstrained": su["precision"],
                    "reports_captured": s["reports_captured"], "reports_captured_unconstrained": su["reports_captured"],
                    "capture_rate": s["report_capture_rate"], "capture_rate_unconstrained": su["report_capture_rate"],
                    "capture_diff_ci95": insp.bootstrap_ratio(w["reports_captured"].to_numpy(float), w["reports_total"].to_numpy(float), rng, nb,
                                                             un["reports_captured"].to_numpy(float), un["reports_total"].to_numpy(float)),
                    "mean_blocks_displaced_per_week": float(summ["div_displaced_from_unconstrained_top_k"].mean()),
                    "zips_selected_total": cz_d["units_with_any"], "zips_selected_total_unconstrained": cz_u["units_with_any"],
                    "top5_zip_share": cz_d["top5_share"], "top5_zip_share_unconstrained": cz_u["top5_share"],
                    "hhi": cz_d["hhi"], "hhi_unconstrained": cz_u["hhi"]})
    write_json(rep / "diversification.json", {"rationale": cfg["diversification"]["rationale"], "results": div})
    return {"rows": len(tab)}


# ============================== geography ==============================
def step_geography(paths: Paths) -> dict:
    cfg = cfg_load(paths)
    base, rep = dirs(paths)
    pol = cfg["primary_policy"]
    zips = zip_of_blocks(paths)
    blocks = gpd.read_parquet(paths.processed / "stage3" / "blocks.parquet")
    g = pd.DataFrame({"zip": zips.reindex(blocks["block_id"]).to_numpy(), "km": blocks["length_m"].to_numpy() / 1000})
    per = g.groupby("zip").agg(blocks=("km", "size"), km=("km", "sum"))
    per["block_share"] = per["blocks"] / per["blocks"].sum()
    # test-period reports, each counted ONCE via its primary (assigned) block -> no double counting across candidates
    rp = reports(paths, "primary").drop_duplicates(["origin", "report_id"])
    per["test_reports"] = rp.assign(zip=zips.reindex(rp["block_id"]).to_numpy()).groupby("zip").size()
    for cap in cfg["capacities"]:
        for tag, f in (("", f"{pol}__{_cap_name(cap)}.parquet"), ("_zipcap", f"{pol}__{_cap_name(cap)}__zipcap{cfg['diversification']['max_multiple_of_block_share']:g}.parquet")):
            sel = pd.read_parquet(base / "selections" / f)
            per[f"sel_{_cap_name(cap)}{tag}"] = zips.reindex(sel["block_id"]).value_counts()
    per = per.fillna(0)
    per["report_share"] = per["test_reports"] / per["test_reports"].sum()
    per["sel_share_1pct"] = per["sel_1pct"] / per["sel_1pct"].sum()
    per["selection_to_block_ratio"] = per["sel_share_1pct"] / per["block_share"]
    per["report_to_block_ratio"] = per["report_share"] / per["block_share"]
    per["reports_per_km"] = per["test_reports"] / per["km"]
    # complaint-independent indicators (pre-test window 2024-07-01..2025-06-30, so not test outcomes)
    a, b = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-06-30")
    bl = blocks.sort_values("block_id").reset_index(drop=True)
    pav = gpd.read_parquet(paths.processed / "pavement.parquet")
    pav = pav[pav["rating_value"].notna() & pav["inspection_local"].between(a, b)]
    pb = map_to_blocks(pav.geometry, pav["oftcode"], bl)
    pav = pav.assign(zip=np.where(pb >= 0, zips.reindex(bl["block_id"].to_numpy()[np.clip(pb, 0, None)]).to_numpy(), None))
    per["mean_pavement_rating_2024_25"] = pav[pav["zip"].notna()].groupby("zip")["rating_value"].mean()
    per["share_rated_poor_le3"] = pav[pav["zip"].notna()].assign(p=pav["rating_value"] <= 3).groupby("zip")["p"].mean()
    wo = gpd.read_parquet(paths.processed / "workorders.parquet")
    wo = wo[wo["rptdate_date"].between(a, b)]
    wb = map_to_blocks(wo.geometry, wo["oft"], bl)
    wo = wo.assign(zip=np.where(wb >= 0, zips.reindex(bl["block_id"].to_numpy()[np.clip(wb, 0, None)]).to_numpy(), None))
    for src in ("YRD", "CTZ"):
        per[f"workorders_{src}_per_km_2024_25"] = wo[(wo["source"] == src) & wo["zip"].notna()].groupby("zip").size() / per["km"]
    pre = pd.read_parquet(paths.processed / "stage3" / "label_reports_primary.parquet")
    pre["origin"] = pd.to_datetime(pre["origin"])
    pre = pre[pre["origin"].between(a, b)].drop_duplicates(["origin", "report_id"])
    per["reports_per_km_2024_25"] = pre.assign(zip=zips.reindex(pre["block_id"]).to_numpy()).groupby("zip").size() / per["km"]
    per = per[per.index != "unknown"].fillna({"workorders_YRD_per_km_2024_25": 0, "workorders_CTZ_per_km_2024_25": 0})
    big = per[per["blocks"] >= 100]
    corr = {}
    for col in ("mean_pavement_rating_2024_25", "share_rated_poor_le3", "workorders_YRD_per_km_2024_25", "workorders_CTZ_per_km_2024_25"):
        ok = big[["reports_per_km_2024_25", col]].dropna()
        rho, pval = spearmanr(ok["reports_per_km_2024_25"], ok[col])
        corr[col] = {"spearman_rho_vs_311_reports_per_km": float(rho), "p_value": float(pval), "zips": int(len(ok))}
    conc = {}
    for c in [x for x in per.columns if x.startswith("sel_") and not x.startswith("sel_share")] + ["test_reports", "blocks"]:
        conc[c] = insp.concentration(per[c])
    per.round(5).reset_index().to_csv(rep / "geographic_coverage_by_zip.csv", index=False)
    out = {"zips": int(len(per)), "concentration": conc, "correlations_zip_level_2024_25": corr,
           "note": "311 report rates mix road condition with reporting propensity; pavement ratings and DOT-yard (YRD) "
                   "work orders are complaint-independent but imperfect indicators (inspection cycles, crew routing)."}
    write_json(rep / "geographic_coverage.json", out)
    return {"zips": len(per)}


# ============================== worklists ==============================
def _explain(paths: Paths, pol: str, rows: pd.DataFrame, feats: list[str], top_n: int) -> pd.DataFrame:
    """Per selected block: top-N positive LightGBM contributions (pred_contrib) of the model that made the prediction."""
    mdir = paths.processed / "stage4" / "models" / pol
    out = []
    for r_date, g in rows.groupby("retrain_date"):
        booster = lgb.Booster(model_file=str(mdir / f"{r_date}.txt"))
        contrib = booster.predict(g[booster.feature_name()], pred_contrib=True)[:, :-1]
        names = booster.feature_name()
        for i, (bid, o) in enumerate(zip(g["block_id"], g["origin"])):
            top = np.argsort(-contrib[i])[:top_n]
            out.append({"block_id": bid, "origin": o,
                        "model_drivers": "; ".join(f"{FEATURE_LABELS.get(names[j], names[j])} (+{contrib[i, j]:.2f})"
                                                    for j in top if contrib[i, j] > 0)})
    return pd.DataFrame(out)


def build_worklists(paths: Paths, pol: str, cap: float, origins: list | None = None, diversify: bool = False,
                    non_adjacent: bool | None = None) -> pd.DataFrame:
    cfg = cfg_load(paths)
    base, _ = dirs(paths)
    E = cfg["eligible_blocks"]
    p = predictions(paths, pol)
    if origins:
        p = p[p["origin"].isin(origins)]
    if p.empty:
        raise insp.SelectionError(f"no saved predictions for {pol} at {origins}")
    zips = zip_of_blocks(paths)
    m = cfg["diversification"]["max_multiple_of_block_share"] if diversify else None
    rule = policy_rule(paths)
    use_nonadj = rule.get("non_adjacent") if non_adjacent is None else non_adjacent   # explicit override (Stage 7)
    nodes = block_nodes(paths) if (use_nonadj and not diversify) else None
    sel, excl, _ = insp.select_all(p, cap, E, zip_of=zips if diversify else None, max_multiple=m, nodes=nodes)
    sel = sel.merge(p[["block_id", "origin", "prob", "retrain_date"]], on=["block_id", "origin"])
    meta = read_json(paths.processed / "stage4" / "models" / pol / "metadata.json")
    feats = meta["feature_order"]
    hist = ["rep_any_30d", "rep_any_365d", "rep_frac_365d", "days_since_last_report", "no_report_in_365d", "nbr_frac_365d"]
    X = load_features(paths, ["block_id", "origin"] + sorted(set(feats + hist)), min(sel["origin"]), max(sel["origin"]))
    X["origin"] = pd.to_datetime(X["origin"]).dt.date
    rows = sel.merge(X, on=["block_id", "origin"], how="left")
    if meta["model"] == "LGBM":
        rows = rows.merge(_explain(paths, pol, rows, feats, cfg["worklist"]["explain_top_n_features"]), on=["block_id", "origin"], how="left")
    else:
        rows["model_drivers"] = "n/a (non-tree model)"
    blocks = pd.read_parquet(paths.processed / "stage3" / "blocks.parquet",
                             columns=["block_id", "full_street_name", "l_zip", "r_zip", "length_m", "number_travel_lanes_num",
                                      "streetwidth_num", "truck_route_type", "bike_lane", "mid_lat", "mid_lon", "geometry_wgs84_wkb"])
    rows = rows.merge(blocks[[c for c in blocks.columns if c == "block_id" or c not in rows.columns]], on="block_id", how="left")
    k = rows.groupby("origin")["rank"].transform("max")
    tiers = cfg["worklist"]["priority_tiers"]
    rows["priority"] = np.where(rows["rank"] <= np.ceil(tiers["P1"] * k), "P1", np.where(rows["rank"] <= np.ceil(tiers["P2"] * k), "P2", "P3"))
    sel_sha = sha256_file(paths.processed / "stage4" / "selection.json")[:12]
    ext = {"LGBM": ".txt", "LR": ".joblib"}.get(meta["model"])
    version = {}
    for r in rows["retrain_date"].unique():
        mf = paths.processed / "stage4" / "models" / pol / f"{r}{ext}" if ext else None
        version[r] = (f"{pol} | retrain {r} | model sha256 {sha256_file(mf)[:12]} | stage4 selection {sel_sha}"
                      if mf is not None and mf.exists() else f"{pol} | retrain {r} | stage4 selection {sel_sha}")
    rows["model_version"] = rows["retrain_date"].map(version)
    rows["rationale"] = [
        (f"{int(a)} report(s) naming this block or its end intersections in the 365 days before the data cutoff "
         f"({int(b)} in the last 30 days); " + (f"last report {int(d)} day(s) before cutoff; " if not nr else "none in 365 days; ")
         + f"about {n:.0f} report(s) attributed to adjoining blocks in 365 days; block length {l:.0f} m.")
        for a, b, d, nr, n, l in zip(rows["rep_any_365d"], rows["rep_any_30d"], rows["days_since_last_report"],
                                     rows["no_report_in_365d"].astype(bool), rows["nbr_frac_365d"], rows["length_m"])]
    rows["recommended_action"] = "INSPECT (not a repair order; a predicted 311 report is not a confirmed defect)"
    rows["capacity_scenario"] = (f"{int(cap)} inspection slots (manually configured; experimental, not an NYC budget)"
                                 if isinstance(cap, (int, np.integer)) else
                                 f"{_cap_name(cap)} of {E} eligible blocks per week (experimental, not an NYC budget)")
    rows["diversified"] = diversify
    rows["selection_rule"] = (("top-K + non-adjacent (validation-decided)" if non_adjacent is None else "top-K + non-adjacent")
                              if nodes is not None else ("top-K + ZIP cap" if diversify else "top-K"))
    rows["geometry_wkt_wgs84"] = shapely.to_wkt(shapely.from_wkb(rows["geometry_wgs84_wkb"].to_numpy()), rounding_precision=6)
    rows["zip"] = rows["l_zip"].fillna(rows["r_zip"])
    cols = ["origin", "priority", "rank", "block_id", "full_street_name", "zip", "score", "prob", "rep_any_365d", "rep_any_30d",
            "days_since_last_report", "nbr_frac_365d", "length_m", "number_travel_lanes_num", "streetwidth_num", "truck_route_type",
            "bike_lane", "rationale", "model_drivers", "recommended_action", "capacity_scenario", "diversified",
            "selection_rule", "model_version",
            "mid_lat", "mid_lon", "geometry_wkt_wgs84"]
    out = rows[cols].rename(columns={"origin": "forecast_week_start", "score": "model_score_relative",
                                     "prob": "platt_probability_ranking_only", "rep_any_365d": "reports_365d",
                                     "rep_any_30d": "reports_30d", "nbr_frac_365d": "adjacent_block_reports_365d",
                                     "number_travel_lanes_num": "travel_lanes", "streetwidth_num": "street_width_ft"})
    return out.sort_values(["forecast_week_start", "rank"]).reset_index(drop=True), excl


def step_worklists(paths: Paths) -> dict:
    cfg = cfg_load(paths)
    base, rep = dirs(paths)
    pol = cfg["primary_policy"]
    done = {}
    for cap in cfg["capacities"]:
        wl, excl = build_worklists(paths, pol, cap)
        f = base / "worklists" / f"{pol}__{_cap_name(cap)}.parquet"
        wl.to_parquet(f, index=False)
        done[_cap_name(cap)] = {"rows": len(wl), "weeks": int(wl["forecast_week_start"].nunique()), "sha256": sha256_file(f),
                                "excluded_block_weeks_without_valid_score": len(excl),
                                "selection_rule": wl["selection_rule"].iloc[0]}
        if cfg["worklist"]["export_csv_weeks_for_primary_policy"] and cap == cfg["worklist"]["export_capacity_for_csv"]:
            for o, g in wl.groupby("forecast_week_start"):
                g.drop(columns=["geometry_wkt_wgs84"]).to_csv(rep / "worklists" / f"worklist_{o}_{_cap_name(cap)}.csv", index=False)
    write_json(base / "worklists" / "manifest.json", {"generated_utc": utc_iso(), "policy": pol, "outputs": done,
                                                      "stage4_selection_sha256": sha256_file(paths.processed / "stage4" / "selection.json")})
    return done


def single_worklist(paths: Paths, week: D, cap: float, pol: str | None, diversify: bool, rule: str = "config") -> list:
    """rule: 'config' = policy_design.json decides the non-adjacent rule (Stage-5 behaviour); 'topk' / 'nonadjacent' override
    it explicitly (e.g. --policy primary__LGBM --rule topk reproduces the dashboard's default strict top-K list)."""
    cfg = cfg_load(paths)
    _, rep = dirs(paths)
    pol = pol or cfg["primary_policy"]
    nonadj = {"config": None, "topk": False, "nonadjacent": True}[rule]
    wl, excl = build_worklists(paths, pol, cap, origins=[week], diversify=diversify, non_adjacent=nonadj)
    if len(excl):
        log.warning("%d block(s) excluded for %s: no valid score", len(excl), week)
    stem = f"worklist_{week}_{pol}_{_cap_name(cap)}{'_zipcap' if diversify else ''}{'' if rule == 'config' else '_' + rule}"
    wl.drop(columns=["geometry_wkt_wgs84"]).to_csv(rep / "worklists" / f"{stem}.csv", index=False)
    gdf = gpd.GeoDataFrame(wl.drop(columns=["geometry_wkt_wgs84"]).astype({"forecast_week_start": str}),
                           geometry=gpd.GeoSeries.from_wkt(wl["geometry_wkt_wgs84"]), crs="EPSG:4326")
    gdf.to_file(rep / "worklists" / f"{stem}.geojson", driver="GeoJSON")
    return [rep / "worklists" / f"{stem}.csv", rep / "worklists" / f"{stem}.geojson"]


# ============================== weekly volume forecast ==============================
def step_volume(paths: Paths) -> dict:
    cfg = cfg_load(paths)
    vc = cfg["volume"]
    base, rep = dirs(paths)
    a = pd.read_parquet(paths.processed / "stage3" / "report_assignments.parquet",
                        columns=["created_date_local", "brooklyn_evidence", "origin"])
    a = a[a["brooklyn_evidence"] != "none"]
    weeks = pd.read_parquet(paths.processed / "stage3" / "label_weeks.parquet")
    noaa = pd.read_parquet(paths.processed / "noaa.parquet")
    f = vol.build_frame(a["created_date_local"].dt.date, weeks, noaa)
    # consistency: our counts reproduce the Stage-3 weekly totals
    chk = a.groupby(pd.to_datetime(a["origin"]).dt.date).size()
    assert all(chk.get(o, 0) == y for o, y in zip(f["origin"], f["y"])), "weekly totals disagree with Stage 3"
    f.to_parquet(base / "volume_frame.parquet", index=False)
    train_from = D.fromisoformat(vc["train_from"])
    obs = f["week_status"] == "observable"

    def train_rows(r: D):
        return f[obs & (f["origin"] >= train_from) & (f["origin"] <= r - dt.timedelta(days=9))]

    # ---- selection on validation only (single fit at 2025-01-06) ----
    v0, v1 = (D.fromisoformat(x) for x in vc["validation"])
    if f.loc[(f["origin"] >= v0) & (f["origin"] <= v1), "origin"].max() >= TEST[0]:
        raise RuntimeError("validation overlaps test")
    tr, va = train_rows(v0), f[obs & f["origin"].between(v0, v1)]
    val = []
    for name in vol.BASELINES:
        val.append({"model": name, "feature_set": None, "alpha": None, **vol.errors(va["y"].to_numpy(), vol.baseline_forecast(va, name))})
    for fs in vc["feature_sets"]:
        for al in vc["alpha_grid"]:
            m = vol.fit_glm(tr, vol.FEATURES[fs], al)
            val.append({"model": "poisson_glm", "feature_set": fs, "alpha": al, **vol.errors(va["y"].to_numpy(), m.predict(va[vol.FEATURES[fs]]))})
    val = pd.DataFrame(val)
    val.to_csv(rep / "volume_validation.csv", index=False)
    glm = val[val["model"] == "poisson_glm"].sort_values("mae").iloc[0]
    best_base = val[val["model"].isin(vol.BASELINES)].sort_values("mae").iloc[0]
    selection = {"frozen_utc": utc_iso(), "rule": "min validation MAE among Poisson GLM feature sets x alpha; baseline = min-MAE baseline",
                 "glm_feature_set": glm["feature_set"], "glm_alpha": float(glm["alpha"]), "glm_val_mae": float(glm["mae"]),
                 "reference_baseline": best_base["model"], "baseline_val_mae": float(best_base["mae"])}
    write_json(base / "volume_selection.json", selection)
    # ---- walk-forward test (monthly retrains, same schedule as Stage 4) ----
    from .modeling import retrain_schedule
    sched = retrain_schedule(*TEST)
    preds = []
    q, last_n = tuple(vc["interval_quantiles"]), vc["interval_calibration_weeks"]
    for r, p_to in sched:
        trw = train_rows(r)
        te = f[f["origin"].between(r, p_to)]
        out = te[["origin", "y", "week_status"]].copy()
        for fs in vc["feature_sets"]:
            m = vol.fit_glm(trw, vol.FEATURES[fs], selection["glm_alpha"])
            out[f"glm_{fs}"] = m.predict(te[vol.FEATURES[fs]])
            if fs == selection["glm_feature_set"]:
                lo, hi = vol.interval(m, trw, vol.FEATURES[fs], q, last_n)
                out["pred"], out["lo"], out["hi"] = out[f"glm_{fs}"], out[f"glm_{fs}"] * lo, out[f"glm_{fs}"] * hi
        for name in vol.BASELINES:
            out[name] = vol.baseline_forecast(te, name)
        hist = trw.tail(52)["y"]
        out["p25_last52"], out["p75_last52"] = hist.quantile(0.25), hist.quantile(0.75)
        out["retrain_date"] = r
        out["train_weeks"] = len(trw)
        preds.append(out)
    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(base / "volume_test_predictions.parquet", index=False)
    ev = P[P["week_status"] == "observable"]
    res = {}
    cols = ["pred"] + [f"glm_{fs}" for fs in vc["feature_sets"]] + vol.BASELINES
    for period, mask in (("overall", np.ones(len(ev), bool)), ("excluding_march_2026", ~ev["origin"].between(*STRESS).to_numpy()),
                         ("march_2026_stress", ev["origin"].between(*STRESS).to_numpy())):
        e = ev[mask]
        res[period] = {c: vol.errors(e["y"].to_numpy(), e[c].to_numpy()) for c in cols}
        res[period]["interval_coverage_selected"] = float(((e["y"] >= e["lo"]) & (e["y"] <= e["hi"])).mean())
    by_month = ev.assign(month=pd.to_datetime(ev["origin"]).dt.to_period("M").astype(str)).groupby("month").apply(
        lambda g: pd.Series({"actual": g["y"].sum(), "forecast": g["pred"].sum(), "baseline": g[selection["reference_baseline"]].sum(),
                             "mae": (g["pred"] - g["y"]).abs().mean(),
                             "mae_baseline": (g[selection["reference_baseline"]] - g["y"]).abs().mean()}), include_groups=False)
    by_month.round(1).reset_index().to_csv(rep / "volume_by_month.csv", index=False)
    # paired bootstrap of MAE difference (selected GLM - reference baseline) over test weeks
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    d_abs = (ev["pred"] - ev["y"]).abs().to_numpy() - (ev[selection["reference_baseline"]] - ev["y"]).abs().to_numpy()
    idx = rng.integers(0, len(d_abs), (cfg["bootstrap"]["resamples"], len(d_abs)))
    res["mae_diff_selected_minus_baseline_ci95"] = [float(np.quantile(d_abs[idx].mean(1), 0.025)), float(np.quantile(d_abs[idx].mean(1), 0.975))]
    res["capacity_guidance"] = capacity_guidance(paths, ev)
    res["selection"] = selection
    write_json(rep / "volume_forecast_results.json", res)
    return {"weeks": int(len(ev))}


def capacity_guidance(paths: Paths, ev: pd.DataFrame | None = None) -> dict:
    """Pre-specified volume-driven capacity rule vs fixed middle capacity, evaluated on the ADOPTED inspection list
    (Stage-7 fix: the original evaluation used the plain top-K list). Reads saved volume test predictions only."""
    cfg = cfg_load(paths)
    base, rep = dirs(paths)
    g = cfg["volume"]["capacity_guidance"]
    if ev is None:
        ev = pd.read_parquet(base / "volume_test_predictions.parquet")
        ev = ev[ev["week_status"] == "observable"]
    ev = ev.assign(scenario=np.where(ev["hi"] > ev["p75_last52"], g["high"], np.where(ev["pred"] < ev["p25_last52"], g["low"], g["middle"])))
    pol = cfg["primary_policy"]
    suffix = "__nonadjacent" if policy_rule(paths).get("non_adjacent") else ""
    out = {"rule": g["rule"], "scenario_weeks": {str(k): int(v) for k, v in ev["scenario"].value_counts().items()}}
    for tag, sfx in (("adopted_list", suffix), ("plain_topk_list", "")):
        W = {c: pd.read_parquet(base / "selections" / f"{pol}__{_cap_name(c)}{sfx}__weekly_outcomes.parquet").set_index("origin")
             for c in cfg["capacities"]}
        adaptive = pd.DataFrame([W[s].loc[o] for o, s in zip(ev["origin"], ev["scenario"])])
        fixed = W[g["middle"]].loc[ev["origin"]]
        out[tag] = {"list": f"{pol}{sfx}", "adaptive": insp.summarize(adaptive.reset_index(drop=True)),
                    "fixed_middle": insp.summarize(fixed.reset_index(drop=True))}
    write_json(rep / "capacity_guidance.json", out)
    return out


def step_capacity_guidance(paths: Paths) -> dict:
    out = capacity_guidance(paths)
    return {k: {"adaptive_per100": round(v["adaptive"]["reports_per_100_inspections"], 1),
                "fixed_per100": round(v["fixed_middle"]["reports_per_100_inspections"], 1),
                "adaptive_captured": v["adaptive"]["reports_captured"], "fixed_captured": v["fixed_middle"]["reports_captured"]}
            for k, v in out.items() if isinstance(v, dict) and "adaptive" in v}


ALTERNATIVE_LISTS = {  # (policy, non_adjacent, file suffix) - Stage-7 alternatives saved alongside the Stage-5 decision
    "strict_topk": ("primary__LGBM", False, "__topk"),
    "strict_nonadjacent": ("primary__LGBM", True, "__nonadjacent"),
}


def step_alternative_worklists(paths: Paths) -> dict:
    cfg = cfg_load(paths)
    base, _ = dirs(paths)
    done = {}
    for key, (pol, nonadj, sfx) in ALTERNATIVE_LISTS.items():
        for cap in cfg["capacities"]:
            wl, excl = build_worklists(paths, pol, cap, non_adjacent=nonadj)
            f = base / "worklists" / f"{pol}__{_cap_name(cap)}{sfx}.parquet"
            wl.to_parquet(f, index=False)
            done[f.name] = {"rows": len(wl), "sha256": sha256_file(f), "excluded": len(excl)}
    write_json(base / "worklists" / "alternatives_manifest.json", {"generated_utc": utc_iso(), "outputs": done})
    return done


def write_report_tables(paths: Paths) -> None:
    _, rep = dirs(paths)
    t = pd.read_csv(rep / "policy_comparison.csv")
    keep = ["policy", "target", "capacity", "k_per_week", "precision", "precision_ci95", "recall", "reports_captured",
            "reports_total", "report_capture_rate", "capture_ci95", "precision_lift_vs_B1", "capture_lift_vs_B1",
            "precision_diff_vs_B1_ci95", "capture_diff_vs_B1_ci95", "average_precision"]
    md = ["# Stage 5 policy comparison (test walk-forward 2025-07-07 -> 2026-10-12; real data)", ""]
    for period in ("overall", "excluding_march_2026", "march_2026_stress"):
        x = t[t["period"] == period][keep].copy()
        for c in ("precision", "recall", "report_capture_rate", "precision_lift_vs_B1", "capture_lift_vs_B1", "average_precision"):
            x[c] = x[c].astype(float).round(4)
        md += [f"## {period}", "", _md_table(x), ""]
    (rep / "policy_comparison.md").write_text("\n".join(md), encoding="utf-8")


CAPTURE_LISTS = [  # (label, saved selection stem) - lists scored under every capture definition (Stage-7 audit)
    ("Stage-5 decision: multi-block LightGBM + non-adjacent", "multi_block__LGBM__{cap}__nonadjacent"),
    ("Plain top-K multi-block LightGBM", "multi_block__LGBM__{cap}"),
    ("Strict single-block LightGBM (top-K)", "primary__LGBM__{cap}"),
    ("Strict single-block LightGBM + non-adjacent", "primary__LGBM__{cap}__nonadjacent"),
    ("Logistic regression (multi-block, top-K)", "multi_block__LR__{cap}"),
    ("Historical-rate baseline", "multi_block__B1_hist_rate__{cap}"),
    ("Recency baseline", "multi_block__B2_recency__{cap}"),
]


def step_capture_definitions(paths: Paths) -> dict:
    """Score every SAVED selection under four capture definitions (inspection.CAPTURE_DEFINITIONS) with week-bootstrap
    CIs and paired differences vs plain top-K. Evaluation only - no list is re-selected."""
    cfg = cfg_load(paths)
    base, rep = dirs(paths)
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    nb = cfg["bootstrap"]["resamples"]
    a = pd.read_parquet(paths.processed / "stage3" / "report_assignments.parquet",
                        columns=["origin", "report_id", "candidate_block_ids", "assigned_block_id", "match_status", "method",
                                 "included_in_labels"])
    a["origin"] = pd.to_datetime(a["origin"]).dt.date
    nodes = block_nodes(paths)
    rf = insp.report_frame(a[(a["origin"] >= TEST[0]) & (a["origin"] <= TEST[1])], nodes)
    composition = {"reports": len(rf), "unique": int((~rf["ambiguous"]).sum()),
                   "ambiguous_intersection_shared_node": int((rf["ambiguous"] & rf["shared_node"]).sum()),
                   "ambiguous_other": int((rf["ambiguous"] & ~rf["shared_node"]).sum())}
    rows, weekly = [], {}
    for cap in cfg["capacities"]:
        for label, stem in CAPTURE_LISTS:
            sel = pd.read_parquet(base / "selections" / f"{stem.format(cap=_cap_name(cap))}.parquet")
            sel["origin"] = pd.to_datetime(sel["origin"]).dt.date
            weekly[(label, cap, "test")] = insp.capture_by_definition(sel, rf)
    # validation weeks (policy-design period): the two LightGBM rankings, top-K and non-adjacent, re-selected
    # deterministically from the saved validation predictions (no retraining)
    from .stage4 import VAL
    rf_val = insp.report_frame(a[(a["origin"] >= VAL[0]) & (a["origin"] <= VAL[1])], nodes)
    vdir = base / "validation_predictions"
    val_lists = {"Stage-5 decision: multi-block LightGBM + non-adjacent": ("multi_block__LGBM", True),
                 "Plain top-K multi-block LightGBM": ("multi_block__LGBM", False),
                 "Strict single-block LightGBM (top-K)": ("primary__LGBM", False),
                 "Strict single-block LightGBM + non-adjacent": ("primary__LGBM", True)}
    if vdir.exists():
        for label, (pol, nonadj) in val_lists.items():
            p = pd.read_parquet(vdir / f"{pol}.parquet")
            p["origin"] = pd.to_datetime(p["origin"]).dt.date
            for cap in cfg["capacities"]:
                sel, _, _ = insp.select_all(p, cap, cfg["eligible_blocks"], nodes=nodes if nonadj else None)
                weekly[(label, cap, "validation")] = insp.capture_by_definition(sel, rf_val)
    for (label, cap, per), w in weekly.items():
        ref = weekly[("Plain top-K multi-block LightGBM", cap, per)]
        assert list(w["origin"]) == list(ref["origin"])
        periods = (("test", np.ones(len(w), bool)), ("march_2026", w["origin"].between(*STRESS).to_numpy())) \
            if per == "test" else (("validation", np.ones(len(w), bool)),)
        for period, mask in periods:
            ww, rr = w[mask], ref[mask]
            row = {"list": label, "capacity": cap, "period": period, "weeks": int(len(ww)), "reports": int(ww["reports_total"].sum())}
            for d in insp.CAPTURE_DEFINITIONS:
                row[d] = float(ww[d].sum() / ww["reports_total"].sum())
                row[f"{d}_ci95"] = insp.bootstrap_ratio(ww[d].to_numpy(), ww["reports_total"].to_numpy(float), rng, nb)
                if label != "Plain top-K multi-block LightGBM":
                    row[f"{d}_diff_vs_plain_topk_ci95"] = insp.bootstrap_ratio(
                        ww[d].to_numpy(), ww["reports_total"].to_numpy(float), rng, nb, rr[d].to_numpy(), rr["reports_total"].to_numpy(float))
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(rep / "capture_definitions.csv", index=False)
    write_json(rep / "capture_definitions.json", {"definitions": insp.CAPTURE_DEFINITIONS, "test_report_composition": composition,
                                                  "bootstrap": {"resamples": nb, "seed": cfg["bootstrap"]["seed"], "unit": "test week"},
                                                  "note": "Stage-7 audit addition; evaluation of saved lists only."})
    return {"rows": len(tab), **composition}


def step_all(paths: Paths) -> dict:
    out = {"compare": step_compare(paths), "geography": step_geography(paths)}
    if not (paths.processed / "stage5" / "validation_predictions").exists():
        out["validation_predictions"] = step_validation_predictions(paths)
    out["policy_design"] = step_policy_design(paths)
    out["policy_test"] = step_policy_test(paths)
    out["worklists"] = step_worklists(paths)
    out["alternative_worklists"] = step_alternative_worklists(paths)   # Stage 7: strict lists shown in the dashboard
    out["capture_definitions"] = step_capture_definitions(paths)       # Stage 7: four capture definitions
    out["volume"] = step_volume(paths)                                 # includes capacity guidance on the adopted list
    write_report_tables(paths)
    return out
