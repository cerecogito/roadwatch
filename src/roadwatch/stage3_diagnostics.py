"""Stage-3 matching and label diagnostics computed from the real Stage-3 outputs.

Writes outputs/stage3/matching_diagnostics.{md,json} and outputs/stage3/label_diagnostics.{md,json}.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Paths
from .storage import read_json, utc_iso, write_json

MATCHED = ("matched", "ambiguous")
VARIANT_NOTES = {
    "primary": "unique + ambiguous (hashed deterministic pick); primary + evidence-backed supplementary reports",
    "no_supplementary": "as primary, supplementary reports excluded",
    "alt_tiebreak": "ambiguous reports assigned with the alternative salt",
    "multi_block": "ambiguous reports make every candidate block positive",
    "confident_only": "only uniquely assigned reports; candidate blocks of ambiguous reports are unknown",
}


def pct(a, b) -> float:
    return round(100.0 * a / b, 2) if b else float("nan")


def annual_rate_drift(rates: pd.Series, tol_pp: float = 3.0) -> list[dict]:
    """Flag years whose match rate (percent) deviates from the median of the other years by more than tol_pp."""
    out = []
    for y, r in rates.items():
        ref = rates.drop(y).median()
        if abs(r - ref) > tol_pp:
            out.append({"year": int(y), "rate_pct": round(float(r), 2), "median_other_years_pct": round(float(ref), 2)})
    return out


def _q(s: pd.Series, qs=(0.5, 0.9, 0.95, 0.99, 1.0)) -> dict:
    s = pd.Series(s).dropna()
    return {f"p{int(q * 100)}": round(float(s.quantile(q)), 2) for q in qs} | {"n": int(len(s))}


def _md_table(df: pd.DataFrame) -> str:
    df = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def matching_diagnostics(paths: Paths) -> dict:
    d = paths.processed / "stage3"
    a = pd.read_parquet(d / "report_assignments.parquet")
    blocks = pd.read_parquet(d / "blocks.parquet", columns=["block_id", "length_m"])
    aliases = pd.read_parquet(d / "street_aliases.parquet")
    man = read_json(d / "stage3_manifest.json")
    a["year"] = pd.to_datetime(a["created_date_local"]).dt.year
    p = a[a["source"] == "primary"]
    s = a[a["source"] == "supplementary"]
    n = len(p)
    out: dict = {"generated_utc": utc_iso()}
    out["blocks"] = {"eligible_blocks": len(blocks), "total_km": round(float(blocks["length_m"].sum()) / 1000, 1),
                     "length_m": _q(blocks["length_m"], (0.05, 0.25, 0.5, 0.75, 0.95)), "checks": man["block_checks"]}
    out["reports"] = {"relevant_filter": man["relevant_filter"], "primary_reports": n,
                      "duplicate_primary_ids_in_source": man["duplicate_primary_ids"],
                      "supplementary_inspected": len(s),
                      "supplementary_id_overlap_with_primary": man["supplementary_id_overlap_with_primary"],
                      "supplementary_classes": s["supplementary_class"].value_counts().to_dict(),
                      "supplementary_recovered_unique": int(s["included_in_labels"].sum()),
                      "supplementary_recovered_by_year": s[s["included_in_labels"]].groupby("year").size().to_dict()}
    st = p["match_status"].value_counts()
    out["primary_status"] = {k: {"n": int(v), "pct": pct(v, n)} for k, v in st.items()}
    meth = p.groupby(["match_status", "method"], dropna=False).size().rename("n").reset_index()
    meth["pct"] = meth["n"].map(lambda v: pct(v, n))
    out["primary_by_method"] = meth.astype({"method": str}).to_dict("records")
    out["primary_by_confidence"] = p["confidence"].value_counts().to_dict()
    in_scope = p[p["match_status"] != "out_of_universe"]
    by = in_scope.groupby(["year", "address_type"]).agg(
        n=("report_id", "size"),
        matched_unique=("match_status", lambda x: int((x == "matched").sum())),
        ambiguous=("match_status", lambda x: int((x == "ambiguous").sum())),
        unmatched=("match_status", lambda x: int((x == "unmatched").sum()))).reset_index()
    by["assigned_pct"] = ((by["matched_unique"] + by["ambiguous"]) / by["n"] * 100).round(2)
    by["unique_pct"] = (by["matched_unique"] / by["n"] * 100).round(2)
    out["primary_by_year_address_type"] = by.to_dict("records")
    yr = in_scope.groupby("year")["match_status"].apply(lambda x: x.isin(MATCHED).mean() * 100)
    out["annual_assigned_rate_pct"] = {int(k): round(float(v), 2) for k, v in yr.items()}
    out["annual_rate_drift_flags_3pp"] = annual_rate_drift(yr)
    yr_u = in_scope.groupby("year")["match_status"].apply(lambda x: (x == "matched").mean() * 100)
    out["annual_unique_rate_pct"] = {int(k): round(float(v), 2) for k, v in yr_u.items()}
    out["annual_unique_rate_drift_flags_3pp"] = annual_rate_drift(yr_u)
    cn = p[p["method"].isin(["coord_name", "coord_name_cross"])]
    out["snap_distance_ft"] = {
        "address_coord_name": _q(cn.loc[cn["address_type"] == "ADDRESS", "assigned_distance_ft"]),
        "blockface_coord_name_2020_2021": _q(cn.loc[cn["address_type"] == "BLOCKFACE", "assigned_distance_ft"]),
        "intersection_point_to_picked_block": _q(p.loc[p["address_type"] == "INTERSECTION", "assigned_distance_ft"]),
        "nearest_eligible_block_any_report": _q(p["nearest_block_distance_ft"])}
    no_xy = p["x"].isna()
    out["assigned_without_coordinates"] = {"n": int((no_xy & p["match_status"].isin(MATCHED)).sum()),
                                           "of_reports_without_coordinates": int(no_xy.sum())}
    it = p[p["address_type"] == "INTERSECTION"]
    out["intersection"] = {
        "n": len(it), "assigned_pct": pct(it["match_status"].isin(MATCHED).sum(), len(it)),
        "unique_pct": pct((it["match_status"] == "matched").sum(), len(it)),
        "ambiguous_pct": pct((it["match_status"] == "ambiguous").sum(), len(it)),
        "coordinate_fallback_n": int((it["method"] == "intersection_coord").sum()),
        "candidate_count_distribution": it.loc[it["match_status"] == "ambiguous", "n_candidates"].value_counts().sort_index().to_dict()}
    amb = p[p["match_status"] == "ambiguous"]
    out["ambiguity"] = {"ambiguous_reports": len(amb), "pct": pct(len(amb), n),
                        "by_address_type": amb["address_type"].value_counts().to_dict(),
                        "primary_vs_alt_pick_differs": int((amb["assigned_block_id"] != amb["alt_block_id"]).sum()),
                        "candidate_count_quantiles": _q(amb["n_candidates"])}
    out["duplicates_and_conflicts"] = {
        "duplicate_report_ids_after_dedup": int(a["report_id"].duplicated().sum()),
        "reports_with_multiple_candidate_blocks": len(amb),
        "note": "each report is assigned by exactly one method; a report contributes at most once per block-week"}
    out["aliases"] = {"accepted": aliases[aliases["accepted"]].to_dict("records"),
                      "rejected_candidates": int((~aliases["accepted"]).sum()),
                      "reports_using_alias": int(p["alias_used"].sum())}
    out["unmatched_reasons"] = p.loc[p["match_status"] == "unmatched", "reason"].value_counts().to_dict()
    out["out_of_universe_reasons"] = p.loc[p["match_status"] == "out_of_universe", "reason"].value_counts().to_dict()
    mo = p.assign(m=pd.to_datetime(p["created_date_local"]).dt.to_period("M")).groupby("m").agg(
        n=("report_id", "size"), assigned=("match_status", lambda x: x.isin(MATCHED).mean() * 100),
        blockface_share=("address_type", lambda x: (x == "BLOCKFACE").mean() * 100))
    med = mo["assigned"].median()
    trailing = mo["n"].rolling(12, min_periods=6).median().shift(1)
    mo["volume_vs_trailing12_median"] = (mo["n"] / trailing).round(2)
    flags = mo[(mo["assigned"] - med).abs() > 5]
    vflags = mo[mo["volume_vs_trailing12_median"] > 2.5]
    out["monthly_anomalies"] = {
        "assigned_rate_more_than_5pp_from_median": {str(k): round(float(v), 2) for k, v in flags["assigned"].items()},
        "volume_above_2.5x_trailing_12m_median": {str(k): float(v) for k, v in vflags["volume_vs_trailing12_median"].items()},
        "monthly": {str(k): {"n": int(r.n), "assigned_pct": round(float(r.assigned), 2),
                             "blockface_pct": round(float(r.blockface_share), 1)} for k, r in mo.iterrows()}}
    return out


def label_diagnostics(paths: Paths) -> dict:
    d = paths.processed / "stage3"
    man = read_json(d / "stage3_manifest.json")
    weeks = pd.read_parquet(d / "label_weeks.parquet")
    out: dict = {"generated_utc": utc_iso()}
    out["calendar"] = {
        "origins": len(weeks), "first_origin": str(weeks["origin"].min()), "last_origin": str(weeks["origin"].max()),
        "data_end_exclusive": man["data_end_exclusive"], "embargo_days": man["embargo_days"],
        "observable_until": man["observable_until"],
        "last_complete_origin": str(weeks.loc[weeks["data_complete"], "origin"].max()),
        "utc_hours_distribution": weeks["utc_hours"].value_counts().to_dict(),
        "week_status": weeks["week_status"].astype(str).value_counts().to_dict(),
        "threshold": man["label_params"]["hidden_share_threshold"],
        "unknown_weeks_by_threshold": {c: int(weeks[c].sum()) for c in weeks.columns if c.startswith("unknown_at_share_")},
        "non_observable_weeks": weeks.loc[weeks["week_status"] != "observable",
                                          ["origin", "located_brooklyn_reports", "hidden_potential_brooklyn",
                                           "hidden_share_upper_bound", "week_status"]].astype(str).to_dict("records")}
    variants = {}
    prim_pos = None
    for v in man["label_summaries"]:
        panel = pd.read_parquet(d / f"labels_{v}.parquet", columns=["block_id", "origin", "label", "label_status",
                                                                     "unknown_reason", "n_reports"])
        lr = pd.read_parquet(d / f"label_reports_{v}.parquet")
        lab = panel[panel["label"].notna()]
        yrs = pd.to_datetime(lab["origin"]).dt.year
        info = {"note": VARIANT_NOTES.get(v, ""), "block_weeks": len(panel),
                "status": panel["label_status"].astype(str).value_counts().to_dict(),
                "unknown_reason": panel.loc[panel["label_status"] == "unknown", "unknown_reason"].astype(str).value_counts().to_dict(),
                "unique_reports_contributing": int(lr["report_id"].nunique()),
                "prevalence_pct_by_year": {int(k): round(float(x) * 100, 3) for k, x in lab.groupby(yrs)["label"].mean().items()},
                "blocks_with_positive_by_year": {int(k): int(x) for k, x in lab[lab["label"] == 1].groupby(
                    pd.to_datetime(lab.loc[lab["label"] == 1, "origin"]).dt.year)["block_id"].nunique().items()}}
        pos = panel[panel["label_status"] == "positive"]
        info["reports_per_positive_block_week"] = pos["n_reports"].value_counts().sort_index().head(10).to_dict()
        if v == "primary":
            mon = lab.groupby(pd.to_datetime(lab["origin"]).dt.to_period("M"))["label"].mean() * 100
            info["prevalence_pct_by_month"] = {str(k): round(float(x), 3) for k, x in mon.items()}
            per_block = lr.groupby("block_id")["report_id"].nunique()
            all_blocks = panel["block_id"].nunique()
            info["reports_per_block_full_period"] = {
                "blocks_with_any_report": int(len(per_block)), "blocks_total": int(all_blocks),
                "quantiles_among_reported_blocks": _q(per_block, (0.5, 0.75, 0.9, 0.99, 1.0)),
                "share_of_reports_on_top_1pct_blocks": round(float(per_block.sort_values(ascending=False).head(int(0.01 * all_blocks)).sum() / per_block.sum() * 100), 2)}
            wk = pos.groupby("origin").size()
            trailing = wk.rolling(12, min_periods=6).median().shift(1)
            ratio = (wk / trailing).dropna()
            info["weeks_positives_above_2.5x_trailing_12wk_median"] = {str(k): round(float(x), 2) for k, x in ratio[ratio > 2.5].items()}
            info["weekly_positive_block_counts"] = {str(k): int(x) for k, x in wk.items()}
            prim_pos = set(zip(pos["block_id"], pos["origin"]))
        else:
            cur = set(zip(pos["block_id"], pos["origin"]))
            inter = len(cur & prim_pos)
            info["vs_primary"] = {"positives_diff": len(cur) - len(prim_pos),
                                  "jaccard_positive_sets": round(inter / len(cur | prim_pos), 4),
                                  "primary_positives_not_in_variant": len(prim_pos - cur),
                                  "variant_positives_not_in_primary": len(cur - prim_pos)}
        variants[v] = info
        del panel
    out["variants"] = variants
    return out


def _render_matching(m: dict) -> str:
    L = ["# Stage 3 - matching diagnostics", "", f"Generated {m['generated_utc']} from data/processed/stage3 (real data).", "",
         f"- Eligible blocks: **{m['blocks']['eligible_blocks']:,}** ({m['blocks']['total_km']} km); length m {m['blocks']['length_m']}",
         f"- Relevant primary reports (Street Condition / Pothole, Brooklyn): **{m['reports']['primary_reports']:,}**; "
         f"duplicate ids {m['reports']['duplicate_primary_ids_in_source']}",
         f"- Supplementary inspected: {m['reports']['supplementary_inspected']:,}; id overlap with primary "
         f"{m['reports']['supplementary_id_overlap_with_primary']}; recovered (evidence-backed): "
         f"**{m['reports']['supplementary_recovered_unique']}** {m['reports']['supplementary_recovered_by_year']}",
         f"- Supplementary classes: {m['reports']['supplementary_classes']}", "", "## Primary status", "",
         _md_table(pd.DataFrame(m["primary_status"]).T), "", "## By method", "", _md_table(pd.DataFrame(m["primary_by_method"])),
         "", f"Confidence: {m['primary_by_confidence']}", "", "## By year and address type (out-of-universe excluded)", "",
         _md_table(pd.DataFrame(m["primary_by_year_address_type"])), "",
         f"Annual assigned rate %: {m['annual_assigned_rate_pct']} -> drift flags (>3 pp): {m['annual_rate_drift_flags_3pp']}",
         "", f"Annual unique rate %: {m['annual_unique_rate_pct']} -> drift flags (>3 pp): {m['annual_unique_rate_drift_flags_3pp']}",
         "", "## Distances (ft)", "", *[f"- {k}: {v}" for k, v in m["snap_distance_ft"].items()], "",
         f"Assigned without coordinates: {m['assigned_without_coordinates']}", "", f"Intersection: {m['intersection']}", "",
         f"Ambiguity: {m['ambiguity']}", "", f"Duplicates/conflicts: {m['duplicates_and_conflicts']}", "",
         "## Coordinate-validated aliases", "", _md_table(pd.DataFrame(m["aliases"]["accepted"])) if m["aliases"]["accepted"] else "none",
         "", f"Rejected alias candidates: {m['aliases']['rejected_candidates']}; reports using an alias: {m['aliases']['reports_using_alias']}",
         "", f"Unmatched reasons: {m['unmatched_reasons']}", "", f"Out-of-universe reasons: {m['out_of_universe_reasons']}", "",
         "## Monthly anomalies", "",
         f"- assigned rate > 5 pp from median: {m['monthly_anomalies']['assigned_rate_more_than_5pp_from_median']}",
         f"- volume > 2.5x trailing 12-month median: {m['monthly_anomalies']['volume_above_2.5x_trailing_12m_median']}"]
    return "\n".join(L) + "\n"


def _render_labels(lbl: dict) -> str:
    c = lbl["calendar"]
    L = ["# Stage 3 - label quality and observability", "", f"Generated {lbl['generated_utc']} (real data).", "",
         "Report counts, unique reports, positive block-weeks and physical road defects are different quantities: one "
         "defect can generate several reports, several reports can fall on one block-week (counted once), an ambiguous "
         "report can be attributed to one or several blocks depending on the variant, and unreported defects never appear.",
         "", f"- Origins: {c['origins']} ({c['first_origin']} -> {c['last_origin']}); last complete origin {c['last_complete_origin']} "
             f"(data end {c['data_end_exclusive']} exclusive, {c['embargo_days']}-day embargo -> observable until {c['observable_until']})",
         f"- Horizon length in UTC hours: {c['utc_hours_distribution']}", f"- Week status: {c['week_status']} (threshold {c['threshold']})",
         f"- Unknown weeks by threshold: {c['unknown_weeks_by_threshold']}", "", "## Non-observable weeks", "",
         _md_table(pd.DataFrame(c["non_observable_weeks"])), "", "## Variants", ""]
    rows = []
    for v, i in lbl["variants"].items():
        rows.append({"variant": v, "positive": i["status"].get("positive", 0), "negative": i["status"].get("negative", 0),
                     "unknown": i["status"].get("unknown", 0), "unique_reports": i["unique_reports_contributing"],
                     "jaccard_vs_primary": i.get("vs_primary", {}).get("jaccard_positive_sets", 1.0), "note": i["note"]})
    L += [_md_table(pd.DataFrame(rows)), ""]
    for v, i in lbl["variants"].items():
        L += [f"### {v}", "", f"- prevalence % by year (labelled block-weeks): {i['prevalence_pct_by_year']}",
              f"- blocks with >= 1 positive week by year: {i['blocks_with_positive_by_year']}",
              f"- unknown reasons: {i['unknown_reason']}", f"- reports per positive block-week: {i['reports_per_positive_block_week']}"]
        if "vs_primary" in i:
            L.append(f"- vs primary: {i['vs_primary']}")
        if v == "primary":
            L += [f"- reports per block (full period): {i['reports_per_block_full_period']}",
                  f"- weeks with positives > 2.5x trailing 12-week median: {i['weeks_positives_above_2.5x_trailing_12wk_median']}"]
        L.append("")
    return "\n".join(L) + "\n"


def write_all(paths: Paths) -> list:
    out_dir = paths.root / "outputs" / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    m = matching_diagnostics(paths)
    write_json(out_dir / "matching_diagnostics.json", m)
    (out_dir / "matching_diagnostics.md").write_text(_render_matching(m), encoding="utf-8")
    lbl = label_diagnostics(paths)
    write_json(out_dir / "label_diagnostics.json", lbl)
    (out_dir / "label_diagnostics.md").write_text(_render_labels(lbl), encoding="utf-8")
    return [out_dir / "matching_diagnostics.md", out_dir / "label_diagnostics.md"]
