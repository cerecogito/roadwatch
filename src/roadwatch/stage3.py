"""Stage 3 orchestration: block index -> report matching -> weekly labels -> diagnostics.

Reads only archived/processed Stage-2 outputs (offline, reproducible). Writes:
  data/processed/stage3/blocks.parquet                 Brooklyn block index (GeoParquet, EPSG:2263)
  data/processed/stage3/nodes.parquet                  centerline nodes with street keys
  data/processed/stage3/street_aliases.parquet         coordinate-validated alias candidates (accepted flag)
  data/processed/stage3/report_assignments.parquet     one row per relevant report (primary + supplementary)
  data/processed/stage3/assignment_candidates.parquet  report x candidate block
  data/processed/stage3/reports_for_inspection.parquet unmatched / ambiguous / out-of-universe / rejected supplementary
  data/processed/stage3/label_weeks.parquet            weekly calendar, observability and hidden-report bounds
  data/processed/stage3/labels_<variant>.parquet       block-week label panels (primary + sensitivity variants)
  data/processed/stage3/label_reports_<variant>.parquet  qualifying report ids per positive block-week
  data/processed/stage3/labels_primary_from_2021-05-03.parquet  sensitivity population
  data/processed/stage3/populations.json, stage3_manifest.json
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import asdict

import geopandas as gpd
import numpy as np
import pandas as pd

from .blocks import build_network, validate_blocks
from .config import AUDIT_REFERENCE, AVAILABILITY_LAGS, Paths
from .labels import LabelParams, assign_week_status, build_calendar, build_panel, week_origin
from .matching import Matcher, MatchParams, derive_aliases
from .names import street_key
from .storage import read_json, sha256_file, utc_iso, write_json

log = logging.getLogger(__name__)

RELEVANT_FILTER = {"complaint_type": "Street Condition", "descriptor": "Pothole"}   # audit §11 primary target
VARIANTS = ["primary", "no_supplementary", "alt_tiebreak", "multi_block", "confident_only"]
HIDDEN_CLASSES = {"unlocatable_no_information", "cross_only_brooklyn_compatible",
                  "matched_in_brooklyn_without_borough_evidence", "brooklyn_names_unmatched"}
SENSITIVITY_START = dt.date(2021, 5, 3)   # first Monday of May 2021 (audit §11a / Stage-3 brief)
SHARE_THRESHOLDS = (0.02, 0.05, 0.10)
REPORT_COLUMNS = ["unique_key", "created_date", "created_date_local", "address_type", "street_name", "incident_address",
                  "cross_street_1", "cross_street_2", "intersection_street_1", "intersection_street_2", "incident_zip",
                  "borough", "x_coordinate_state_plane_num", "y_coordinate_state_plane_num", "latitude_num",
                  "longitude_num", "status", "closed_date_local", "resolution_description"]


def _write(df, path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return sha256_file(path)


def relevant_reports(df: pd.DataFrame, source: str) -> pd.DataFrame:
    m = np.ones(len(df), bool)
    for c, v in RELEVANT_FILTER.items():
        m &= (df[c] == v).to_numpy()
    r = df.loc[m, REPORT_COLUMNS].copy()
    r.insert(0, "source", source)
    r = r.rename(columns={"unique_key": "report_id", "x_coordinate_state_plane_num": "x",
                          "y_coordinate_state_plane_num": "y", "status": "sr_status"})
    r["origin"] = week_origin(r["created_date_local"])
    return r.reset_index(drop=True)


def classify_supplementary(rep: pd.DataFrame, res: pd.DataFrame, matcher: Matcher, block_zips: dict[int, set]) -> pd.Series:
    out = []
    for r, m in zip(rep.to_dict("records"), res.to_dict("records")):
        s, i1, i2 = matcher.key(r["street_name"]), matcher.key(r["intersection_street_1"]), matcher.key(r["intersection_street_2"])
        a, b = matcher.key(r["cross_street_1"]), matcher.key(r["cross_street_2"])
        xy = pd.notna(r["x"]) and pd.notna(r["y"])
        if not (s or i1 or i2) and not xy:
            if a or b:
                out.append("cross_only_brooklyn_compatible" if matcher.pair_compatible(a, b) else "cross_only_brooklyn_incompatible")
            else:
                out.append("unlocatable_no_information")
            continue
        if m["status"] in ("matched", "ambiguous"):
            zips = set().union(*(block_zips.get(c, set()) for c in m["candidate_block_ids"]))
            if xy and m["name_corroborated"] == True:  # noqa: E712 - may be bool, numpy bool or None
                out.append("accepted_coord_and_name")
            elif isinstance(r["incident_zip"], str) and r["incident_zip"] in zips:
                out.append("accepted_names_and_zip")
            else:
                out.append("matched_in_brooklyn_without_borough_evidence")
            continue
        names = [n for n in (s, i1, i2, a, b) if n]
        out.append("brooklyn_names_unmatched" if any(n in matcher.net.all_names for n in names) else "not_brooklyn_names")
    return pd.Series(out, index=rep.index)


def _positives(asg: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    use = asg[asg["included_in_labels"]]
    if variant == "no_supplementary":
        use = use[use["source"] == "primary"]
    use = use[use["match_status"].isin(["matched", "ambiguous"])]
    amb = use["match_status"] == "ambiguous"
    cols = ["report_id", "origin", "source", "match_status", "method"]
    if variant in ("primary", "no_supplementary"):
        pos = use[cols].assign(block_id=use["assigned_block_id"])
        return pos, None
    if variant == "alt_tiebreak":
        return use[cols].assign(block_id=use["alt_block_id"]), None
    if variant == "multi_block":
        single = use.loc[~amb, cols].assign(block_id=use.loc[~amb, "assigned_block_id"])
        multi = use.loc[amb, cols + ["candidate_block_ids"]].explode("candidate_block_ids").rename(
            columns={"candidate_block_ids": "block_id"})
        return pd.concat([single, multi], ignore_index=True), None
    if variant == "confident_only":
        pos = use.loc[~amb, cols].assign(block_id=use.loc[~amb, "assigned_block_id"])
        cand = use.loc[amb, ["origin", "candidate_block_ids"]].explode("candidate_block_ids").rename(
            columns={"candidate_block_ids": "block_id"})
        return pos, cand
    raise ValueError(variant)


def run(paths: Paths, match_params: MatchParams = MatchParams(), label_params: LabelParams = LabelParams(),
        expected_blocks: int | None = AUDIT_REFERENCE["centerline_bk_eligible_blocks"]) -> dict:
    out_dir = paths.processed / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    inputs = {k: read_json(paths.processed / f"{k}.meta.json") for k in ("centerline", "sr311", "sr311_unassigned")}

    # 1. block index
    net = build_network(gpd.read_parquet(paths.processed / "centerline.parquet"))
    block_checks = validate_blocks(net.blocks, expected_blocks, tolerance=0)
    outputs["blocks.parquet"] = _write(net.blocks, out_dir / "blocks.parquet")
    nodes = pd.DataFrame({"node_id": np.arange(len(net.node_xy)), "x": net.node_xy[:, 0], "y": net.node_xy[:, 1],
                          "degree": net.node_degree,
                          "street_keys": [sorted(net.node_streets.get(i, ())) for i in range(len(net.node_xy))]})
    outputs["nodes.parquet"] = _write(nodes, out_dir / "nodes.parquet")
    log.info("block index: %d eligible blocks, %d nodes", len(net.blocks), len(nodes))

    # 2. report populations
    prim = relevant_reports(pd.read_parquet(paths.processed / "sr311.parquet"), "primary")
    supp_all = pd.read_parquet(paths.processed / "sr311_unassigned.parquet")
    supp = relevant_reports(supp_all, "supplementary")
    dup_primary = int(prim["report_id"].duplicated().sum())
    overlap = int(supp["report_id"].isin(set(prim["report_id"])).sum())
    prim = prim.drop_duplicates("report_id")
    supp = supp[~supp["report_id"].isin(set(prim["report_id"]))].drop_duplicates("report_id")

    # 3. coordinate-validated aliases (from primary ADDRESS reports only)
    aliases_tbl = derive_aliases(prim, net, match_params)
    outputs["street_aliases.parquet"] = _write(aliases_tbl, out_dir / "street_aliases.parquet")
    aliases = dict(zip(aliases_tbl.loc[aliases_tbl["accepted"], "report_key"], aliases_tbl.loc[aliases_tbl["accepted"], "cscl_key"]))
    matcher = Matcher(net, match_params, aliases)

    # 4. matching
    res_p = matcher.match(prim)
    res_s = matcher.match(supp)
    zips: dict[int, set] = {}
    for bid, lz, rz in zip(net.blocks["block_id"].astype("int64"), net.blocks["l_zip"], net.blocks["r_zip"]):
        zips[int(bid)] = {z for z in (lz, rz) if isinstance(z, str)}
    supp_class = classify_supplementary(supp, res_s, matcher, zips)
    alias_keys = set(aliases)

    def assemble(rep, res, cls):
        a = pd.concat([rep.reset_index(drop=True), res.reset_index(drop=True).rename(columns={"status": "match_status"})], axis=1)
        a["supplementary_class"] = cls.reset_index(drop=True) if cls is not None else None
        a["alias_used"] = [any(street_key(v) in alias_keys for v in vals if isinstance(v, str)) for vals in
                           a[["street_name", "cross_street_1", "cross_street_2", "intersection_street_1", "intersection_street_2"]].to_numpy()]
        return a

    asg = pd.concat([assemble(prim, res_p, None), assemble(supp, res_s, supp_class)], ignore_index=True)
    asg["brooklyn_evidence"] = np.where(asg["source"] == "primary", "primary_borough_field",
                                        np.where(asg["supplementary_class"].fillna("").str.startswith("accepted"),
                                                 asg["supplementary_class"], "none"))
    asg["included_in_labels"] = (asg["brooklyn_evidence"] != "none") & asg["match_status"].isin(["matched", "ambiguous"])
    asg["hidden_potential_brooklyn"] = asg["supplementary_class"].isin(HIDDEN_CLASSES)
    for c in ("assigned_block_id", "alt_block_id"):
        asg[c] = asg[c].astype("Int64")
    outputs["report_assignments.parquet"] = _write(asg, out_dir / "report_assignments.parquet")
    cand = asg[["report_id", "source", "match_status", "assigned_block_id", "alt_block_id", "candidate_block_ids"]].explode(
        "candidate_block_ids").dropna(subset=["candidate_block_ids"]).rename(columns={"candidate_block_ids": "block_id"})
    cand["block_id"] = cand["block_id"].astype("int64")
    cand["is_primary_pick"] = cand["block_id"] == cand["assigned_block_id"]
    cand["is_alt_pick"] = cand["block_id"] == cand["alt_block_id"]
    outputs["assignment_candidates.parquet"] = _write(cand.drop(columns=["assigned_block_id", "alt_block_id"]),
                                                      out_dir / "assignment_candidates.parquet")
    insp = asg[(asg["match_status"] != "matched") | ((asg["source"] == "supplementary") & ~asg["included_in_labels"])]
    outputs["reports_for_inspection.parquet"] = _write(insp, out_dir / "reports_for_inspection.parquet")

    # 5. weekly calendar and observability
    sr_state = read_json(paths.root / inputs["sr311"]["raw_snapshot"] / "_state.json")
    data_start = dt.date.fromisoformat(sr_state["requested_start"])
    data_end = dt.date.fromisoformat(sr_state["requested_end_exclusive"])
    embargo = AVAILABILITY_LAGS["sr311"]["days"]
    weeks = build_calendar(data_start, data_end, embargo)
    bk = asg[asg["brooklyn_evidence"] != "none"]
    located = bk.groupby("origin").size()
    hidden = asg[asg["hidden_potential_brooklyn"]].groupby("origin").size()
    weeks = assign_week_status(weeks, located, hidden, label_params.hidden_share_threshold)
    for t in SHARE_THRESHOLDS:
        weeks[f"unknown_at_share_{int(t * 100)}pct"] = weeks["data_complete"] & (weeks["hidden_share_upper_bound"] > t)
    weeks["unmatched_brooklyn_reports"] = weeks["origin"].map(
        bk[~bk["match_status"].isin(["matched", "ambiguous"])].groupby("origin").size()).fillna(0).astype(int)
    outputs["label_weeks.parquet"] = _write(weeks, out_dir / "label_weeks.parquet")

    # 6. label panels
    block_ids = net.blocks["block_id"].astype("int64").to_numpy()
    summaries = {}
    for v in VARIANTS:
        pos, amb_cand = _positives(asg, v)
        pos = pos.assign(block_id=pos["block_id"].astype("int64"))
        if amb_cand is not None:
            amb_cand = amb_cand.assign(block_id=amb_cand["block_id"].astype("int64"))
        panel = build_panel(block_ids, weeks, pos, amb_cand)
        outputs[f"labels_{v}.parquet"] = _write(panel, out_dir / f"labels_{v}.parquet")
        lr = pos.drop_duplicates(["block_id", "origin", "report_id"]).sort_values(["origin", "block_id", "report_id"])
        outputs[f"label_reports_{v}.parquet"] = _write(lr, out_dir / f"label_reports_{v}.parquet")
        summaries[v] = {"positive": int((panel["label_status"] == "positive").sum()),
                        "negative": int((panel["label_status"] == "negative").sum()),
                        "unknown": int((panel["label_status"] == "unknown").sum())}
        if v == "primary":
            sens = panel[panel["origin"] >= SENSITIVITY_START]
            outputs["labels_primary_from_2021-05-03.parquet"] = _write(sens, out_dir / "labels_primary_from_2021-05-03.parquet")
        del panel

    last_complete = weeks.loc[weeks["data_complete"], "origin"].max()
    populations = {
        "generated_utc": utc_iso(),
        "target": "at least one relevant 311 Street Condition/Pothole REPORT assigned to the block in [t, t+7d) local",
        "label_file": "labels_primary.parquet",
        "row_filter": "label is not null (label_status != 'unknown')",
        "recommended_primary": {"origin_from": "2021-01-04", "origin_to": str(last_complete),
                                "note": "2020 origins kept as 365-day look-back history only; unknown weeks excluded",
                                "splits": {"train": ["2021-01-04", "2024-12-30"], "validation": ["2025-01-06", "2025-06-30"],
                                           "test": ["2025-07-07", str(last_complete)]}},
        "sensitivity_from_2021_05": {"origin_from": str(SENSITIVITY_START), "origin_to": str(last_complete),
                                     "file": "labels_primary_from_2021-05-03.parquet"},
        "look_back_contamination_window": {"origin_from": None, "origin_to": None,
                                           "note": "filled below from affected weeks"},
    }
    affected = weeks.loc[weeks["week_status"] == "unlocated_reports", "origin"]
    if len(affected):
        populations["look_back_contamination_window"] = {
            "origin_from": str(affected.min()), "origin_to": str(affected.max() + dt.timedelta(days=365 + 7)),
            "note": "origins whose 365-day report-history window overlaps unlocated-report weeks; history features may "
                    "be deflated (hidden reports cannot be attributed to blocks)"}
    write_json(out_dir / "populations.json", populations)

    manifest = {"stage": 3, "generated_utc": utc_iso(), "inputs": inputs, "relevant_filter": RELEVANT_FILTER,
                "match_params": asdict(match_params), "label_params": asdict(label_params),
                "embargo_days": embargo, "data_start": str(data_start), "data_end_exclusive": str(data_end),
                "observable_until": weeks.attrs.get("observable_until"), "accepted_aliases": aliases,
                "block_checks": block_checks, "duplicate_primary_ids": dup_primary,
                "supplementary_id_overlap_with_primary": overlap, "label_summaries": summaries,
                "outputs_sha256": outputs}
    write_json(out_dir / "stage3_manifest.json", manifest)
    return manifest
