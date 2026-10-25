"""Assemble outputs/data_quality/data_quality_report.{md,json} from the latest per-source validation reports."""
from __future__ import annotations

import json

from .config import Paths
from .sources import SOURCES
from .storage import read_json, utc_iso, write_json

KEY_INFO = {"observed_created_range", "eligible_block_length_m", "reporting_lag_days", "recent_daily_counts",
            "inspection_years", "rptdate_years", "paving_end_years", "future_dated_rows", "latest_dates",
            "source_flags_by_year", "units", "publication_date_unverified", "descriptor_counts",
            "pothole_missing_coords_by_year_address_type", "unassigned_borough_values", "created_hour_of_day_profile"}


def latest_reports(paths: Paths) -> dict[str, dict]:
    out = {}
    for key in SOURCES:
        files = sorted(paths.quality.glob(f"{key}__*.json"))
        if files:
            out[key] = read_json(files[-1])
    return out


def _fmt(v, limit: int = 600) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= limit else s[:limit] + " ..."


def build_report(paths: Paths) -> tuple:
    reports = latest_reports(paths)
    registry = read_json(paths.local_registry) if paths.local_registry.exists() else None
    lines = [f"# RoadWatch data-quality report", "", f"Generated {utc_iso()} from the latest validation report per source "
             "(`outputs/data_quality/<source>__<snapshot>.json`). All numbers come from executed pipeline runs.", "",
             "| Source | Role | Snapshot (UTC) | Extract | Validation | Rows (raw -> dedup) | Observed date range | Processed |",
             "|---|---|---|---|---|---|---|---|"]
    for key, spec in SOURCES.items():
        r = reports.get(key)
        if not r:
            lines.append(f"| `{key}` | {spec.role} | - | NOT RETRIEVED | - | - | - | - |")
            continue
        v = r["validation"]
        dr = r.get("observed_date_range") or {}
        rng = f"{(dr.get('min') or '')[:10]} -> {(dr.get('max') or '')[:10]}" if dr else "n/a (no date field)"
        t = r["totals"]
        lines.append(f"| `{key}` | {r['role']} | {r['snapshot_id']} | {r['extract_status']} | **{v['status'].upper()}** "
                     f"(P{v['counts']['PASS']}/W{v['counts']['WARN']}/F{v['counts']['FAIL']}) | "
                     f"{t['raw_rows']:,} -> {t['rows_after_dedup']:,} | {rng} | {'yes' if r.get('processed') else 'NO'} |")
    for key, r in reports.items():
        lines += ["", f"## `{key}` - {r['title']}", "",
                  f"- Endpoint: `{r['endpoint']}` (id `{r['identifier']}`); app token used: {r['app_token_used']}",
                  f"- Query: `{r['query']['base_where']}`; requested {r['query']['requested_start']} -> "
                  f"{r['query']['requested_end_exclusive']} (exclusive); {r['query']['partitioning']}; page size {r['query']['page_size']}",
                  f"- Retrieved {r['retrieval_started_utc']} -> {r['retrieval_completed_utc']}",
                  f"- Fitness: {r['fitness']}",
                  f"- Availability lag: {r.get('availability_lag')}",
                  "- Caveats: " + " | ".join(r["caveats"]), "", "| Check | Status | Detail |", "|---|---|---|"]
        for c in r["validation"]["checks"]:
            if c["status"] != "INFO" or c["check"] in KEY_INFO:
                detail = c["detail"] + (f" -> `{_fmt(c['value'])}`" if c["status"] == "INFO" and c["value"] is not None else "")
                lines.append(f"| {c['check']} | {c['status']} | {detail.replace('|', '/')} |")
        miss = next((c["value"] for c in r["validation"]["checks"] if c["check"] == "missingness"), {})
        spec = SOURCES[key]
        req_miss = {f: miss.get(f) for f in spec.required_fields if f in miss}
        lines.append(f"\nMissingness of required fields (share null): `{_fmt(req_miss, 2000)}`")
    if registry:
        lines += ["", "## Original local files (preserved in place)", "", "| File | Status | Pipeline use | Reason | sha256 (first 16) |",
                  "|---|---|---|---|---|"]
        for e in registry["files"]:
            lines.append(f"| `{e['file']}` | {e['extract_status']} | {e['pipeline_use']} | {e['reason']} | "
                         f"{(e.get('sha256') or 'missing')[:16]} |")
        for e in registry["excluded_sources_without_file"]:
            lines.append(f"| {e['source']} | - | {e['pipeline_use']} | {e['reason']} | - |")
    md = "\n".join(lines) + "\n"
    paths.quality.mkdir(parents=True, exist_ok=True)
    (paths.quality / "data_quality_report.md").write_text(md, encoding="utf-8")
    write_json(paths.quality / "data_quality_report.json",
               {"generated_utc": utc_iso(), "sources": {k: {kk: r[kk] for kk in ("snapshot_id", "extract_status", "totals",
                                                                                 "observed_date_range", "processed")}
                                                        | {"validation": {"status": r["validation"]["status"], "counts": r["validation"]["counts"]}}
                                                        for k, r in reports.items()}})
    return paths.quality / "data_quality_report.md", reports
