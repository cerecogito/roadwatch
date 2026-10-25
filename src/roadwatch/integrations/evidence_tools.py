"""Read-only evidence tools: the ONLY way the assistants obtain facts.

Each tool validates its arguments against a JSON schema, reads saved RoadWatch outputs through dashboard_data (never
models, never SQL, never a shell), and returns {"data": ..., "sources": [...]} or {"error": ...}. Nothing here
computes a new score or changes a ranking.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .. import dashboard_data as dd
from ..config import Paths

LISTS = list(dd.LISTS)
CAPS = {"0.5%": 0.005, "1%": 0.01, "2%": 0.02}
PERIODS = ["test", "validation", "march_2026"]
CAPTURE_DEFS = ["any_candidate", "assigned_block", "fractional", "intersection_node"]

METRICS = {
    "precision": "Share of the listed blocks that received at least one report in the forecast week (against the list's own target labels).",
    "recall": "Share of all block-weeks with a report that were on the list. RoadWatch reports capture instead, because recall is per block-week and many reports are ambiguous.",
    "report_capture": "Share of next-week unique reports that fell on listed blocks. It has four variants (any_candidate, assigned_block, fractional, intersection_node); always state which.",
    "any_candidate": "A report counts as captured if ANY of its candidate blocks is listed. Optimistic for ambiguous reports.",
    "assigned_block": "A report counts only if its single assigned block (a deterministic hash pick for ambiguous reports) is listed. Unbiased estimate of fractional capture.",
    "fractional": "Credit equals the share of the report's candidate blocks that are listed (expected capture if the true location is uniform over candidates).",
    "intersection_node": "Full credit for an intersection report when any adjoining block is listed; valid only if inspecting a block also covers its end intersections. Other ambiguous reports get fractional credit.",
    "relative_score": "The model's raw ranking score. Comparable only within the same week and list. Not a probability.",
    "strict_target": "Each report counts on exactly one block (its unique match, or a deterministic hash pick among candidates).",
    "multi_block_target": "Every candidate block of a report counts as positive. Easier than the strict target, so its precision is not comparable.",
    "prevalence": "Share of block-weeks that are positive.",
    "calibration_in_the_large": "Predicted divided by observed number of positives; 1 means the right total volume.",
}

SCHEMAS = {
    "get_block_evidence": {
        "description": "Verified report history, road attributes, model drivers and rank for ONE block in a saved forecast week.",
        "parameters": {"type": "object", "properties": {
            "block_id": {"type": "integer", "description": "CSCL block id"},
            "week": {"type": "string", "description": "Forecast Monday, YYYY-MM-DD"},
            "capacity": {"type": "string", "enum": list(CAPS)}, "list_key": {"type": "string", "enum": LISTS}},
            "required": ["block_id", "week"]}},
    "get_worklist_position": {
        "description": "Where a block, or the top N blocks, sit in a week's inspection worklist (rank, priority, score).",
        "parameters": {"type": "object", "properties": {
            "week": {"type": "string"}, "block_id": {"type": "integer"},
            "top_n": {"type": "integer", "description": "Return the top N rows instead (max 10)"},
            "capacity": {"type": "string", "enum": list(CAPS)}, "list_key": {"type": "string", "enum": LISTS}},
            "required": ["week"]}},
    "compare_inspection_policies": {
        "description": "Unique-report capture of each inspection list under the four capture definitions, with 95% CIs.",
        "parameters": {"type": "object", "properties": {
            "period": {"type": "string", "enum": PERIODS}, "capacity": {"type": "string", "enum": list(CAPS)}}, "required": ["period"]}},
    "get_metric_definition": {
        "description": "Plain-language definition of a RoadWatch metric or term.",
        "parameters": {"type": "object", "properties": {"metric": {"type": "string", "enum": list(METRICS)}}, "required": ["metric"]}},
    "get_historical_period_results": {
        "description": "Precision, capture and counts for every list over the whole test period or the March 2026 surge.",
        "parameters": {"type": "object", "properties": {
            "period": {"type": "string", "enum": ["test", "march_2026"]}, "capacity": {"type": "string", "enum": list(CAPS)}}, "required": ["period"]}},
}


class ToolError(ValueError):
    pass


def _validate(name: str, args: dict) -> dict:
    """Minimal JSON-schema validation (type, enum, required, unknown keys rejected)."""
    sch = SCHEMAS[name]["parameters"]
    if not isinstance(args, dict):
        raise ToolError("arguments must be an object")
    extra = set(args) - set(sch["properties"])
    if extra:
        raise ToolError(f"unknown arguments: {sorted(extra)}")
    for k in sch.get("required", []):
        if k not in args:
            raise ToolError(f"missing required argument: {k}")
    for k, v in args.items():
        p = sch["properties"][k]
        if p["type"] == "integer" and (isinstance(v, bool) or not isinstance(v, int)):
            raise ToolError(f"{k} must be an integer")
        if p["type"] == "string" and not isinstance(v, str):
            raise ToolError(f"{k} must be a string")
        if "enum" in p and v not in p["enum"]:
            raise ToolError(f"{k} must be one of {p['enum']}")
    return args


class EvidenceTools:
    def __init__(self, paths: Paths | None = None, default_list: str | None = None):
        self.paths = paths or Paths()
        self.default_list = default_list or dd.default_list(self.paths)

    # ---- helpers
    def _week(self, s: str) -> dt.date:
        try:
            d = dt.date.fromisoformat(s)
        except ValueError:
            raise ToolError(f"'{s}' is not a valid date (YYYY-MM-DD)")
        weeks = dd.available_weeks(self.paths)
        if d not in weeks:
            raise ToolError(f"no saved forecast for {d}; available Mondays run {weeks[0]} to {weeks[-1]}")
        return d

    def _wl(self, week, capacity, list_key):
        return dd.worklist(self.paths, week, CAPS[capacity or "1%"], list_key or self.default_list)

    @staticmethod
    def _row(r) -> dict:
        return {"rank": int(r["rank"]), "priority": r["priority"], "block_id": int(r["block_id"]),
                "street": " ".join(str(r["full_street_name"]).split()), "zip": str(r["zip"]),
                "relative_score": round(float(r["model_score_relative"]), 3)}

    # ---- tools
    def get_block_evidence(self, block_id, week, capacity="1%", list_key=None) -> dict:
        d = self._week(week)
        lk = list_key or self.default_list
        wl = self._wl(d, capacity, lk)
        row = wl[wl["block_id"] == block_id]
        sources = [f"data/processed/stage5/worklists ({dd.LISTS[lk][1].format(cap=dd.CAP_NAME[CAPS[capacity]])})", "data/processed/stage4/predictions"]
        if row.empty:
            sc = dd.week_scores(self.paths, d, dd.LISTS[lk][2])
            r = sc[sc["block_id"] == block_id]
            if r.empty:
                return {"error": f"block {block_id} has no saved score for {d}"}
            return {"data": {"block_id": block_id, "week": str(d), "on_worklist": False, "list": dd.LISTS[lk][0],
                             "capacity": capacity, "within_week_rank_of_all_blocks": int(r["rank"].iloc[0]),
                             "blocks_ranked": int(len(sc)), "relative_score": round(float(r["score"].iloc[0]), 3),
                             "note": "Not selected at this capacity."}, "sources": sources}
        r = row.iloc[0]
        return {"data": {**self._row(r), "week": str(d), "on_worklist": True, "list": dd.LISTS[lk][0], "capacity": capacity,
                         "selection_rule": r["selection_rule"], "reports_365d": int(r["reports_365d"]), "reports_30d": int(r["reports_30d"]),
                         "days_since_last_report": float(r["days_since_last_report"]),
                         "adjacent_block_reports_365d": float(r["adjacent_block_reports_365d"]), "length_m": round(float(r["length_m"])),
                         "travel_lanes": None if r["travel_lanes"] != r["travel_lanes"] else float(r["travel_lanes"]),
                         "street_width_ft": None if r["street_width_ft"] != r["street_width_ft"] else float(r["street_width_ft"]),
                         "rationale": r["rationale"], "model_drivers": r["model_drivers"], "model_version": r["model_version"],
                         "recommended_action": r["recommended_action"]}, "sources": sources}

    def get_worklist_position(self, week, block_id=None, top_n=None, capacity="1%", list_key=None) -> dict:
        d = self._week(week)
        lk = list_key or self.default_list
        wl = self._wl(d, capacity, lk)
        src = ["data/processed/stage5/worklists"]
        if block_id is not None:
            r = wl[wl["block_id"] == block_id]
            return ({"data": {**self._row(r.iloc[0]), "of": int(len(wl)), "list": dd.LISTS[lk][0], "capacity": capacity}, "sources": src}
                    if len(r) else {"error": f"block {block_id} is not on the {capacity} {dd.LISTS[lk][0]} list for {d}"})
        n = min(int(top_n or 5), 10)
        return {"data": {"week": str(d), "list": dd.LISTS[lk][0], "capacity": capacity, "of": int(len(wl)),
                         "top": [self._row(r) for _, r in wl.head(n).iterrows()]}, "sources": src}

    def compare_inspection_policies(self, period, capacity="1%") -> dict:
        t, meta = dd.capture_definitions(self.paths)
        s = t[(t["capacity"] == CAPS[capacity]) & (t["period"] == period)]
        if s.empty:
            return {"error": f"no capture evidence for period={period} capacity={capacity}"}
        rows = []
        for _, r in s.iterrows():
            ent = {"list": r["list"]}
            for k in CAPTURE_DEFS:
                ci = json.loads(r[f"{k}_ci95"]) if isinstance(r[f"{k}_ci95"], str) else r[f"{k}_ci95"]
                ent[k] = {"value_pct": round(100 * float(r[k]), 1), "ci95_pct": [round(100 * ci[0], 1), round(100 * ci[1], 1)]}
            rows.append(ent)
        return {"data": {"period": period, "capacity": capacity, "weeks": int(s["weeks"].iloc[0]), "reports": int(s["reports"].iloc[0]),
                         "lists": rows, "definitions": meta["definitions"],
                         "caveat": "Non-adjacent lists win only under intersection_node/any_candidate; under fractional/assigned_block the strict top-K list is best."},
                "sources": ["outputs/stage5/capture_definitions.csv", "outputs/stage5/capture_definitions.json"]}

    def get_metric_definition(self, metric) -> dict:
        return {"data": {"metric": metric, "definition": METRICS[metric]}, "sources": ["docs/final/technical_manual.md (glossary)"]}

    def get_historical_period_results(self, period, capacity="1%") -> dict:
        s = dd.policy_summary(self.paths)
        s = s[(s["capacity"] == CAPS[capacity]) & (s["period"] == period)]
        if s.empty:
            return {"error": f"no summary for period={period} capacity={capacity}"}
        rows = [{"policy": r["policy"], "slots_per_week": int(r["k_per_week"]), "weeks": int(r["weeks"]),
                 "reports_captured_any_candidate": int(r["captured_any_candidate"]), "reports_total": int(r["reports_total"]),
                 "capture_any_pct": round(100 * r["capture_any_candidate"], 1), "capture_assigned_pct": round(100 * r["capture_assigned_block"], 1),
                 "precision_own_target": round(float(r["precision_own_target"]), 3), "precision_target": r["precision_target"]}
                for _, r in s.iterrows()]
        return {"data": {"period": period, "capacity": capacity, "policies": rows,
                         "caveat": "Precision is measured against each list's own target and is not comparable across targets."},
                "sources": ["outputs/stage6/policy_summary.csv"]}

    # ---- single entry point used by every caller (validates, never raises)
    def call(self, name: str, args: dict | None) -> dict:
        if name not in SCHEMAS:
            return {"error": f"unknown tool '{name}'"}
        try:
            return getattr(self, name)(**_validate(name, dict(args or {})))
        except ToolError as e:
            return {"error": str(e)}
        except dd.MissingOutput as e:
            return {"error": f"a required saved output is missing ({e.path}); run: {e.hint}"}
