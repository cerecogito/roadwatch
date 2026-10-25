"""Reproducible evidence briefs from a curated, versioned table of VALIDATED evaluation results, via Snowflake.

Flow: curated table (built from saved outputs only; aggregates, no complaint data) -> approved parameterised SQL template
(Snowflake SQL API, read-only role) -> rows + metadata -> Cortex chat completions (draft prose) -> every number checked
against the rows -> brief with model identity, template, parameters, evidence version and query hash.
The language model never writes SQL: only the templates below can run. With Snowflake off, `local_brief` produces a
deterministic brief from the same curated table, so the capability is reproducible without any account.
Endpoints (official docs): POST https://<account>.snowflakecomputing.com/api/v2/statements (SQL API) and
POST https://<account>.snowflakecomputing.com/api/v2/cortex/v1/chat/completions (Cortex REST API, OpenAI-style body); auth
`Authorization: Bearer <PAT>` + `X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests

from ..config import Paths
from . import config
from .gemini_assistant import NUM

DEFAULT_MODEL = "claude-sonnet-4-5"           # model name used in Snowflake's Cortex REST examples; override with SNOWFLAKE_CORTEX_MODEL
TABLE = "ROADWATCH_EVIDENCE"
CACHE_DIR = config.ROOT / "outputs" / "integrations" / "snowflake_cache"
TIMEOUT_S = 60
MAX_ROWS = 200

# Approved templates: id -> (SQL with ? placeholders in order, parameter names, description). No other SQL is ever executed.
TEMPLATES = {
    "capture_by_period": (
        f"SELECT list_name, metric, value_pct, ci_lo_pct, ci_hi_pct, weeks, reports FROM {TABLE} "
        f"WHERE family = 'capture' AND period = ? AND capacity = ? ORDER BY list_name, metric LIMIT {MAX_ROWS}",
        ("period", "capacity"), "Unique-report capture of every list under the four capture definitions"),
    "model_vs_baselines": (
        f"SELECT target, model, metric, value_pct, weeks FROM {TABLE} WHERE family = 'model' AND target = ? AND period = ? "
        f"ORDER BY model, metric LIMIT {MAX_ROWS}",
        ("target", "period"), "Strict or multi-block model metrics against baselines"),
}
PARAM_ENUMS = {"period": {"test", "validation", "march_2026", "overall", "march_2026_stress"}, "capacity": {"0.5pct", "1pct", "2pct"},
               "target": {"primary", "multi_block"}}
SYSTEM = ("You write a concise evidence brief (max 150 words) for road-inspection analysts from the JSON rows provided. Use ONLY "
          "numbers present in the rows, quoted exactly. Always state the target (strict single-block vs multi-block) and the capture "
          "definition for every figure. Never claim confirmed defects, prevented failures or savings. State plainly that figures "
          "from different targets or capture definitions are not comparable.")


# ------------------------------------------------------------------ curated, versioned evidence
def build_curated(paths: Paths | None = None) -> pd.DataFrame:
    root = (paths or Paths()).root
    cd = pd.read_csv(root / "outputs/stage5/capture_definitions.csv")
    rows = []
    for _, r in cd.iterrows():
        for k in ("any_candidate", "assigned_block", "fractional", "intersection_node"):
            ci = json.loads(r[f"{k}_ci95"])
            rows.append({"family": "capture", "period": r["period"], "capacity": {0.005: "0.5pct", 0.01: "1pct", 0.02: "2pct"}[r["capacity"]],
                         "list_name": r["list"], "target": "", "model": "", "metric": k, "value_pct": round(100 * r[k], 1),
                         "ci_lo_pct": round(100 * ci[0], 1), "ci_hi_pct": round(100 * ci[1], 1), "weeks": int(r["weeks"]), "reports": int(r["reports"])})
    t4 = pd.read_csv(root / "outputs/stage4/test_results.csv")
    t4 = t4[(t4["group"] == "main") & t4["variant"].isin(["primary", "multi_block"]) & t4["period"].isin(["overall", "march_2026_stress"])]
    for _, r in t4.iterrows():
        for m, col in (("precision_at_1pct", "top_1pct_precision"), ("report_capture_at_1pct", "top_1pct_report_capture"), ("average_precision", "average_precision")):
            rows.append({"family": "model", "period": r["period"], "capacity": "1pct", "list_name": "", "target": r["variant"], "model": r["model"],
                         "metric": m, "value_pct": round(100 * r[col], 1), "ci_lo_pct": None, "ci_hi_pct": None, "weeks": int(r["weeks"]), "reports": None})
    df = pd.DataFrame(rows)
    df.insert(0, "evidence_id", [hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()[:16] for r in df.to_dict("records")])
    return df


def evidence_version(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]


def rows_for(df: pd.DataFrame, template: str, params: dict) -> pd.DataFrame:
    """Local execution of the SAME templates over the curated table (used offline and to cross-check Snowflake rows)."""
    if template == "capture_by_period":
        s = df[(df["family"] == "capture") & (df["period"] == params["period"]) & (df["capacity"] == params["capacity"])]
        return s.sort_values(["list_name", "metric"])[["list_name", "metric", "value_pct", "ci_lo_pct", "ci_hi_pct", "weeks", "reports"]].head(MAX_ROWS)
    s = df[(df["family"] == "model") & (df["target"] == params["target"]) & (df["period"] == params["period"])]
    return s.sort_values(["model", "metric"])[["target", "model", "metric", "value_pct", "weeks"]].head(MAX_ROWS)


def check_request(template: str, params: dict) -> dict:
    if template not in TEMPLATES:
        raise ValueError(f"unknown template '{template}'")
    need = TEMPLATES[template][1]
    if set(params) != set(need):
        raise ValueError(f"template {template} needs exactly {need}")
    for k, v in params.items():
        if v not in PARAM_ENUMS[k]:
            raise ValueError(f"{k} must be one of {sorted(PARAM_ENUMS[k])}")
    return params


# ------------------------------------------------------------------ number verification and provenance
def verify_numbers(text: str, rows: pd.DataFrame) -> list[str]:
    """Numbers in the brief that are not present in the query rows (empty list = every number traces to the evidence)."""
    pool = {f"{v:.1f}" for c in ("value_pct", "ci_lo_pct", "ci_hi_pct") if c in rows for v in rows[c].dropna()}
    pool |= {str(int(v)) for c in ("weeks", "reports") if c in rows for v in rows[c].dropna()} | {f"{int(v):,}" for c in ("reports",) if c in rows for v in rows[c].dropna()}
    bad = []
    for n in NUM.findall(text):
        c = n.rstrip(".,")
        if c in pool or c.replace(",", "") in pool or len(c) <= 1 or c in {"95", "100", "150"} or re.fullmatch(r"20\d\d", c):
            continue
        bad.append(n)
    return bad


def local_brief(df: pd.DataFrame, template: str, params: dict) -> dict:
    """Deterministic brief (no language model): formats the query rows with their definitions and caveats."""
    check_request(template, params)
    rows = rows_for(df, template, params)
    if template == "capture_by_period":
        lists = []
        for name, g in rows.groupby("list_name"):
            d = g.set_index("metric")
            lists.append(f"{name}: fractional {d.loc['fractional','value_pct']}% ({d.loc['fractional','ci_lo_pct']}-{d.loc['fractional','ci_hi_pct']}), "
                         f"any-candidate {d.loc['any_candidate','value_pct']}%, intersection-node {d.loc['intersection_node','value_pct']}%")
        text = (f"Unique-report capture, {params['period']} period, {params['capacity']} capacity ({int(rows['weeks'].iloc[0])} weeks, "
                f"{int(rows['reports'].iloc[0]):,} reports). " + "; ".join(lists) + ". Capture definitions are not interchangeable: any-candidate is "
                "optimistic, fractional is the expected value, and intersection-node assumes an inspection covers a block's end intersections.")
    else:
        parts = []
        for model, g in rows.groupby("model"):
            d = g.set_index("metric")["value_pct"]
            parts.append(f"{model}: precision@1% {d['precision_at_1pct']}%, report capture {d['report_capture_at_1pct']}%")
        text = (f"{params['target']} target, {params['period']} period: " + "; ".join(parts) + ". Strict and multi-block targets are different "
                "questions and their precision values must not be compared with each other.")
    return _wrap(text, rows, template, params, df, model="local-deterministic (no language model)", mode="local")


def _wrap(text, rows, template, params, df, model, mode, usage=None) -> dict:
    return {"brief": text, "mode": mode, "unverified_numbers": verify_numbers(text, rows), "provenance": {
        "model": model, "template": template, "parameters": params, "evidence_version": evidence_version(df),
        "query_sha256": hashlib.sha256((TEMPLATES[template][0] + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:16],
        "rows_returned": int(len(rows)), "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "usage": usage, "status": "draft explanation; numbers verified against query rows" if not verify_numbers(text, rows) else "draft; contains unverified numbers"}}


# ------------------------------------------------------------------ Snowflake client
class SnowflakeEvidence:
    def __init__(self, post=None, max_requests: int | None = None, cache_dir: Path | None = None):
        self._post = post or requests.post
        self.max_requests = max_requests if max_requests is not None else int(os.environ.get("ROADWATCH_SNOWFLAKE_MAX_REQUESTS", "10"))
        self.used = 0
        self.cache_dir = cache_dir or CACHE_DIR

    @property
    def enabled(self) -> bool:
        return config.flag("snowflake") and not config.missing_credentials("snowflake")

    def _base(self) -> str:
        return f"https://{config.secret('SNOWFLAKE_ACCOUNT')}.snowflakecomputing.com"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {config.secret('SNOWFLAKE_PAT')}", "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
                "Content-Type": "application/json", "Accept": "application/json", "User-Agent": "roadwatch/0.2"}

    def _context(self) -> dict:
        ctx = {}
        for k, v in (("warehouse", "SNOWFLAKE_WAREHOUSE"), ("database", "SNOWFLAKE_DATABASE"), ("schema", "SNOWFLAKE_SCHEMA"), ("role", "SNOWFLAKE_ROLE")):
            if config.secret(v):
                ctx[k] = config.secret(v)
        return ctx

    def query(self, template: str, params: dict) -> pd.DataFrame:
        check_request(template, params)
        sql, names, _ = TEMPLATES[template]
        body = {"statement": sql, "timeout": 30, **self._context(), "bindings": {str(i + 1): {"type": "TEXT", "value": params[n]} for i, n in enumerate(names)}}
        r = self._post(self._base() + "/api/v2/statements", json=body, headers=self._headers(), timeout=TIMEOUT_S)
        r.raise_for_status()
        j = r.json()
        cols = [c["name"].lower() for c in j["resultSetMetaData"]["rowType"]]
        df = pd.DataFrame(j["data"], columns=cols)
        for c in df.columns:
            if c.endswith("_pct") or c in ("weeks", "reports"):
                df[c] = pd.to_numeric(df[c])
        return df

    def brief(self, df_local: pd.DataFrame, template: str, params: dict) -> dict:
        """Remote brief; falls back to the deterministic local brief (with a note) on any problem or limit."""
        check_request(template, params)
        if not self.enabled:
            out = local_brief(df_local, template, params)
            out["note"] = "Snowflake is disabled or credentials are missing; deterministic local brief shown."
            return out
        key = hashlib.sha256(json.dumps([template, params, evidence_version(df_local), os.environ.get("SNOWFLAKE_CORTEX_MODEL", DEFAULT_MODEL)]).encode()).hexdigest()[:24]
        cp = self.cache_dir / f"{key}.json"
        if cp.exists():
            out = json.loads(cp.read_text(encoding="utf-8"))
            out["note"] = "Served from cache (no Snowflake call)."
            return out
        if self.used >= self.max_requests:
            out = local_brief(df_local, template, params)
            out["note"] = f"Session limit of {self.max_requests} Snowflake briefs reached; local brief shown."
            return out
        model = os.environ.get("SNOWFLAKE_CORTEX_MODEL", DEFAULT_MODEL)
        try:
            self.used += 1
            rows = self.query(template, params)
            if rows.empty:
                raise ValueError("query returned no rows")
            payload = {"model": model, "max_completion_tokens": 400, "temperature": 0,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": json.dumps({"rows": rows.to_dict("records"), "parameters": params,
                                                                           "evidence_version": evidence_version(df_local)}, default=str)}]}
            r = self._post(self._base() + "/api/v2/cortex/v1/chat/completions", json=payload, headers=self._headers(), timeout=TIMEOUT_S)
            r.raise_for_status()
            j = r.json()
            text = j["choices"][0]["message"]["content"].strip()
        except Exception as e:
            out = local_brief(df_local, template, params)
            out["note"] = f"Snowflake request failed ({type(e).__name__}); deterministic local brief shown."
            return out
        out = _wrap(text, rows, template, params, df_local, model=model, mode="snowflake-cortex", usage=j.get("usage"))
        if out["unverified_numbers"]:
            out["note"] = "The generated text contains numbers not present in the query rows; treat as unverified draft."
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(out), encoding="utf-8")
        return out


# One-time loader (needs a WRITE role; the reader role used above is read-only). Parameterised: one JSON binding.
def load_sql() -> list[str]:
    cols = "family, period, capacity, list_name, target, model, metric, value_pct, ci_lo_pct, ci_hi_pct, weeks, reports"
    return [f"CREATE TABLE IF NOT EXISTS {TABLE} (evidence_id STRING, family STRING, period STRING, capacity STRING, list_name STRING, target STRING, "
            "model STRING, metric STRING, value_pct FLOAT, ci_lo_pct FLOAT, ci_hi_pct FLOAT, weeks INT, reports INT, evidence_version STRING)",
            f"DELETE FROM {TABLE} WHERE evidence_version = ?",
            f"INSERT INTO {TABLE} SELECT value:evidence_id::STRING, " + ", ".join(f"value:{c.strip()}::" + ("FLOAT" if c.strip().endswith("pct") else "INT" if c.strip() in ("weeks", "reports") else "STRING") for c in cols.split(","))
            + ", ? FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?)))"]
