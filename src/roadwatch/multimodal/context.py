"""Display-only context for a block-week: nearby 311 water/sewer COMPLAINTS and experiment results.

Nothing here feeds the ranking model or the decision engine. If the experimental tables have not been built the functions
return None and callers simply show nothing.
"""
from __future__ import annotations

import datetime as dt
import json

import pyarrow as pa
import pyarrow.dataset as pads

from ..config import Paths
from .ingest import GROUPS

LABELS = {"water_main_leak": "water-main break or leak complaints", "hydrant_water": "hydrant running/leaking complaints",
          "drain_flood": "catch-basin, manhole or street-flooding complaints"}


def infra_context(paths: Paths, block_id: int, week: dt.date) -> dict | None:
    f = paths.processed / "multimodal" / "infra_features.parquet"
    if not f.exists():
        return None
    ds = pads.dataset(str(f), format="parquet")
    flt = (pads.field("block_id") == int(block_id)) & (pads.field("origin") == pa.scalar(dt.datetime(week.year, week.month, week.day), pa.timestamp("ns")))
    t = ds.to_table(filter=flt).to_pandas()
    if t.empty:
        return None
    r = t.iloc[0]
    return {"week": str(week), "block_id": int(block_id), "groups": {
        g: {"label": LABELS[g], "on_block_365d": int(r[f"inf_{g}_n365"]), "on_block_90d": int(r[f"inf_{g}_n90"]),
            "within_300ft_365d": int(r[f"inf_{g}_area365"]), "days_since_latest": int(r[f"inf_{g}_days_since"])} for g in GROUPS},
        "note": "311 complaints, not confirmed leaks or flooding. Context only: not used by the ranking model (no demonstrated predictive gain).",
        "sources": ["data/processed/multimodal/infra_features.parquet (NYC 311 erm2-nwe9, water and sewer complaint types)"]}


def infra_line(ctx: dict) -> str:
    parts = []
    for g, v in ctx["groups"].items():
        if v["on_block_365d"] or v["within_300ft_365d"]:
            parts.append(f"{v['on_block_365d']} {v['label']} on the block ({v['within_300ft_365d']} within 300 ft) in the past year")
    return "; ".join(parts) if parts else "no water or sewer complaints on or within 300 ft of this block in the past year"


def experiment_results(paths: Paths) -> dict | None:
    d = paths.root / "outputs" / "multimodal"
    out = {}
    for name in ("validation_lgbm", "test_lgbm", "validation_catboost"):
        f = d / f"comparison_{name.split('_', 1)[0]}_{name.split('_', 1)[1]}.json"
        if f.exists():
            out[name] = json.loads(f.read_text(encoding="utf-8"))
    return out or None
