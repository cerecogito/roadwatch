"""Ablation chart and experiment manifest from the saved comparison files (no model is run here)."""
from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import Paths  # noqa: E402
from ..storage import sha256_file, utc_iso  # noqa: E402


def ablation_plot(paths: Paths) -> Path:
    o = paths.root / "outputs" / "multimodal"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, stage in zip(axes, ("validation", "test")):
        r = json.loads((o / f"comparison_{stage}_lgbm.json").read_text(encoding="utf-8"))
        names, vals, lo, hi = [], [], [], []
        for sid, v in r["sets"].items():
            c = next(x for x in v["capacities"] if x["capacity"] == 0.01)
            names.append(sid)
            vals.append(100 * c["precision"])
            d = c.get("d_precision_ci95")
            lo.append(0 if d is None else 100 * d[0])
            hi.append(0 if d is None else 100 * d[1])
        colors = ["#2a78d6" if n == "F0" else "#eb6834" for n in names]
        ax.bar(names, vals, color=colors, width=0.45)
        for i, (n, v) in enumerate(zip(names, vals)):
            ax.text(i, v + 0.15, f"{v:.2f}%", ha="center", fontsize=10)
            if n != "F0":
                ax.text(i, v / 2, f"paired diff vs F0\n{lo[i]:+.2f} to {hi[i]:+.2f} pp\n(95% CI)", ha="center", color="white", fontsize=8.5)
        ax.set_title(f"{stage.capitalize()} weeks ({r['weeks']}): precision at 1% capacity", fontsize=10.5, loc="left")
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_xticks(range(len(names)), [f"{n}\n{'production features' if n == 'F0' else '+ 311 water/sewer complaints'}" for n in names], fontsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Feature-group ablation (strict single-block target, LightGBM, frozen hyper-parameters)", fontsize=10, y=1.02)
    fig.tight_layout()
    f = o / "ablation_precision.png"
    fig.savefig(f, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f


def manifest(paths: Paths, tests_summary: str = "") -> Path:
    o = paths.root / "outputs" / "multimodal"
    a = paths.processed / "multimodal"
    raw = sorted((paths.raw / "multimodal" / "infra311").glob("*/manifest.json"))
    fits = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted(o.glob("fits_*.json"))}
    man = {
        "generated_utc": utc_iso(), "branch": "experiment/multimodal", "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "pyarrow", "shapely", "rasterio")},
        "frozen_inputs": {"selection_json_sha256": sha256_file(paths.processed / "stage4" / "selection.json"),
                          "features_manifest_sha256": sha256_file(paths.processed / "stage4" / "features_manifest.json")},
        "sources": {"infra311_raw_manifest": str(raw[-1]) if raw else None, "infra311_raw_manifest_sha256": sha256_file(raw[-1]) if raw else None,
                    "infra311_parquet_sha256": sha256_file(a / "infra311.parquet"), "infra_features_sha256": sha256_file(a / "infra_features.parquet"),
                    "displacement": "not downloaded (Earthdata login required; see multimodal_data_audit.md)",
                    "terrain": "not processed (1 GB/min unattainable host throughput; see multimodal_data_audit.md)", "permits": "excluded by audit"},
        "evaluation": {"validation": ["2025-01-06", "2025-06-30"], "test_exploratory": ["2025-07-07", "2026-10-12"],
                       "capacities": [0.005, 0.01, 0.02], "bootstrap": "paired, resampling weeks, 1000 resamples, seed 11",
                       "label": "strict single-block (primary)", "model_params": "frozen production LightGBM params; CatBoost untuned (defaults noted in code)"},
        "predictions": {f.name: sha256_file(f) for f in sorted((a / "predictions").glob("*.parquet"))},
        "fits": {k: {s: {"seconds_total": v["seconds_total"], "n_features": len(v["features"]), "predictions_sha256": v["predictions_sha256"]} for s, v in d.items()} for k, d in fits.items()},
        "tests": tests_summary,
    }
    f = o / "experiment_manifest.json"
    f.write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    return f
