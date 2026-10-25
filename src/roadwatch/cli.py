"""Command line: python -m roadwatch <command>. See README.md."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

from . import pipeline, snapshot
from .config import PROJECT_ROOT, Paths
from .env import load_dotenv, socrata_app_token
from .http import RetryingClient
from .local_files import register
from .report import build_report
from .sources import SOURCES
from .storage import read_json, utc_now

GROUPS = {"core": ["sr311", "centerline", "noaa"],
          "conditional": ["pavement", "workorders", "resurfacing_inhouse", "resurfacing_schedule"],
          "supplementary": ["sr311_unassigned"]}
GROUPS["all"] = GROUPS["core"] + GROUPS["supplementary"] + GROUPS["conditional"]
USER_AGENT = "RoadWatch-pipeline/0.2 (research; NYC Open Data + NOAA NCEI)"


def _expand(names: list[str]) -> list[str]:
    out = []
    for n in names:
        for k in GROUPS.get(n, [n]):
            if k not in SOURCES:
                raise SystemExit(f"unknown source {k!r}; choose from {sorted(SOURCES)} or {sorted(GROUPS)}")
            if k not in out:
                out.append(k)
    return out


def _client(kind: str, token: str | None) -> RetryingClient:
    headers = {"User-Agent": USER_AGENT}
    if kind == "socrata" and token:
        headers["X-App-Token"] = token
    return RetryingClient(headers=headers)


def _setup_logging(paths: Paths, verbose: bool) -> Path:
    log_dir = paths.root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{utc_now():%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")])
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return log_file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="roadwatch", description="RoadWatch Stage-2 ingestion and validation")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT, help="project root (default: repository root)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="download (or resume), validate, finalise and promote")
    p.add_argument("sources", nargs="+", help=f"source keys or groups {sorted(GROUPS)}")
    p.add_argument("--new", action="store_true", help="start a new snapshot instead of resuming an unfinished one")
    p.add_argument("--start", type=dt.date.fromisoformat, help="override requested start (time-partitioned sources)")
    p.add_argument("--end", type=dt.date.fromisoformat, help="override exclusive end date")
    p.add_argument("--page-size", type=int)
    p = sub.add_parser("rebuild", help="rebuild processed tables from archived snapshots (no network)")
    p.add_argument("sources", nargs="+")
    p.add_argument("--snapshot", help="snapshot id (default: latest finalized)")
    sub.add_parser("register-local", help="checksum and register the original local files (no moving)")
    sub.add_parser("report", help="write outputs/data_quality/data_quality_report.md")
    sub.add_parser("status", help="list snapshots and their state")
    sub.add_parser("stage3", help="block index, report matching, weekly labels, diagnostics (offline)")
    p = sub.add_parser("stage4", help="features | select | test | evaluate (offline)")
    p.add_argument("step", choices=["features", "select", "test", "evaluate", "uncertainty"])
    p.add_argument("--only", nargs="*", help="test step: restrict to these experiment ids")
    sub.add_parser("stage6", help="precompute dashboard inputs (map paths, unified policy summary)").add_argument(
        "step", choices=["build"])
    p = sub.add_parser("stage5", help="compare | geography | worklists | volume | all | worklist (single week)")
    p.add_argument("step", choices=["compare", "geography", "validation-predictions", "policy-design", "policy-test", "capture-definitions", "capacity-guidance", "alternative-worklists", "worklists", "volume", "all", "worklist"])
    p.add_argument("--week", type=dt.date.fromisoformat, help="worklist: forecast week start (Monday)")
    p.add_argument("--capacity", type=float, default=0.01, help="worklist: share of eligible blocks (e.g. 0.01)")
    p.add_argument("--policy", help="worklist: Stage-4 experiment id (default from config/stage5.json)")
    p.add_argument("--diversify", action="store_true", help="worklist: apply the configured ZIP cap")
    p.add_argument("--rule", choices=["config", "topk", "nonadjacent"], default="config",
                   help="worklist: selection rule; 'config' = as decided in policy_design.json, 'topk'/'nonadjacent' override it "
                        "(dashboard default list = --policy primary__LGBM --rule topk)")
    args = ap.parse_args(argv)

    paths = Paths(args.root.resolve())
    log_file = _setup_logging(paths, args.verbose)
    log = logging.getLogger("roadwatch")
    load_dotenv(paths.root / ".env")

    if args.cmd == "ingest":
        token = socrata_app_token()
        if not token:
            log.warning("SOCRATA_APP_TOKEN not set: using anonymous access (lower rate limits; retries/back-off apply)")
        failures = []
        for key in _expand(args.sources):
            spec = SOURCES[key]
            try:
                res = pipeline.ingest(spec, paths, _client(spec.kind, token), start=args.start, end=args.end,
                                      page_size=args.page_size, new=args.new,
                                      token_used=bool(token) and spec.kind == "socrata")
            except Exception as e:  # one source failing must not stop the others
                log.exception("%s: ingestion error: %s", key, e)
                failures.append(key)
                continue
            if isinstance(res, snapshot.SnapshotResult):
                failures.append(key)
            else:
                log.info("%s: snapshot %s validation=%s promoted=%s rows=%d report=%s", key, res.snapshot_id,
                         res.validation_status, res.promoted, res.rows, res.report_path)
                if res.validation_status == "fail":
                    failures.append(key)
        log.info("log written to %s", log_file)
        core_failed = [k for k in failures if k in GROUPS["core"]]
        if failures:
            log.error("sources not successfully ingested: %s", failures)
        return 1 if core_failed else (2 if failures else 0)
    if args.cmd == "rebuild":
        status = 0
        for key in _expand(args.sources):
            res = pipeline.rebuild(SOURCES[key], paths, args.snapshot)
            log.info("%s: rebuilt from %s validation=%s promoted=%s rows=%d", key, res.snapshot_id,
                     res.validation_status, res.promoted, res.rows)
            status |= res.validation_status == "fail"
        return status
    if args.cmd == "register-local":
        reg = register(paths)
        for e in reg["files"]:
            log.info("%s: %s, %s, sha256=%s", e["file"], e["extract_status"], e["pipeline_use"], (e.get("sha256") or "missing")[:16])
        return 0
    if args.cmd == "report":
        md, reports = build_report(paths)
        log.info("wrote %s (%d sources)", md, len(reports))
        return 0
    if args.cmd == "stage3":
        from . import stage3, stage3_diagnostics
        manifest = stage3.run(paths)
        log.info("stage3 outputs: %s", sorted(manifest["outputs_sha256"]))
        for v, s in manifest["label_summaries"].items():
            log.info("labels %-17s positive=%d negative=%d unknown=%d", v, s["positive"], s["negative"], s["unknown"])
        files = stage3_diagnostics.write_all(paths)
        log.info("diagnostics: %s", [str(f) for f in files])
        return 0
    if args.cmd == "stage4":
        from . import stage4
        fn = {"features": stage4.step_features, "select": stage4.step_select, "evaluate": stage4.step_evaluate,
              "uncertainty": stage4.step_uncertainty}.get(args.step)
        res = stage4.step_test(paths, args.only) if args.step == "test" else fn(paths)
        log.info("stage4 %s done: %s", args.step, json.dumps(res, default=str)[:2000])
        return 0
    if args.cmd == "stage6":
        from . import dashboard_data
        log.info("stage6 build: %s", dashboard_data.build(paths))
        return 0
    if args.cmd == "stage5":
        from . import stage5
        if args.step == "worklist":
            if not args.week:
                raise SystemExit("--week is required for the worklist step")
            files = stage5.single_worklist(paths, args.week, args.capacity, args.policy, args.diversify, args.rule)
            log.info("worklist written: %s", [str(f) for f in files])
            return 0
        fn = {"compare": stage5.step_compare, "geography": stage5.step_geography, "worklists": stage5.step_worklists,
              "volume": stage5.step_volume, "all": stage5.step_all, "validation-predictions": stage5.step_validation_predictions, "policy-design": stage5.step_policy_design, "policy-test": stage5.step_policy_test, "capture-definitions": stage5.step_capture_definitions, "capacity-guidance": stage5.step_capacity_guidance, "alternative-worklists": stage5.step_alternative_worklists}[args.step]
        res = fn(paths)
        if args.step == "compare":
            stage5.write_report_tables(paths)
        log.info("stage5 %s done: %s", args.step, json.dumps(res, default=str)[:1500])
        return 0
    if args.cmd == "status":
        for key, spec in SOURCES.items():
            for s in snapshot.list_snapshots(paths, spec):
                st = read_json(s / "_state.json")
                print(f"{key:22s} {st['snapshot_id']}  status={st['status']:11s} "
                      f"range={st.get('requested_start')}..{st['requested_end_exclusive']}  "
                      f"failed={[k for k, v in st.get('partition_results', {}).items() if v != 'ok']}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
