"""Streamlit rendering of the optional 'Assistant & integrations' view. Imported only when that view is opened; makes no
remote call until the user presses a button."""
from __future__ import annotations

import json

import altair as alt
import pandas as pd

from .. import dashboard_data as dd
from . import config, solana_receipt as sr, tiger_data as td
from .evidence_tools import CAPS, EvidenceTools
from .gemini_assistant import Assistant
from .snowflake_evidence import SnowflakeEvidence, TEMPLATES, build_curated

BADGE = {"verified": "Verified with the real service", "configured_unverified": "Configured, not yet verified", "disabled": "Disabled (default)",
         "missing_credentials": "Enabled, credentials missing", "failed": "Last real test failed", "unsupported": "Not supported here"}


def render(st, paths, week, cap, list_key, wl) -> None:
    st.info("Everything on this page is optional and separate from the prediction pipeline. Nothing here can change a score, a ranking or a worklist, "
            "or issue a repair order. No remote service is contacted until you press a button.", icon=":material/extension:")
    cap_s = {0.005: "0.5%", 0.01: "1%", 0.02: "2%"}.get(cap, "1%")

    st.subheader("Integration status")
    rows = [{"integration": s.label, "status": BADGE[s.status], "detail": s.detail} for s in config.all_statuses()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("'Verified' appears only after a real smoke test against the service has been recorded; mocked tests never set it.")

    # ---------------- assistant
    st.subheader("Evidence assistant")
    a_state = st.session_state.setdefault("rw_assistant", Assistant(EvidenceTools(paths)))
    st.caption(("Gemini explains figures retrieved by five read-only evidence tools. " if a_state.remote_enabled else
                "Local mode (no Gemini key): answers are assembled from the same read-only evidence tools. ")
               + "Try: 'Why was block {} selected on {}?', 'Compare the policies in March 2026', 'What limitations should an inspector consider?'".format(
                   int(wl["block_id"].iloc[0]) if len(wl) else 52878, week))
    q = st.text_input("Ask about a block, a policy or a metric", key="rw_q", max_chars=600)
    if st.button("Ask", key="rw_ask") and q:
        with st.spinner("Retrieving evidence..."):
            ans = a_state.ask(q, week=str(week), capacity=cap_s, list_key=list_key)
        st.write(ans.text)
        st.caption(f"Answer mode: {ans.mode}" + (f" | {ans.note}" if ans.note else ""))
        if ans.unverified_numbers:
            st.warning("These numbers could not be traced to retrieved evidence: " + ", ".join(ans.unverified_numbers))
        if ans.sources:
            st.caption("Sources: " + "; ".join(ans.sources))
        if ans.tool_calls:
            with st.expander("Evidence tools called"):
                st.json(ans.tool_calls)

    # ---------------- temporal analytics
    st.subheader("Temporal analytics")
    use_tiger = False
    if config.status("tiger").status in ("configured_unverified", "verified"):
        use_tiger = st.checkbox("Query Tiger Data instead of local files", value=False)
    ls = st.session_state.setdefault("rw_local_series", td.LocalSeries(paths=paths))
    series, src = ls, "local saved outputs"
    if use_tiger:
        try:
            series, src = td.TigerSeries(td.connect()), "Tiger Data (TimescaleDB)"
        except Exception as e:
            st.error(f"Tiger Data unavailable ({type(e).__name__}); using local files.")
    st.caption(f"Source: {src}. Reports are 311 pothole reports, not defects.")
    mr = series.monthly_reports()
    st.altair_chart(alt.Chart(mr).mark_bar().encode(x=alt.X("bucket:T", title="month"), y=alt.Y("reports:Q", title="reports per month"),
                                                    tooltip=["bucket:T", "reports", "peak_week_reports", "weeks"]).properties(height=240,
                    title="Monthly Brooklyn pothole reports (March 2026 surge)"), use_container_width=True)
    block = st.number_input("Block id for rank history", min_value=1, value=int(wl["block_id"].iloc[0]) if len(wl) else 52878, step=1)
    rh = series.rank_history(int(block))
    if rh.empty:
        st.caption("This block was never in the top 500 ranked blocks of any week, so no rank history is stored.")
    else:
        st.altair_chart(alt.Chart(rh).mark_line(point=True).encode(x=alt.X("week_start:T", title="forecast week"),
                        y=alt.Y("rank:Q", scale=alt.Scale(reverse=True), title="within-week rank (1 = highest)"), tooltip=["week_start:T", "rank", "score"]).properties(
                        height=220, title=f"Risk-rank history of block {int(block)} (weeks in the top 500)"), use_container_width=True)
    mm = series.monthly_metrics("Strict single-block LightGBM (top-K)", "1pct")
    if len(mm):
        mm2 = mm.melt("month", ["capture_any", "capture_assigned"], var_name="definition", value_name="capture")
        st.altair_chart(alt.Chart(mm2).mark_line(point=True).encode(x=alt.X("month:T"), y=alt.Y("capture:Q", axis=alt.Axis(format="%")), color="definition:N",
                        tooltip=["month:T", "definition", alt.Tooltip("capture:Q", format=".1%")]).properties(height=220,
                        title="Monthly report capture of the default list at 1% (any candidate vs assigned block)"), use_container_width=True)

    # ---------------- evidence brief
    st.subheader("Evidence brief")
    st.caption("A reproducible brief from a curated, versioned table of validated evaluation results (Snowflake Cortex when enabled; otherwise a "
               "deterministic local brief). Only approved queries run; every number is checked against the query rows.")
    cur = st.session_state.setdefault("rw_curated", build_curated(paths))
    t = st.selectbox("Brief type", list(TEMPLATES), format_func=lambda k: TEMPLATES[k][2])
    if t == "capture_by_period":
        p = {"period": st.selectbox("Period", ["test", "validation", "march_2026"]), "capacity": st.selectbox("Capacity", ["0.5pct", "1pct", "2pct"], index=1)}
    else:
        p = {"target": st.selectbox("Target", ["primary", "multi_block"]), "period": st.selectbox("Evaluation period", ["overall", "march_2026_stress"])}
    if st.button("Generate evidence brief", key="rw_brief"):
        with st.spinner("Building brief..."):
            out = st.session_state.setdefault("rw_sf", SnowflakeEvidence()).brief(cur, t, p)
        st.write(out["brief"])
        st.caption(f"Mode: {out['mode']}" + (f" | {out.get('note')}" if out.get("note") else ""))
        if out["unverified_numbers"]:
            st.warning("Unverified numbers: " + ", ".join(out["unverified_numbers"]))
        with st.expander("Provenance"):
            st.json(out["provenance"])

    # ---------------- receipts
    st.subheader("Worklist integrity receipt")
    st.caption("A local manifest checksum is not a blockchain transaction. A receipt is shown as confirmed only if a genuine Solana Devnet "
               "transaction exists. The manifest holds checksums and the model version only: no street-level records.")
    csv = dd.to_csv_bytes(wl)
    man = sr.build_manifest(csv, str(week), list_key, cap_s.replace("%", "pct"), str(wl["model_version"].iloc[0]))
    st.code(sr.memo_text(man), language="text")
    st.caption("Digest above includes the generation time and changes each run; saving freezes it.")
    if st.button("Save manifest (local only)", key="rw_man"):
        sr.RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
        mp = sr.RECEIPT_DIR / f"{week}_{list_key}_{cap_s.replace('%', 'pct')}.manifest.json"
        mp.write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
        st.session_state["rw_manifest"] = str(mp)
        st.success(f"Saved {mp.name}. Verify later with: python -I -m roadwatch.integrations.solana_receipt verify --manifest <file> --worklist <csv>")
    mp = st.session_state.get("rw_manifest")
    if mp:
        rcp = sr.Path(mp).with_name(sr.Path(mp).name.replace(".manifest.json", ".receipt.json"))
        if rcp.exists():
            rc = json.loads(rcp.read_text(encoding="utf-8"))
            st.success(f"Confirmed on Solana Devnet ({rc['confirmation']}). [Open in explorer]({rc['explorer']})")
        elif config.status("solana").status in ("configured_unverified", "verified"):
            ok = st.checkbox("I confirm I want to submit this digest to Solana Devnet (uses a Devnet-funded key I control)")
            if st.button("Anchor on Devnet", key="rw_anchor", disabled=not ok):
                with st.spinner("Submitting and waiting for confirmation..."):
                    res = sr.anchor(sr.Path(mp), confirm=True)
                (st.success if res["status"] == "confirmed" else st.error)(res.get("explorer") or f"{res['status']}: {res.get('reason')}")
        else:
            st.caption("On-chain anchoring is off: " + config.status("solana").detail)

    # ---------------- human factors lab
    st.subheader("Human Factors Lab (experimental, disabled)")
    st.info(config.status("presage").detail + " No camera is accessed and no measurement is produced. A consent-based study would need a supported SDK; "
            "see the integration plan.", icon=":material/science:")
