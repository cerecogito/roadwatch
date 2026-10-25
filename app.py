"""RoadWatch dashboard - historical evaluation and decision-support prototype (NOT a live municipal deployment).

Launch:  .venv\\Scripts\\python.exe -m streamlit run app.py
All business logic lives in src/roadwatch/dashboard_data.py, dashboard_map.py and inspection.py; this file only renders.
"""
from __future__ import annotations

import datetime as dt
import json

import altair as alt
import pandas as pd
import streamlit as st

from roadwatch import dashboard_data as dd
from roadwatch import dashboard_map as dm
from roadwatch.config import Paths

st.set_page_config(page_title="RoadWatch - Brooklyn inspection prioritisation", page_icon=":material/route:", layout="wide")
PATHS = Paths()
CAP_LABELS = {"0.5% (118 blocks)": 0.005, "1% (235 blocks)": 0.01, "2% (469 blocks)": 0.02, "Custom number of blocks": None}
ACTION = "Inspect, not an automatic repair order."


# ---------------- cached loaders (precomputed outputs only) ----------------
@st.cache_data(show_spinner=False)
def weeks():
    return dd.available_weeks(PATHS)


@st.cache_data(show_spinner=False)
def worklist(week, cap, list_key):
    return dd.worklist(PATHS, week, cap, list_key)


@st.cache_data(show_spinner="Selecting blocks with the Stage-5 engine...")
def custom_worklist(week, k, list_key):
    return dd.custom_worklist(PATHS, week, k, list_key)


@st.cache_data(show_spinner=False)
def scores(week, policy):
    return dd.week_scores(PATHS, week, policy)


@st.cache_data(show_spinner=False)
def capture_defs():
    return dd.capture_definitions(PATHS)


@st.cache_resource(show_spinner="Loading block geometry...")
def block_paths():
    return dd.block_paths(PATHS)


@st.cache_data(show_spinner=False)
def summary():
    return dd.policy_summary(PATHS), dd.policy_monthly(PATHS)


@st.cache_data(show_spinner=False)
def volume():
    return dd.volume_series(PATHS)


@st.cache_data(show_spinner=False)
def evidence():
    return dd.evidence(PATHS)


@st.cache_data(show_spinner=False)
def provenance():
    return dd.provenance(PATHS)


def missing(e: dd.MissingOutput):
    st.error(f"A required saved output is missing: `{e.path}`.\n\nGenerate it with: `{e.hint}`")
    st.stop()


# ---------------- styling (colours/fonts in .streamlit/config.toml; this adds the chunky card look) ----------------
INK, TOMATO = "#111111", "#E0532C"
st.html(f"""<style>
.stApp h1, .stApp h2, .stApp h3, [data-testid="stMetricValue"], .rw-headline {{
  font-stretch: 125%; font-weight: 900 !important; text-transform: uppercase; letter-spacing: -.01em; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ border-right: 3px solid {INK}; }}
[data-testid="stSidebar"] h2 {{ font-size: 1.15rem; }}
/* white cards, black outline, tomato offset shadow */
[data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stVegaLiteChart"],
[data-testid="stDeckGlJsonChart"], [data-testid="stExpander"] details {{
  background: #fff; border: 3px solid {INK}; border-radius: 16px; box-shadow: 7px 7px 0 {TOMATO}; }}
[data-testid="stMetric"] {{ padding: 14px 18px; }}
[data-testid="stMetric"] label p {{ text-transform: uppercase; font-weight: 700; font-size: .75rem; letter-spacing: .04em; }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
[data-testid="stDataFrame"], [data-testid="stDeckGlJsonChart"] {{ overflow: hidden; }}
[data-testid="stVegaLiteChart"] {{ padding: 12px; }}
[data-testid="stVegaLiteChart"] svg {{ background-color: #fff !important; }}
[data-testid="stAlertContainer"] {{ border: 3px solid {INK}; border-radius: 14px; box-shadow: 5px 5px 0 {INK}; background: #fff; color: {INK}; }}
/* tomato buttons with black outline + hard shadow */
.stButton button, .stDownloadButton button {{
  background: {TOMATO}; color: #fff; border: 3px solid {INK}; border-radius: 10px; box-shadow: 4px 4px 0 {INK};
  font-stretch: 125%; font-weight: 900; text-transform: uppercase; transition: transform .08s, box-shadow .08s; }}
.stButton button:hover, .stDownloadButton button:hover {{ color: #fff; border-color: {INK}; transform: translate(-2px, -2px); box-shadow: 6px 6px 0 {INK}; }}
.stButton button:active, .stDownloadButton button:active {{ transform: translate(3px, 3px); box-shadow: 1px 1px 0 {INK}; }}
/* view switcher as the site nav: uppercase links, active one becomes the tomato "SHOP" pill */
.st-key-nav {{ align-items: center; }}
.st-key-nav [role="radiogroup"] {{ flex-wrap: wrap; gap: 6px 8px; justify-content: center; }}
.st-key-nav label[data-testid="stRadioOption"] > div > div:not([data-testid]) {{ display: none; }}
.st-key-nav label[data-testid="stRadioOption"] {{ padding: 8px 14px; border: 3px solid transparent; border-radius: 10px; cursor: pointer; }}
.st-key-nav label[data-testid="stRadioOption"] p {{ font-stretch: 125%; font-weight: 900; text-transform: uppercase; font-size: .85rem; }}
.st-key-nav label[data-selected="true"] {{ background: {TOMATO}; border-color: {INK}; box-shadow: 4px 4px 0 {INK}; }}
.st-key-nav label[data-selected="true"] p {{ color: #fff; }}
/* logo + headline */
.rw-logo {{ text-align: center; line-height: .9; margin-top: -1.5rem; }}
.rw-logo b {{ font-size: clamp(2.6rem, 6vw, 4.2rem); font-stretch: 125%; font-weight: 900; color: #fff;
  -webkit-text-stroke: 7px {INK}; paint-order: stroke fill; text-shadow: 5px 5px 0 {INK}; }}
.rw-logo i {{ display: inline-block; font-style: normal; background: {INK}; color: #fff; padding: 2px 14px 4px;
  border-radius: 8px; font-weight: 900; font-stretch: 125%; transform: rotate(-3deg); margin-top: 4px; }}
.rw-headline {{ text-align: center; font-size: clamp(1.4rem, 3.4vw, 2.6rem); line-height: 1.05; margin: 18px 0 22px; color: {INK}; }}
</style>""")
st.html('<div class="rw-logo"><b>ROADWATCH&#10022;</b><br><i>BROOKLYN</i></div>'
        '<div class="rw-headline">Pothole-report inspection prioritisation</div>')
st.warning("**Historical evaluation and decision-support prototype - not a live municipal deployment.** "
           "Predictions forecast *311 pothole reports*, not confirmed road failures. Capacity scenarios are experimental, "
           "not NYC budgets. No cost, savings or failures-prevented figures exist or are shown.", icon=":material/warning:")

try:
    available = weeks()
except dd.MissingOutput as e:
    missing(e)

with st.sidebar:
    st.header("Scenario")
    requested = st.query_params.get("week")
    default_week, msg = dd.resolve_week(requested, available)
    if msg:
        st.info(msg)
    week = st.selectbox("Forecast week (Monday)", available, index=available.index(default_week),
                        format_func=lambda d: d.strftime("%a %d %b %Y"))
    cap_label = st.radio("Inspection capacity (manual; not set by the volume forecast)", list(CAP_LABELS), index=1)
    cap = CAP_LABELS[cap_label]
    k_custom = None
    if cap is None:
        k_custom = int(st.number_input("Inspection slots this week", min_value=1, max_value=2000, value=235, step=5))
    keys = list(dd.LISTS)
    list_key = st.radio("Inspection list", keys, index=keys.index(dd.default_list(PATHS)),
                        format_func=lambda k: dd.LISTS[k][0])
    st.caption("Which list is best depends on whether inspecting a block also covers its end intersections - "
               "see 'Decision-policy comparison'.")
    st.header("Map")
    min_pct = st.select_slider("Show background blocks in the top...", options=[1, 5, 20, 50, 100], value=20,
                               format_func=lambda v: f"{v}% of risk ranking")
    st.caption("Colours are within-week **ranks** of the model score, not probabilities.")
    st.divider()
    st.caption("Test period 2025-07-07 to 2026-10-12 (67 weekly forecasts, monthly walk-forward retraining).")

try:
    wl = worklist(week, cap, list_key) if cap is not None else custom_worklist(week, k_custom, list_key)
except dd.MissingOutput as e:
    missing(e)

VIEWS = {"map": "Inspection worklist & map", "policy": "Decision-policy comparison", "volume": "Weekly volume & scenarios",
         "evidence": "Evidence & limitations", "data": "Data provenance", "extras": "Assistant & integrations (optional)"}
req_view = st.query_params.get("view", "map")
view = st.radio("View", list(VIEWS.values()), index=list(VIEWS).index(req_view) if req_view in VIEWS else 0, horizontal=True,
                label_visibility="collapsed")   # only the active view is computed (faster than st.tabs)

# ============================== 1. worklist & map ==============================
if view == VIEWS["map"]:
    mv = wl["model_version"].iloc[0] if len(wl) else "n/a"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast week starting", week.strftime("%d %b %Y"))
    c2.metric("Blocks to inspect", f"{len(wl):,}")
    c3.metric("Capacity scenario", cap_label.split(" (")[0] if cap is not None else f"{k_custom} slots")
    c4.metric("Priority P1 / P2 / P3", " / ".join(str(int((wl["priority"] == p).sum())) for p in ("P1", "P2", "P3")))
    st.caption(f"List: **{dd.LISTS[list_key][0]}** | selection rule: **{wl['selection_rule'].iloc[0] if len(wl) else 'n/a'}**"
               + (" - a block is skipped if it shares an intersection node with a block already selected." if dd.LISTS[list_key][3] else "."))
    strict = dd.LISTS[list_key][2].startswith("primary")
    where = ("to be assigned to this block (single-block target: each report counts on exactly one block)" if strict else
             "on the block or at one of its plausible candidate locations (multi-block target)")
    st.info(f"**Recommended action for every listed block: {ACTION}** A high rank means a 311 pothole report is relatively "
            f"likely {where}; it is not proof of a physical defect.", icon=":material/search:")
    st.caption(f"Model version: `{mv}` | horizon: reports created {week} to {week + dt.timedelta(days=6)} (local time) | "
               "inputs: 311 reports created before the forecast date minus 2 days, static street attributes.")

    try:
        bg, sel = dm.layer_frames(block_paths(), scores(week, dd.LISTS[list_key][2]), wl, 100 - min_pct)
    except dd.MissingOutput as e:
        missing(e)
    left, right = st.columns([3, 2])
    with left:
        st.pydeck_chart(dm.deck(bg, sel), height=560)
        legend = " ".join(f"<span style='color:rgb{tuple(c)}'>&#9632;</span> {n}" for _, n, c in dm.BANDS)
        st.markdown(f"{legend} &nbsp; <span style='color:rgb{tuple(dm.SELECTED_COLOR)}'>&#9632;</span> **selected for inspection**",
                    unsafe_allow_html=True)
        st.caption(f"Map shows {len(sel):,} selected blocks (black, thick) and {len(bg):,} background blocks. "
                   "Basemap tiles load from CARTO if online; block lines need no network.")
    with right:
        q = st.text_input("Search street, ZIP or block id", "")
        pri = st.multiselect("Priority", ["P1", "P2", "P3"], default=["P1", "P2", "P3"])
        shown = dd.filter_worklist(wl, q, pri)
        st.dataframe(
            shown[["rank", "priority", "full_street_name", "zip", "model_score_relative", "reports_365d", "reports_30d",
                  "days_since_last_report", "length_m", "travel_lanes", "block_id"]],
            hide_index=True, height=470, use_container_width=True,
            column_config={"model_score_relative": st.column_config.NumberColumn("relative score", format="%.3f",
                                                                                  help="ranking score; not a calibrated probability"),
                           "full_street_name": "street", "reports_365d": "reports 365 d", "reports_30d": "reports 30 d",
                           "days_since_last_report": "days since last", "length_m": st.column_config.NumberColumn("length m", format="%.0f")})
        st.caption(f"{len(shown):,} of {len(wl):,} listed blocks match. Click column headers to sort.")
    st.subheader("Why each block is listed")
    pick = st.selectbox("Block", shown["rank"].tolist() if len(shown) else [None],
                        format_func=lambda r: "-" if r is None else f"#{r} {wl.loc[wl['rank'] == r, 'full_street_name'].iloc[0]}")
    if pick is not None:
        r = wl[wl["rank"] == pick].iloc[0]
        st.markdown(f"**#{r['rank']} {r['full_street_name']} (ZIP {r['zip']}, block {r['block_id']}) - {r['priority']}**")
        st.write(r["rationale"])
        st.write(f"Model drivers (LightGBM contributions, supporting evidence only, not causes): {r['model_drivers']}")
        st.caption(f"Road attributes: length {r['length_m']:.0f} m | travel lanes {r['travel_lanes']} | width {r['street_width_ft']} ft | "
                   f"truck route: {'yes' if pd.notna(r['truck_route_type']) else 'no'} | bike lane: {'yes' if pd.notna(r['bike_lane']) else 'no'}")
        try:
            from roadwatch.multimodal import context as mm_ctx             # optional experimental table; never used by the ranking
            ctx = mm_ctx.infra_context(PATHS, int(r["block_id"]), week)
        except Exception:                                                   # missing optional data must never break the page
            ctx = None
        if ctx:
            st.caption("Context only (not used by the model): " + mm_ctx.infra_line(ctx) + ". These are 311 complaints, not confirmed leaks or flooding.")
    cap_tag = f"{cap * 100:g}pct" if cap is not None else f"K{k_custom}"          # no '%' in file names (Stage 7)
    stem = f"roadwatch_worklist_{week}_{list_key}_{cap_tag}"
    d1, d2 = st.columns(2)
    d1.download_button("Download worklist (CSV)", dd.to_csv_bytes(wl), f"{stem}.csv", "text/csv", use_container_width=True)
    d2.download_button("Download worklist (GeoJSON)", dd.to_geojson_bytes(wl), f"{stem}.geojson", "application/geo+json",
                       use_container_width=True)
    with st.expander("Data limitations for this list"):
        st.markdown(
            "- The target is a **311 pothole report** (public complaint), not a verified defect; reports reflect reporting behaviour as well as road condition.\n"
            + ("- The *strict single-block* target counts each report on one block: its unique match, or for ambiguous reports a "
               "deterministic pseudo-random pick among the plausible candidates. In the test period 44% of reports were ambiguous, so "
               "for them the label marks one of several equally plausible blocks.\n" if strict else
               "- The *multi-block* target counts a report on every **plausible candidate block**: blocks meeting at the reported intersection, "
               "or, for ambiguous blockface descriptions (about 22% of ambiguous reports in the test period), alternative locations along the "
               "street - **not necessarily an intersection**. In the test period 44% of reports could not be assigned to a single block.\n") +
            "- Scores rank blocks within a week; probabilities are not reliable estimates of weekly volume (March 2026 predicted/observed: "
            "0.35 strict single-block LightGBM, 0.23 multi-block LightGBM).\n"
            "- Street attributes are a 2026 centerline snapshot; history uses 311 data published by the forecast date (2-day lag).")

# ============================== 2. policy comparison ==============================
if view == VIEWS["policy"]:
    try:
        summ, monthly = summary()
    except dd.MissingOutput as e:
        missing(e)
    pcap = cap if cap is not None else 0.01
    if cap is None:
        st.info("Policy evidence exists for the three configured capacities; showing 1%.")
    period = st.radio("Period", ["Whole test period (67 weeks)", "March 2026 surge (5 weeks)"], horizontal=True)
    per = "test" if period.startswith("Whole") else "march_2026"
    st.warning("**Audit finding (Stage 7): which list captures more reports depends on what counts as 'captured'.** "
               "Non-adjacent lists win only if inspecting one block also covers a pothole reported at its end intersection. "
               "Under *expected* capture (an ambiguous report's true block is equally likely to be any candidate), they capture "
               "fewer reports than plain top-K, and the strict single-block top-K list is best. The strict list beats plain "
               "top-K under every definition, on both validation and test weeks.", icon=":material/rule:")
    try:
        cdef, cmeta = capture_defs()
    except dd.MissingOutput as e:
        missing(e)
    cper = st.radio("Evidence period for the four-definition table",
                    ["Test (67 weeks)", "Validation (26 weeks, policy-design period)", "March 2026 surge (5 weeks)"], horizontal=True)
    cper_key = {"Test": "test", "Validation": "validation", "March": "march_2026"}[cper.split(" ")[0]]
    st.markdown(f"**Unique reports captured at {pcap * 100:g}% capacity under four definitions (95% CI)**")
    st.dataframe(dd.capture_table(cdef, pcap, cper_key), hide_index=True, use_container_width=True)
    with st.expander("Definitions"):
        for k, v in cmeta["definitions"].items():
            st.markdown(f"- **{k}**: {v}")
        st.caption(f"Test-period reports: {cmeta['test_report_composition']}")
    st.markdown("**Same lists: precision and capture (whole test period or March surge)**")
    st.dataframe(dd.policy_table(summ, pcap, per), hide_index=True, use_container_width=True)
    st.caption("All rows evaluate saved lists on identical weeks. 95% CIs: percentile bootstrap over weeks (unpaired here; "
               "paired differences vs plain top-K are in outputs/stage5/capture_definitions.csv). March intervals rest on only "
               "5 weeks and understate uncertainty. Precision uses each list's own target labels and is not comparable across targets.")
    s = summ[(summ["capacity"] == pcap) & (summ["period"] == per)]
    long = pd.concat([pd.DataFrame({"policy": s["policy"], "definition": "any candidate block", "capture": s["capture_any_candidate"],
                                    "lo": s["capture_any_ci95"].map(lambda c: c[0]), "hi": s["capture_any_ci95"].map(lambda c: c[1])}),
                      pd.DataFrame({"policy": s["policy"], "definition": "assigned block only", "capture": s["capture_assigned_block"],
                                    "lo": s["capture_assigned_ci95"].map(lambda c: c[0]), "hi": s["capture_assigned_ci95"].map(lambda c: c[1])})])
    base = alt.Chart(long).encode(y=alt.Y("policy:N", sort=[p[0] for p in dd.POLICIES], title=None,
                                          axis=alt.Axis(labelLimit=360)),
                                  yOffset="definition:N", color=alt.Color("definition:N", title="report counted as captured if..."))
    st.altair_chart((base.mark_bar().encode(x=alt.X("capture:Q", axis=alt.Axis(format="%"), title="unique reports captured")) +
                     base.mark_rule(color="black").encode(x="lo:Q", x2="hi:Q")).properties(height=320), use_container_width=True)
    st.markdown(
        "**How to read this.** *Any candidate block* counts a report as captured if any plausible location is inspected - for "
        "intersection reports any block meeting at the intersection; for ambiguous blockface reports any alternative block along the "
        "street (*not* necessarily an intersection). This is optimistic, and the non-adjacent rule is built to win it: it puts at most "
        "one slot on each intersection. *Assigned block* counts only the block Stage 3 assigned; because that assignment is a "
        "deterministic pseudo-random pick among candidates, it is an unbiased estimate of *fractional* (expected) capture. "
        "*Intersection-node coverage* gives full credit for intersection reports when any adjoining block is inspected, and expected "
        "credit otherwise. The non-adjacent rule was chosen on validation weeks with the any-candidate objective, but the idea arose "
        "after test-period results were seen, so its test-period numbers are not an untouched confirmation.")
    mm = monthly[monthly["capacity"] == pcap]
    st.altair_chart(alt.Chart(mm).mark_line(point=True).encode(
        x=alt.X("month:N", title=None), y=alt.Y("capture_any_candidate:Q", axis=alt.Axis(format="%"), title="capture (any candidate)"),
        color=alt.Color("policy:N", sort=[p[0] for p in dd.POLICIES]),
        tooltip=["policy", "month", alt.Tooltip("capture_any_candidate:Q", format=".1%"),
                 alt.Tooltip("capture_assigned_block:Q", format=".1%"), "reports_total"]).properties(height=320, title="Monthly unique-report capture"),
        use_container_width=True)

# ============================== 3. volume ==============================
if view == VIEWS["volume"]:
    try:
        fr, te, res = volume()
    except dd.MissingOutput as e:
        missing(e)
    st.error("**The weekly-volume model did not reliably outperform simple baselines** (test MAE 64.2 vs 64.1 for 'same as last 7 days'; "
             "difference vs 4-week mean not significant) and **under-forecast the March 2026 surge by ~42%**. It is shown as context only; "
             "inspection capacity stays manually configured.", icon=":material/trending_down:")
    hist = fr[(fr["week_status"] == "observable") & (pd.to_datetime(fr["origin"]) >= "2021-01-04")].copy()
    hist["origin"] = pd.to_datetime(hist["origin"])
    t = te[te["week_status"] == "observable"].copy()
    t["origin"] = pd.to_datetime(t["origin"])
    band = alt.Chart(t).mark_area(opacity=0.25, color=TOMATO).encode(x="origin:T", y="lo:Q", y2="hi:Q")
    fc = alt.Chart(t).mark_line(color=TOMATO, strokeDash=[4, 3]).encode(x="origin:T", y=alt.Y("pred:Q", title="reports per week"),
                                                                             tooltip=[alt.Tooltip("origin:T"), alt.Tooltip("pred:Q", format=".0f"),
                                                                                      alt.Tooltip("lo:Q", format=".0f"), alt.Tooltip("hi:Q", format=".0f")])
    act = alt.Chart(hist).mark_line(color=INK).encode(x=alt.X("origin:T", title="forecast week"), y="y:Q",
                                                            tooltip=[alt.Tooltip("origin:T"), alt.Tooltip("y:Q", title="actual reports")])
    st.altair_chart((act + band + fc).properties(height=380, title="Actual weekly Brooklyn pothole reports (black) and walk-forward forecast with 80% interval (red)"),
                    use_container_width=True)
    rows = []
    for per_, lab in (("overall", "All test weeks"), ("excluding_march_2026", "Excluding March 2026"), ("march_2026_stress", "March 2026")):
        r = res[per_]
        rows.append({"period": lab, "forecast MAE": round(r["pred"]["mae"], 1), "last-7-days MAE": round(r["naive_last7"]["mae"], 1),
                     "4-week-mean MAE": round(r["mean_4wk"]["mae"], 1), "forecast bias": round(r["pred"]["bias_mean_pred_minus_actual"], 1),
                     "80% interval coverage": f"{100 * r['interval_coverage_selected']:.0f}%", "mean actual": round(r["pred"]["mean_actual"], 0)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    wk = t[t["origin"].dt.date == week]
    if len(wk):
        w0 = wk.iloc[0]
        st.metric(f"Context for {week}: forecast reports (80% interval)", f"{w0['pred']:.0f} ({w0['lo']:.0f}-{w0['hi']:.0f})",
                  help="Contextual only. Actual reports for this historical week are shown in the chart.")
    st.caption("Forecast and actuals count REPORTS, which are not physical defects or required repairs. Capacity scenarios are experimental.")

# ============================== 4. evidence ==============================
if view == VIEWS["evidence"]:
    try:
        ev = evidence()
    except dd.MissingOutput as e:
        missing(e)
    st.subheader("Walk-forward test (2025-07-07 to 2026-10-12, 16 monthly retrains)")
    tab4 = ev["stage4_table"]
    show = tab4[tab4["period"].isin(["overall", "march_2026_stress"])][
        ["variant", "model", "period", "positives", "prevalence", "average_precision", "top_1pct_precision", "top_1pct_lift_vs_B1",
         "top_1pct_report_capture", "calibration_in_the_large_ratio"]].copy()
    show["prevalence"] = (100 * show["prevalence"]).round(2).astype(str) + "%"
    for c in ("average_precision", "top_1pct_precision", "top_1pct_lift_vs_B1", "top_1pct_report_capture", "calibration_in_the_large_ratio"):
        show[c] = show[c].round(3)
    st.dataframe(show.rename(columns={"calibration_in_the_large_ratio": "predicted/observed volume"}), hide_index=True, use_container_width=True)
    bt = pd.DataFrame(ev["bootstrap"])[["comparison", "p_at_1pct_a", "p_at_1pct_b", "p_at_1pct_diff_ci95", "ap_diff_ci95"]]
    bt["p_at_1pct_diff_ci95"] = bt["p_at_1pct_diff_ci95"].map(lambda c: f"{c[0]:+.3f} to {c[1]:+.3f}")
    bt["ap_diff_ci95"] = bt["ap_diff_ci95"].map(lambda c: f"{c[0]:+.3f} to {c[1]:+.3f}")
    st.markdown("**Paired week-level bootstrap (95% CI of differences)**")
    st.dataframe(bt, hide_index=True, use_container_width=True)
    cdef, _ = capture_defs()
    st.markdown("**Unique reports captured on test weeks: optimistic (any candidate) / conservative (fractional)**")
    c1, c2, c3 = st.columns(3)
    for col, capv in zip((c1, c2, c3), (0.005, 0.01, 0.02)):
        x = cdef[(cdef["capacity"] == capv) & (cdef["period"] == "test")].set_index("list")
        dec, stk = x.loc["Stage-5 decision: multi-block LightGBM + non-adjacent"], x.loc["Strict single-block LightGBM (top-K)"]
        col.metric(f"Strict single-block top-K at {capv * 100:g}%", f"{100 * stk['any_candidate']:.1f}% / {100 * stk['fractional']:.1f}%")
        col.metric(f"Stage-5 decision at {capv * 100:g}%", f"{100 * dec['any_candidate']:.1f}% / {100 * dec['fractional']:.1f}%")
    st.subheader("What these numbers do and do not mean")
    cal = tab4[(tab4["model"] == "LGBM")].set_index(["variant", "period"])["calibration_in_the_large_ratio"]
    st.markdown(
        "- **Reports captured != defects found != failures prevented.** We measure whether future 311 reports fell on inspected blocks. "
        "We have no data on inspection findings, repair costs or prevented failures, so none are claimed.\n"
        "- **Ranking, not probability.** Scores order blocks within a week. Predicted/observed report volume (calibration in the "
        f"large) over the test period was {cal[('primary', 'overall')]:.2f} for the strict single-block LightGBM "
        f"({cal[('multi_block', 'overall')]:.2f} multi-block) but {cal[('primary', 'march_2026_stress')]:.2f} "
        f"({cal[('multi_block', 'march_2026_stress')]:.2f}) in March 2026, and it varied widely between months.\n"
        "- **March 2026 stress test.** Weekly reports were ~4x normal; the learned rankings still beat the historical-rate baseline "
        "(e.g. strict single-block top-K captured 16.9% vs 13.1% at 1%, any-candidate; 12.9% vs 9.2% fractional) while volume forecasts "
        "and probabilities failed. March intervals rest on 5 weeks.\n"
        "- **Capture definitions matter.** The non-adjacent rule's advantage exists only if inspecting a block covers its end "
        "intersections; under expected (fractional) capture it is a disadvantage (Stage-7 audit).\n"
        "- **Label uncertainty.** 44% of test-period reports cannot be assigned to one block (40% over 2020-2026; mostly intersections); "
        "23 historical weeks are unknown because some reports lacked locations; matching quality changed when blockface reports lost "
        "coordinates in 2022.\n"
        "- **Reporting bias.** Areas that report less may be under-served; DOT's own repair records track report hotspots (rho 0.77 by ZIP), "
        "pavement ratings do not.\n"
        "- **Multi-block target.** A report counts on every plausible candidate block (intersection-adjacent blocks, or alternative "
        "blockface locations for ambiguous descriptions).")
    st.caption("Sources: outputs/stage4/test_results.csv, bootstrap_comparisons.json, outputs/stage5/*.")

# ============================== 6. optional integrations ==============================
if view == VIEWS["extras"]:
    from roadwatch.integrations import ui as integrations_ui          # imported only for this view
    integrations_ui.render(st, PATHS, week, cap, list_key, wl)

# ============================== 5. provenance ==============================
if view == VIEWS["data"]:
    st.subheader("Source snapshots (immutable manifests)")
    pv = provenance()
    if pv.empty:
        st.warning("No raw manifest index found (data/raw/MANIFEST.jsonl).")
    else:
        st.dataframe(pv, hide_index=True, use_container_width=True)
    st.caption("All data are archived snapshots retrieved on 2026-10-24 from NYC Open Data and NOAA NCEI; nothing is fetched live. "
               "The dashboard reads saved Stage 3-5 outputs only and never retrains models.")
    man = PATHS.processed / "stage6" / "dashboard_manifest.json"
    if man.exists():
        with st.expander("Dashboard input checksums"):
            st.json(json.loads(man.read_text(encoding="utf-8")))
