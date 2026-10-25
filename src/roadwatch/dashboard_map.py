"""Map layer construction for the dashboard (pure data -> pydeck spec; no Streamlit)."""
from __future__ import annotations

import pandas as pd
import pydeck as pdk

# relative-risk bands (within-week percentile of the model score); colours are ranks, not probabilities
BANDS = [(99.0, "Top 1%", [189, 0, 38]), (95.0, "Top 1-5%", [240, 59, 32]), (80.0, "Top 5-20%", [253, 141, 60]),
         (0.0, "Remaining 80%", [190, 190, 190])]
SELECTED_COLOR = [17, 17, 17]
VIEW = pdk.ViewState(latitude=40.645, longitude=-73.945, zoom=11, pitch=0)


def band_of(pct: float) -> tuple[str, list[int]]:
    for lo, name, col in BANDS:
        if pct >= lo:
            return name, col
    return BANDS[-1][1], BANDS[-1][2]


def layer_frames(paths_df: pd.DataFrame, scores: pd.DataFrame, worklist: pd.DataFrame,
                 min_percentile: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Background (risk-coloured) blocks at/above `min_percentile` and the selected inspection blocks.
    The selected frame is built from the worklist itself, so map features always match the displayed list."""
    m = paths_df.merge(scores, on="block_id", how="inner")
    bg = m[m["risk_percentile"] >= min_percentile].copy()
    bands = bg["risk_percentile"].map(band_of)
    bg["band"] = bands.map(lambda x: x[0])
    bg["color"] = bands.map(lambda x: x[1])
    bg["tooltip"] = ("<b>" + bg["street"].fillna("") + "</b> (ZIP " + bg["zip"].astype(str) + ")<br/>within-week risk rank "
                     + bg["rank"].astype(str) + " (" + bg["band"] + ")")
    sel = worklist[["block_id", "rank", "priority", "full_street_name", "zip", "reports_365d"]].merge(
        paths_df[["block_id", "path"]], on="block_id", how="left")
    sel["color"] = [SELECTED_COLOR] * len(sel)
    sel["tooltip"] = ("<b>Inspection #" + sel["rank"].astype(str) + " (" + sel["priority"] + ")</b><br/>"
                      + sel["full_street_name"].fillna("") + ", ZIP " + sel["zip"].astype(str)
                      + "<br/>reports in last 365 days: " + sel["reports_365d"].astype(int).astype(str))
    return bg[["block_id", "path", "color", "tooltip", "risk_percentile"]], sel[["block_id", "path", "color", "tooltip", "rank"]]


def deck(bg: pd.DataFrame, sel: pd.DataFrame) -> pdk.Deck:
    layers = [
        pdk.Layer("PathLayer", bg, get_path="path", get_color="color", width_min_pixels=1, get_width=6, pickable=True,
                  auto_highlight=True),
        pdk.Layer("PathLayer", sel, get_path="path", get_color="color", width_min_pixels=4, get_width=18, pickable=True,
                  auto_highlight=True),
    ]
    return pdk.Deck(layers=layers, initial_view_state=VIEW, map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
                    tooltip={"html": "{tooltip}", "style": {"fontSize": "12px", "background": "#fff", "color": "#111", "border": "3px solid #111", "borderRadius": "10px"}})
