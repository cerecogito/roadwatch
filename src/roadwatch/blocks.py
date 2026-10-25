"""Brooklyn block index and centerline node topology.

Eligibility (technical plan §2, audit §3): CSCL rows with rw_type = 1 (Street) and status = 2 (Constructed).
Block identifier: CSCL `physicalid` ("intersection to intersection stretch"), unique among eligible Brooklyn rows.
Distances use NAD83 / New York Long Island State Plane, US feet (EPSG:2263), the CSCL source CRS.

Nodes are segment endpoints keyed by coordinates rounded to 0.01 ft. All centerline rows (including highways,
ramps, paths) contribute street names to nodes, so a block can be recognised as bounded by e.g. a ramp; only
eligible rows become blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .names import normalize_street, street_key

PROJECTED_CRS = "EPSG:2263"
FT_TO_M = 0.3048

BLOCK_ATTRIBUTES = [
    "physicalid", "globalid", "b5sc", "full_street_name", "street_name", "stname_label", "pre_type", "post_type",
    "pre_directional", "post_directional", "rw_type", "status", "trafdir", "l_low_hn", "l_high_hn", "r_low_hn",
    "r_high_hn", "l_zip", "r_zip", "l_blockfaceid", "r_blockfaceid", "from_level_code", "to_level_code",
    "streetwidth_num", "number_travel_lanes_num", "number_park_lanes_num", "number_total_lanes_num",
    "posted_speed_num", "segmentlength_num", "truck_route_type", "snow_priority", "bike_lane", "bike_trafdir",
    "rwjurisdiction", "created_date", "modified_date",
]


class BlockIndexError(Exception):
    pass


@dataclass
class Network:
    blocks: gpd.GeoDataFrame                       # eligible blocks, EPSG:2263, indexed 0..n-1
    segments: gpd.GeoDataFrame                     # all centerline rows, EPSG:2263
    node_streets: dict[int, frozenset] = field(repr=False)
    node_xy: np.ndarray = field(repr=False)        # node_id -> (x, y)
    node_degree: np.ndarray = field(repr=False)
    blocks_by_name: dict[str, list[int]] = field(repr=False)      # street_key -> block positions
    blocks_by_node: dict[int, list[int]] = field(repr=False)
    segments_by_name: dict[str, list[int]] = field(repr=False)    # street_key -> segment positions (all rw_types)
    seg_to_block: dict[int, int] = field(repr=False)
    eligible_names: frozenset = field(repr=False)                 # street keys
    all_names: frozenset = field(repr=False)


def _endpoints(geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    merged = shapely.line_merge(geoms)
    starts, ends = [], []
    for g in merged:
        if g.geom_type == "MultiLineString":  # defensive: take outer ends of the first/last part
            starts.append(g.geoms[0].coords[0][:2])
            ends.append(g.geoms[-1].coords[-1][:2])
        else:
            starts.append(g.coords[0][:2])
            ends.append(g.coords[-1][:2])
    return np.asarray(starts), np.asarray(ends)


def build_network(centerline: gpd.GeoDataFrame) -> Network:
    """`centerline` is data/processed/centerline.parquet (or a test frame with the same columns)."""
    if centerline.crs is None:
        raise BlockIndexError("centerline has no CRS")
    seg = centerline.to_crs(PROJECTED_CRS).reset_index(drop=True)
    seg["norm_name"] = seg["full_street_name"].map(normalize_street)
    seg["name_key"] = seg["full_street_name"].map(street_key)
    seg["seg_idx"] = np.arange(len(seg))
    s, e = _endpoints(seg.geometry.to_numpy())
    keys = np.round(np.vstack([s, e]), 2)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    seg["node_from"], seg["node_to"] = inv[: len(seg)], inv[len(seg):]
    degree = np.bincount(inv, minlength=len(uniq))
    node_streets: dict[int, set] = {}
    segs_by_name: dict[str, list[int]] = {}
    for i, (nf, nt, nm) in enumerate(zip(seg["node_from"], seg["node_to"], seg["name_key"])):
        if nm:
            node_streets.setdefault(int(nf), set()).add(nm)
            node_streets.setdefault(int(nt), set()).add(nm)
            segs_by_name.setdefault(nm, []).append(i)

    elig = seg["is_eligible_block"].astype(bool)
    blocks = seg[elig].copy()
    blocks["block_id"] = pd.to_numeric(blocks["physicalid"], errors="coerce").astype("Int64")
    keep = ["block_id", "norm_name", "name_key", "seg_idx", "node_from", "node_to"] + \
           [c for c in BLOCK_ATTRIBUTES if c in blocks.columns]
    blocks = gpd.GeoDataFrame(blocks[keep], geometry=blocks.geometry, crs=PROJECTED_CRS)
    blocks["length_m"] = blocks.length * FT_TO_M
    mid = shapely.line_interpolate_point(blocks.geometry.to_numpy(), 0.5, normalized=True)
    mid_ll = gpd.GeoSeries(mid, crs=PROJECTED_CRS).to_crs("EPSG:4326")
    blocks["mid_lon"], blocks["mid_lat"] = mid_ll.x.to_numpy(), mid_ll.y.to_numpy()
    blocks["geometry_wgs84_wkb"] = shapely.to_wkb(blocks.geometry.to_crs("EPSG:4326").to_numpy())
    blocks = blocks.sort_values("block_id", kind="stable").reset_index(drop=True)

    by_name: dict[str, list[int]] = {}
    by_node: dict[int, list[int]] = {}
    for i, (nm, nf, nt) in enumerate(zip(blocks["name_key"], blocks["node_from"], blocks["node_to"])):
        by_name.setdefault(nm, []).append(i)
        by_node.setdefault(int(nf), []).append(i)
        if nt != nf:
            by_node.setdefault(int(nt), []).append(i)
    return Network(blocks=blocks, segments=seg, node_streets={k: frozenset(v) for k, v in node_streets.items()},
                   node_xy=uniq, node_degree=degree, blocks_by_name=by_name, blocks_by_node=by_node,
                   segments_by_name=segs_by_name,
                   seg_to_block={int(s_): i for i, s_ in enumerate(blocks["seg_idx"])},
                   eligible_names=frozenset(n for n in blocks["name_key"] if n),
                   all_names=frozenset(n for n in seg["name_key"] if n))


def validate_blocks(blocks: gpd.GeoDataFrame, expected_count: int | None, tolerance: int = 0) -> list[dict]:
    """Returns check records; raises BlockIndexError on conditions that make the index unusable."""
    checks = []
    if blocks.crs is None or blocks.crs.to_epsg() != 2263:
        raise BlockIndexError(f"block index must be in EPSG:2263, got {blocks.crs}")
    if blocks["block_id"].isna().any():
        raise BlockIndexError(f"{int(blocks['block_id'].isna().sum())} blocks without an identifier")
    dup = int(blocks["block_id"].duplicated().sum())
    if dup:
        raise BlockIndexError(f"{dup} duplicated block_id values")
    geoms = blocks.geometry.to_numpy()
    missing = int(pd.isna(geoms).sum())
    invalid = int((~shapely.is_valid(geoms[~pd.isna(geoms)])).sum()) if missing < len(geoms) else 0
    empty = int(shapely.is_empty(geoms[~pd.isna(geoms)]).sum()) if missing < len(geoms) else 0
    if missing or invalid or empty:
        raise BlockIndexError(f"invalid block geometry: missing {missing}, invalid {invalid}, empty {empty}")
    checks.append({"check": "ids_unique_geometry_valid", "status": "PASS", "detail": f"{len(blocks)} blocks"})
    if expected_count is not None:
        diff = len(blocks) - expected_count
        if abs(diff) > tolerance:
            raise BlockIndexError(f"block coverage changed: {len(blocks)} vs expected {expected_count} (tolerance {tolerance})")
        checks.append({"check": "coverage_vs_expected", "status": "PASS",
                       "detail": f"{len(blocks)} vs expected {expected_count}"})
    short = int((blocks["length_m"] < 1).sum())
    checks.append({"check": "very_short_blocks", "status": "INFO", "detail": f"{short} blocks shorter than 1 m"})
    return checks
