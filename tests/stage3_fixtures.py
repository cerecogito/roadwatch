"""SYNTHETIC TEST FIXTURES for Stage 3. Invented street network and reports - not NYC data.

Network (EPSG:2263 feet):
    1 AVE  x=0      blocks 101 (y 0-300), 102 (300-600), bridge 103 (600-700, rw_type 3, ineligible), 104 (700-1000)
    2 AVE  x=1000   blocks 201 (0-300), 202 (300-600)
    E 1 ST y=0 (301), E 2 ST y=300 (302), E 3 ST y=600 (303), E 4 ST y=1000 (304), all x 0-1000
    DIV RD two carriageways x=2000 (401) and x=2020 (402) between A ST (y=0) and B ST (y=300)
    BELT PKWY highway (900, rw_type 2, ineligible)
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

SEGMENTS = [
    (101, "1 AVE", [(0, 0), (0, 300)], 1), (102, "1 AVE", [(0, 300), (0, 600)], 1),
    (103, "1 AVE", [(0, 600), (0, 700)], 3), (104, "1 AVE", [(0, 700), (0, 1000)], 1),
    (201, "2 AVE", [(1000, 0), (1000, 300)], 1), (202, "2 AVE", [(1000, 300), (1000, 600)], 1),
    (301, "E  1 ST", [(0, 0), (1000, 0)], 1), (302, "E  2 ST", [(0, 300), (1000, 300)], 1),
    (303, "E  3 ST", [(0, 600), (1000, 600)], 1), (304, "E  4 ST", [(0, 1000), (1000, 1000)], 1),
    (401, "DIV RD", [(2000, 0), (2000, 300)], 1), (402, "DIV RD", [(2020, 0), (2020, 300)], 1),
    (501, "A ST", [(1980, 0), (2000, 0)], 1), (502, "A ST", [(2000, 0), (2020, 0)], 1), (503, "A ST", [(2020, 0), (2040, 0)], 1),
    (511, "B ST", [(1980, 300), (2000, 300)], 1), (512, "B ST", [(2000, 300), (2020, 300)], 1), (513, "B ST", [(2020, 300), (2040, 300)], 1),
    (900, "BELT PKWY", [(0, 1000), (-500, 1300)], 2),
]


def centerline(crs: str = "EPSG:2263") -> gpd.GeoDataFrame:
    rows = []
    for pid, name, coords, rw in SEGMENTS:
        rows.append({"physicalid": str(pid), "full_street_name": name, "rw_type": str(rw), "status": "2",
                     "is_eligible_block": rw == 1, "l_zip": "11299", "r_zip": "11299", "boroughcode": "3",
                     "geometry": LineString(coords)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def report(rid, address_type, street=None, c1=None, c2=None, i1=None, i2=None, x=None, y=None, zip_=None,
           created="2025-01-08T10:00:00.000") -> dict:
    return {"report_id": rid, "address_type": address_type, "street_name": street, "cross_street_1": c1,
            "cross_street_2": c2, "intersection_street_1": i1, "intersection_street_2": i2,
            "x": float("nan") if x is None else float(x), "y": float("nan") if y is None else float(y),
            "incident_zip": zip_, "created_date": created}


def reports(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
