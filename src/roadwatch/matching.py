"""Deterministic hierarchical matching of 311 pothole reports to Brooklyn street blocks.

Evidence hierarchy (follows the verified semantics of each 311 address_type):
- ADDRESS / PLACENAME: geocoded XY lies on the centerline (audit: p50 3 ft) AND street_name names the block's street.
  1. coord_name            nearest block of the SAME street within snap_max_ft (two consistent evidences)
  2. coord_name_cross      tied same-street blocks resolved by the (street, cross_street_1, cross_street_2) topology
  3. blockface_* (below)   no same-street block near the point -> street/cross-street topology, flagged
- BLOCKFACE: street_name + cross_street_1 + cross_street_2 = the block between two cross streets (no XY since 2022):
  blockface_names (exact block) > blockface_chain (same-street segments, possibly through bridges, between the cross
  streets; eligible blocks on the chain) > blockface_cross_proximity (cross street not topologically adjacent but
  within cross_proximity_ft) > blockface_one_sided (only one cross street resolvable).
- INTERSECTION: intersection_street_1/2 (order carries no meaning: alphabetical in 94.9% of rows). The node where both
  streets meet (nearest to XY when present); candidates = eligible blocks of either street at that node.

Ties are never broken by row order: the primary pick minimises sha256(salt|report_id|block_id) - stable, independent of
input order and uncorrelated with block attributes. A second salt gives the sensitivity pick. Ambiguity is flagged.
"""
from __future__ import annotations

import hashlib
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .blocks import Network
from .names import NON_STREET_CROSS, street_key


@dataclass(frozen=True)
class MatchParams:
    snap_max_ft: float = 50.0          # coordinate evidence radius (audit ADDRESS p99 47.5 ft; Stage-3 calibration)
    tie_tol_ft: float = 1.0            # candidates within this distance of the nearest are tied
    node_max_ft: float = 50.0          # max point-to-node distance for intersection reports
    cross_proximity_ft: float = 100.0  # max block-to-cross-street distance for blockface_cross_proximity
    max_chain_segments: int = 8
    primary_salt: str = "roadwatch-primary-v1"
    alt_salt: str = "roadwatch-alt-v1"
    alias_min_reports: int = 5
    alias_min_share: float = 0.9
    alias_max_dist_ft: float = 15.0


def hashed_pick(report_id: str, candidates: list[int], salt: str) -> int:
    """Deterministic, order-independent choice among tied candidate block ids."""
    return min(sorted(candidates), key=lambda b: hashlib.sha256(f"{salt}|{report_id}|{b}".encode()).hexdigest())


class Matcher:
    def __init__(self, net: Network, params: MatchParams = MatchParams(), aliases: dict[str, str] | None = None):
        self.net, self.p = net, params
        self.aliases = dict(aliases or {})
        b = net.blocks
        self.bid = b["block_id"].astype("int64").to_numpy()
        self.bname = b["name_key"].to_numpy()
        self.bfrom, self.bto = b["node_from"].to_numpy(), b["node_to"].to_numpy()
        self.bgeom = b.geometry.to_numpy()
        self.tree = STRtree(self.bgeom)
        seg = net.segments
        self.sfrom, self.sto = seg["node_from"].to_numpy(), seg["node_to"].to_numpy()
        self.sgeom = seg.geometry.to_numpy()
        self.node_tree = STRtree(shapely.points(net.node_xy))
        self.nodes_by_name: dict[str, set[int]] = {}
        for node, names in net.node_streets.items():
            for nm in names:
                self.nodes_by_name.setdefault(nm, set()).add(node)

    def key(self, name) -> str | None:
        k = street_key(name)
        return self.aliases.get(k, k) if k else None

    # ---------------- topology helpers ----------------
    def _touches(self, node: int, name: str) -> bool:
        return name in self.net.node_streets.get(int(node), ())

    def _known(self, name: str | None) -> bool:
        return bool(name) and name not in NON_STREET_CROSS and name in self.net.all_names

    def _chain(self, s: str, a: str, b: str) -> list[int]:
        """Eligible blocks on all shortest chains of same-street segments (any rw_type, e.g. through a bridge) from a
        node touching `a` to a node touching `b`."""
        adj: dict[int, list[tuple[int, int]]] = {}
        for i in self.net.segments_by_name.get(s, []):
            f, t = int(self.sfrom[i]), int(self.sto[i])
            adj.setdefault(f, []).append((t, i))
            adj.setdefault(t, []).append((f, i))
        starts = [n for n in adj if self._touches(n, a)]
        dist = {n: 0 for n in starts}
        parents: dict[int, list[tuple[int, int]]] = {n: [] for n in starts}
        q, goals, best = deque(starts), [], None
        while q:
            n = q.popleft()
            if best is not None and dist[n] >= best:
                continue
            for m, sg in adj.get(n, []):
                d = dist[n] + 1
                if d > self.p.max_chain_segments:
                    continue
                if m not in dist:
                    dist[m], parents[m] = d, [(n, sg)]
                    q.append(m)
                    if self._touches(m, b):
                        goals.append(m)
                        best = d if best is None else min(best, d)
                elif dist[m] == d:
                    parents[m].append((n, sg))
        if best is None:
            return []
        segs, stack, seen = set(), [g for g in goals if dist[g] == best], set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            for prev, sg in parents.get(n, []):
                segs.add(sg)
                stack.append(prev)
        return sorted(self.net.seg_to_block[sg] for sg in segs if sg in self.net.seg_to_block)

    def _dist_to_street(self, block_pos: int, name: str) -> float:
        segs = self.net.segments_by_name.get(name, [])
        if not segs:
            return np.inf
        return float(shapely.distance(self.bgeom[block_pos], self.sgeom[segs]).min())

    def blockface(self, s: str | None, a: str | None, b: str | None) -> tuple[list[int], str | None, str | None]:
        same = self.net.blocks_by_name.get(s, []) if s else []
        if not same:
            if s and s in self.net.all_names:
                return [], None, "street_not_in_block_universe"
            return [], None, "street_not_found"
        ka, kb = self._known(a), self._known(b)
        if ka and kb:
            direct = [i for i in same if (self._touches(self.bfrom[i], a) and self._touches(self.bto[i], b))
                      or (self._touches(self.bfrom[i], b) and self._touches(self.bto[i], a))]
            if direct:
                return direct, "blockface_names", None
            chain = self._chain(s, a, b)
            if chain:
                return chain, "blockface_chain", None
            scored = []
            for anchor, other in ((a, b), (b, a)):
                for i in same:
                    if self._touches(self.bfrom[i], anchor) or self._touches(self.bto[i], anchor):
                        scored.append((self._dist_to_street(i, other), i))
            scored = [(d, i) for d, i in scored if d <= self.p.cross_proximity_ft]
            if scored:
                dmin = min(d for d, _ in scored)
                return sorted({i for d, i in scored if d <= dmin + self.p.tie_tol_ft}), "blockface_cross_proximity", None
            return [], None, "cross_streets_not_adjacent"
        one = a if ka else b if kb else None
        if one:
            cands = [i for i in same if self._touches(self.bfrom[i], one) or self._touches(self.bto[i], one)]
            other = b if one == a else a
            if other == "DEADEND" and len(cands) > 1:
                dead = [i for i in cands if min(self.net.node_degree[self.bfrom[i]], self.net.node_degree[self.bto[i]]) == 1]
                cands = dead or cands
            if cands:
                return cands, "blockface_one_sided", None
        return [], None, "cross_streets_not_found"

    def pair_compatible(self, a: str | None, b: str | None) -> bool:
        """Is there an eligible Brooklyn block bounded by a node on `a` and a node on `b`? (plausibility only)"""
        na, nb = self.nodes_by_name.get(a or "", set()), self.nodes_by_name.get(b or "", set())
        for n in na:
            for i in self.net.blocks_by_node.get(n, []):
                other = self.bto[i] if self.bfrom[i] == n else self.bfrom[i]
                if int(other) in nb:
                    return True
        return False

    # ---------------- per-report matching ----------------
    def _result(self, rid, status, method, cands, confidence, reason=None, corroborated=None, pt=None):
        ids = sorted({int(self.bid[i]) for i in cands})
        pick = hashed_pick(rid, ids, self.p.primary_salt) if ids else None
        alt = hashed_pick(rid, ids, self.p.alt_salt) if ids else None
        dist = np.nan
        if pt is not None and pick is not None:
            dist = float(shapely.distance(pt, self.bgeom[int(np.flatnonzero(self.bid == pick)[0])]))
        if status == "matched" and len(ids) > 1:
            status = "ambiguous"
        return {"status": status, "method": method, "confidence": confidence if status == "matched" else
                ("ambiguous" if status == "ambiguous" else "none"), "n_candidates": len(ids),
                "candidate_block_ids": ids, "assigned_block_id": pick, "alt_block_id": alt,
                "assigned_distance_ft": dist, "name_corroborated": corroborated, "reason": reason}

    def match_one(self, r: dict, near: list[tuple[int, float]] | None) -> dict:
        rid, at = r["report_id"], r["address_type"]
        pt = shapely.Point(r["x"], r["y"]) if pd.notna(r["x"]) and pd.notna(r["y"]) else None
        if at == "INTERSECTION":
            return self._intersection(rid, self.key(r["intersection_street_1"]), self.key(r["intersection_street_2"]), pt)
        s, a, b = self.key(r["street_name"]), self.key(r["cross_street_1"]), self.key(r["cross_street_2"])
        if not s and not a and not b:
            return self._result(rid, "unmatched", None, [], "none", "no_location_fields", pt=pt)
        if pt is not None and s and near:
            same = [(i, d) for i, d in near if self.bname[i] == s]
            if same:
                dmin = min(d for _, d in same)
                tied = [i for i, d in same if d <= dmin + self.p.tie_tol_ft]
                if len(tied) == 1:
                    return self._result(rid, "matched", "coord_name", tied, "high", corroborated=True, pt=pt)
                bf, _, _ = self.blockface(s, a, b)
                inter = sorted(set(tied) & set(bf))
                if len(inter) == 1:
                    return self._result(rid, "matched", "coord_name_cross", inter, "high", corroborated=True, pt=pt)
                return self._result(rid, "matched", "coord_name", tied, "high", corroborated=True, pt=pt)
        bf, method, reason = self.blockface(s, a, b)
        if bf:
            conf = "high" if method == "blockface_names" else "medium"
            if pt is not None:  # coordinates present but no same-street block within snap_max_ft
                return self._result(rid, "matched", method, bf, "medium", reason="coordinates_not_corroborating",
                                    corroborated=False, pt=pt)
            return self._result(rid, "matched", method, bf, conf)
        if reason == "street_not_in_block_universe":
            return self._result(rid, "out_of_universe", None, [], "none", reason, pt=pt)
        return self._result(rid, "unmatched", None, [], "none", reason, pt=pt)

    def _intersection(self, rid, s1, s2, pt) -> dict:
        if not s1 or not s2:
            return self._result(rid, "unmatched", None, [], "none", "intersection_streets_missing", pt=pt)
        nodes = sorted(self.nodes_by_name.get(s1, set()) & self.nodes_by_name.get(s2, set()))
        corroborated, method = None, None
        if pt is not None and nodes:
            d = shapely.distance(pt, shapely.points(self.net.node_xy[nodes]))
            ok = d <= self.p.node_max_ft
            if ok.any():
                dmin = d[ok].min()
                nodes = [n for n, dd in zip(nodes, d) if dd <= dmin + self.p.tie_tol_ft]
                corroborated = True
            else:
                corroborated = False  # the two streets meet in Brooklyn, but not near the reported point
        if pt is not None and corroborated is not True:
            # Geosupport resolved the intersection to a point; our name gazetteer could not confirm it (local name
            # variants, e.g. Ditmas Park numbered streets). Use the nearest centerline node if it is within node_max_ft.
            nd = self.nearest_node(pt)
            if nd is not None:
                nodes, corroborated, method = [nd], False, "intersection_coord"
        if not nodes:
            known = [x for x in (s1, s2) if x in self.net.all_names]
            elig = [x for x in (s1, s2) if x in self.net.eligible_names]
            reason = "intersection_not_found" if len(known) == 2 else "street_not_found"
            status = "out_of_universe" if known and not elig else "unmatched"
            return self._result(rid, status, None, [], "none", reason, pt=pt)
        cands = sorted({i for n in nodes for i in self.net.blocks_by_node.get(n, []) if self.bname[i] in (s1, s2)})
        if not cands:
            cands = sorted({i for n in nodes for i in self.net.blocks_by_node.get(n, [])})
        if not cands:
            return self._result(rid, "out_of_universe", None, [], "none", "no_eligible_block_at_node", pt=pt)
        method = method or ("intersection_node" if len(nodes) == 1 else "intersection_multi_node")
        conf = "medium" if corroborated is False else "high"
        reason = "coordinates_not_corroborating" if corroborated is False else None
        return self._result(rid, "matched", method, cands, conf, reason=reason, corroborated=corroborated, pt=pt)

    def nearest_node(self, pt) -> int | None:
        """Nearest centerline node within node_max_ft that has at least one eligible block; ties -> lowest node id."""
        idx = self.node_tree.query(pt, predicate="dwithin", distance=self.p.node_max_ft)
        idx = [int(i) for i in idx if self.net.blocks_by_node.get(int(i))]
        if not idx:
            return None
        d = shapely.distance(pt, shapely.points(self.net.node_xy[idx]))
        return min(zip(d.round(6), idx))[1]

    def near_blocks(self, xy: np.ndarray, radius: float) -> dict[int, list[tuple[int, float]]]:
        has = ~np.isnan(xy).any(axis=1)
        near: dict[int, list[tuple[int, float]]] = {}
        if has.any():
            pts = shapely.points(xy[has])
            qi, bi = self.tree.query(pts, predicate="dwithin", distance=radius)
            dist = shapely.distance(pts[qi], self.bgeom[bi])
            rows = np.flatnonzero(has)[qi]
            for r, i, d in zip(rows, bi, dist):
                near.setdefault(int(r), []).append((int(i), float(d)))
        return near

    def match(self, reports: pd.DataFrame) -> pd.DataFrame:
        """`reports`: report_id, address_type, street_name, cross_street_1, cross_street_2, intersection_street_1,
        intersection_street_2, x, y (EPSG:2263 ft). Returns one row per input row, same order and index."""
        near = self.near_blocks(reports[["x", "y"]].to_numpy(dtype=float), self.p.snap_max_ft)
        out = [self.match_one(rec, near.get(k)) for k, rec in enumerate(reports.to_dict("records"))]
        res = pd.DataFrame(out, index=reports.index)
        res["nearest_block_distance_ft"] = [min((d for _, d in near[k]), default=np.nan) if k in near else np.nan
                                            for k in range(len(reports))]
        return res


def derive_aliases(reports: pd.DataFrame, net: Network, params: MatchParams = MatchParams()) -> pd.DataFrame:
    """Coordinate-validated street-name aliases (labelled examples = geocoded ADDRESS reports).

    For ADDRESS reports whose point lies within alias_max_dist_ft of exactly one eligible block, pair the report's
    street key with that block's key. An alias report_key -> cscl_key is accepted only if report_key never occurs in
    the Brooklyn centerline, it has >= alias_min_reports such reports, and >= alias_min_share of them sit on blocks
    with the same CSCL name."""
    m = Matcher(net, params)
    a = reports[(reports["address_type"] == "ADDRESS")].reset_index(drop=True)
    near = m.near_blocks(a[["x", "y"]].to_numpy(dtype=float), params.alias_max_dist_ft)
    pairs = []
    for k, cand in near.items():
        names = {m.bname[i] for i, _ in cand}
        rk = street_key(a.at[k, "street_name"])
        if rk and len(names) == 1 and rk not in net.all_names:
            pairs.append((rk, next(iter(names))))
    rows = []
    for rk, cnt in Counter(p[0] for p in pairs).items():
        top, n_top = Counter(c for r, c in pairs if r == rk).most_common(1)[0]
        share = n_top / cnt
        rows.append({"report_key": rk, "cscl_key": top, "n_reports": cnt, "share": round(share, 4),
                     "accepted": cnt >= params.alias_min_reports and share >= params.alias_min_share})
    return pd.DataFrame(rows, columns=["report_key", "cscl_key", "n_reports", "share", "accepted"]).sort_values(
        ["accepted", "n_reports"], ascending=[False, False]).reset_index(drop=True)
