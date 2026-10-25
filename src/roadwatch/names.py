"""Street-name normalisation shared by 311 reports and CSCL centerline names.

311 publishes Geosupport full forms ("EAST   18 STREET", "MC DONALD AVENUE"); CSCL publishes abbreviated forms
("E 18 ST", "MCDONALD AVE"). Both sides pass through the same functions; equality of `street_key` is the criterion.

Rules:
1. upper-case; apostrophes deleted ("ST MARK'S" -> "ST MARKS"); other punctuation -> space; whitespace collapsed;
2. ordinal suffixes stripped ("18TH" -> "18");
3. full words -> standard abbreviations only (never the reverse): street types per USPS Publication 28 Appendix C1,
   directionals per Appendix B, and FORT -> FT;
4. `street_key` additionally removes spaces, because Geosupport and CSCL differ in particle spacing
   ("MC DONALD" / "MCDONALD", "DE KALB" / "DEKALB"); verified to cause no collisions among Brooklyn CSCL names.
Spelling variants and renamed streets are handled only through the coordinate-validated alias table
(matching.derive_aliases), never by fuzzy matching.
"""
from __future__ import annotations

import re

SUFFIX_ABBREVIATIONS = {
    "AVENUE": "AVE", "STREET": "ST", "ROAD": "RD", "PARKWAY": "PKWY", "PLACE": "PL", "BOULEVARD": "BLVD",
    "HIGHWAY": "HWY", "COURT": "CT", "DRIVE": "DR", "LANE": "LN", "BRIDGE": "BRG", "PLAZA": "PLZ", "CIRCLE": "CIR",
    "TERRACE": "TER", "SQUARE": "SQ", "ALLEY": "ALY", "COVE": "CV", "ESPLANADE": "ESPL", "OVERPASS": "OPAS",
    "TUNNEL": "TUNL", "HEIGHTS": "HTS", "COMMONS": "CMNS", "EXPRESSWAY": "EXPY", "BEACH": "BCH", "CENTER": "CTR",
    "FORT": "FT",
}
DIRECTIONAL_ABBREVIATIONS = {"EAST": "E", "WEST": "W", "NORTH": "N", "SOUTH": "S",
                             "NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW"}
TOKEN_MAP = {**SUFFIX_ABBREVIATIONS, **DIRECTIONAL_ABBREVIATIONS}

# Cross-street values that are not streets (observed in real 311 rows); they can bound a block but never name one.
NON_STREET_CROSS = {"DEADEND", "BEND"}

_ORDINAL = re.compile(r"\b(\d+)(ST|ND|RD|TH)\b")
_NON_ALNUM = re.compile(r"[^A-Z0-9 ]")


def normalize_street(name) -> str | None:
    """Readable normalised form (tokens separated by single spaces)."""
    if not isinstance(name, str):
        return None
    t = _NON_ALNUM.sub(" ", name.upper().replace("'", "").replace("’", ""))
    t = _ORDINAL.sub(r"\1", t)
    tokens = [TOKEN_MAP.get(tok, tok) for tok in t.split()]
    return " ".join(tokens) or None


def street_key(name) -> str | None:
    """Comparison key: normalised form without spaces."""
    n = normalize_street(name)
    return n.replace(" ", "") if n else None
