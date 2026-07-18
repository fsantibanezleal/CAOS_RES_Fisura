"""Severity CONTEXT from published guidance (data + banding; the caveats are first-class output).

Transcribed from the verified research record (dossier 04, section 3.6):

- ACI 224R-01, Control of Cracking in Concrete Structures (American Concrete Institute, reapproved
  guide): the guide table of tolerable crack widths at the tensile face under service conditions.
  The document ITSELF warns that crack width is not always a reliable indicator of corrosion or
  deterioration and calls the table "a general guide". That warning ships with every output.
- EN 1992-1-1:2004 (Eurocode 2), Table 7.1N: recommended limiting calculated crack widths under the
  quasi-permanent combination. Values are Nationally Determined Parameters (the National Annex can
  change them); for X0/XC1 the limit is set for appearance rather than durability. The research
  pass verified the table through secondary engineering sources only, which disagree on the XC1
  row; both readings are represented and the primary-text check is flagged.

FRAMING (binding): these are design/serviceability reference bands for calculated widths, used
here as CONTEXT for measured widths. The lab outputs width percentiles against the bands and NEVER
a structural safety verdict.
"""
from __future__ import annotations

from dataclasses import dataclass

ACI_224R01_CAVEAT = (
    "ACI 224R-01 itself cautions that crack width is not always a reliable indicator of expected "
    "steel corrosion and concrete deterioration; its table is 'a general guide'. Reference bands, "
    "not a safety verdict."
)

EC2_CAVEAT = (
    "EN 1992-1-1 Table 7.1N values are Nationally Determined Parameters (the applicable National "
    "Annex governs); for X0/XC1 the recommended limit addresses appearance, not durability. "
    "UNVERIFIED-primary: secondary sources disagree on whether XC1 sits in the 0.4 or 0.3 row; "
    "both readings are shown until checked against the purchased standard text."
)


@dataclass(frozen=True)
class Band:
    source: str
    exposure: str
    limit_mm: float
    note: str = ""


ACI_224R01_BANDS: tuple[Band, ...] = (
    Band("ACI 224R-01", "dry air or protective membrane", 0.41),
    Band("ACI 224R-01", "humidity, moist air, soil", 0.30),
    Band("ACI 224R-01", "deicing chemicals", 0.18),
    Band("ACI 224R-01", "seawater and seawater spray, wetting and drying", 0.15),
    Band("ACI 224R-01", "water-retaining structures (excluding nonpressure pipes)", 0.10),
)

EC2_71N_BANDS: tuple[Band, ...] = (
    Band("EN 1992-1-1 Table 7.1N", "X0, XC1 (appearance; reading A)", 0.40, "relaxable where appearance is not critical"),
    Band("EN 1992-1-1 Table 7.1N", "XC1 (reading B, per some secondary sources)", 0.30, "see EC2_CAVEAT"),
    Band("EN 1992-1-1 Table 7.1N", "XC2 to XC4, XD, XS", 0.30),
)


def band_widths(width_mm_median: float, width_mm_p95: float) -> dict:
    """Compare measured width percentiles against every band. Pure context; caveats included."""
    rows = []
    for band in ACI_224R01_BANDS + EC2_71N_BANDS:
        rows.append(
            {
                "source": band.source,
                "exposure": band.exposure,
                "limit_mm": band.limit_mm,
                "note": band.note,
                "median_within": bool(width_mm_median <= band.limit_mm),
                "p95_within": bool(width_mm_p95 <= band.limit_mm),
            }
        )
    return {
        "width_mm_median": width_mm_median,
        "width_mm_p95": width_mm_p95,
        "bands": rows,
        "caveats": [ACI_224R01_CAVEAT, EC2_CAVEAT],
        "framing": "reference bands for context; not a structural safety verdict",
    }
