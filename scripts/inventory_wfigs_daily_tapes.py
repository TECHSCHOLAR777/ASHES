"""Inventory WFIGS Daily 2023-2026 progression tapes (Job C book).

Does not download polygons yet. Attributes only, paged. Writes a machine-readable
inventory so we can gate fires by n_times, span, acres, CONUS, and Daily vs Final.

    python scripts/inventory_wfigs_daily_tapes.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clients.timed_perimeters import WFIGS_DAILY_URL, parse_arcgis_datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.inventory_daily")

OUT = Path("data/training/job_c_2023/tape_inventory.json")
YEARS = (2023, 2024, 2025, 2026)
PAGE = 2000
CONUS_LAT = (24.0, 50.0)
CONUS_LNG = (-125.0, -66.0)
OUT_FIELDS = (
    "attr_UniqueFireIdentifier,poly_PolygonDateTime,poly_GISAcres,poly_MapMethod,"
    "poly_FeatureCategory,poly_IncidentName,poly_IRWINID,attr_InitialLatitude,"
    "attr_InitialLongitude,attr_IncidentSize,attr_IncidentTypeCategory,attr_POOState"
)


def _in_conus(lat: float | None, lng: float | None, state: str | None) -> bool:
    if state in {"AK", "HI", "PR", "GU", "AS", "VI", "MP"}:
        return False
    if lat is None or lng is None:
        return True  # keep; we'll drop later if still unknown
    return CONUS_LAT[0] <= lat <= CONUS_LAT[1] and CONUS_LNG[0] <= lng <= CONUS_LNG[1]


def page_year(client: httpx.Client, year: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    where = (
        f"attr_UniqueFireIdentifier LIKE '{year}-%' AND "
        "(poly_FeatureCategory='Wildfire Daily Fire Perimeter' OR "
        "poly_FeatureCategory='Wildfire Final Fire Perimeter')"
    )
    while True:
        params = {
            "f": "json",
            "where": where,
            "outFields": OUT_FIELDS,
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": PAGE,
            "outSR": 4326,
        }
        resp = client.get(WFIGS_DAILY_URL, params=params, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        batch = data.get("features") or []
        logger.info("year=%s offset=%d batch=%d", year, offset, len(batch))
        for feat in batch:
            rows.append(feat.get("attributes") or {})
        if len(batch) < PAGE:
            break
        offset += PAGE
        if offset > 200_000:
            break
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for attrs in rows:
        fid = attrs.get("attr_UniqueFireIdentifier")
        if not fid:
            continue
        by_id[fid].append(attrs)
    fires = []
    for fid, recs in by_id.items():
        times = []
        cats = []
        methods = []
        acres = []
        names = set()
        irwins = set()
        lats, lngs, states, kinds = [], [], [], []
        for a in recs:
            t = parse_arcgis_datetime(a.get("poly_PolygonDateTime"))
            if t is not None:
                times.append(t)
            cats.append(a.get("poly_FeatureCategory"))
            if a.get("poly_MapMethod"):
                methods.append(a.get("poly_MapMethod"))
            if a.get("poly_GISAcres") is not None:
                acres.append(float(a["poly_GISAcres"]))
            if a.get("poly_IncidentName"):
                names.add(str(a["poly_IncidentName"]).strip())
            if a.get("poly_IRWINID"):
                irwins.add(str(a["poly_IRWINID"]))
            if a.get("attr_InitialLatitude") is not None:
                lats.append(float(a["attr_InitialLatitude"]))
            if a.get("attr_InitialLongitude") is not None:
                lngs.append(float(a["attr_InitialLongitude"]))
            if a.get("attr_POOState"):
                states.append(str(a["attr_POOState"]))
            if a.get("attr_IncidentTypeCategory"):
                kinds.append(str(a["attr_IncidentTypeCategory"]))
        uniq_t = sorted(set(times))
        n_times = len(uniq_t)
        span_h = (uniq_t[-1] - uniq_t[0]).total_seconds() / 3600.0 if n_times >= 2 else 0.0
        n_daily = sum(1 for c in cats if c == "Wildfire Daily Fire Perimeter")
        n_final = sum(1 for c in cats if c == "Wildfire Final Fire Perimeter")
        lat = sum(lats) / len(lats) if lats else None
        lng = sum(lngs) / len(lngs) if lngs else None
        state = states[0] if states else None
        kind = kinds[0] if kinds else None
        max_ac = max(acres) if acres else 0.0
        fires.append(
            {
                "fire_id": fid,
                "name": sorted(names)[0] if names else fid,
                "irwin_ids": sorted(irwins),
                "year": int(fid[:4]) if fid[:4].isdigit() else None,
                "lat": lat,
                "lng": lng,
                "state": state,
                "kind": kind,
                "n_rows": len(recs),
                "n_times": n_times,
                "n_daily_rows": n_daily,
                "n_final_rows": n_final,
                "span_hours": span_h,
                "max_acres": max_ac,
                "map_methods": sorted(set(methods)),
                "seed_time": uniq_t[0].isoformat() if uniq_t else None,
                "last_time": uniq_t[-1].isoformat() if uniq_t else None,
                "conus": _in_conus(lat, lng, state),
            }
        )
    fires.sort(key=lambda f: (-f["n_times"], -f["span_hours"], -f["max_acres"]))
    return fires


def gate(f: dict) -> bool:
    """Strict Job C tape: real Daily progression, not a single final."""
    if not f.get("conus"):
        return False
    if f.get("kind") not in {None, "WF", "Wildfire"}:
        # keep WF; drop RX if tagged
        if f.get("kind") in {"RX", "Prescribed Fire"}:
            return False
    if f["n_daily_rows"] < 3:
        return False
    if f["n_times"] < 4:
        return False
    if f["span_hours"] < 72.0:
        return False
    if f["max_acres"] < 500.0:
        return False
    # 282 million acres on Cougar Cr. #1 is a GIS explode, not a fire.
    if f["max_acres"] > 2_000_000.0:
        return False
    if f["lat"] is None or f["lng"] is None:
        return False
    return True


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with httpx.Client() as client:
        for year in YEARS:
            rows.extend(page_year(client, year))
    logger.info("raw attribute rows %d", len(rows))
    fires = aggregate(rows)
    usable = [f for f in fires if gate(f)]

    def _first_fail(f: dict) -> str:
        if not f.get("conus"):
            return "not_conus"
        if f.get("kind") in {"RX", "Prescribed Fire"}:
            return "rx"
        if f["n_daily_rows"] < 3:
            return "n_daily_rows"
        if f["n_times"] < 4:
            return "n_times"
        if f["span_hours"] < 72.0:
            return "span_hours"
        if f["max_acres"] < 500.0:
            return "acres_low"
        if f["max_acres"] > 2_000_000.0:
            return "acres_explode"
        if f["lat"] is None or f["lng"] is None:
            return "no_latlng"
        return "ok"

    funnel = dict(Counter(_first_fail(f) for f in fires))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "years": list(YEARS),
        "n_raw_rows": len(rows),
        "n_unique_fires": len(fires),
        "n_usable_job_c": len(usable),
        "usable_by_year": dict(Counter(f["year"] for f in usable)),
        "usable_n_times_hist": dict(sorted(Counter(f["n_times"] for f in usable).items())),
        "usable_span_p50": (
            sorted(f["span_hours"] for f in usable)[len(usable) // 2] if usable else None
        ),
        "gates": {
            "conus": True,
            "n_daily_rows_min": 3,
            "n_times_min": 4,
            "span_hours_min": 72,
            "max_acres_min": 500,
            "max_acres_max": 2_000_000,
            "join": "attr_UniqueFireIdentifier exact (not 25 km bbox)",
        },
        "rejection_funnel": funnel,
        "usable_fire_ids": [f["fire_id"] for f in usable],
        "fires": usable,
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s unique=%d usable=%d by_year=%s",
        OUT,
        len(fires),
        len(usable),
        summary["usable_by_year"],
    )
    print(json.dumps({k: summary[k] for k in summary if k != "fires"}, indent=2))


if __name__ == "__main__":
    main()
