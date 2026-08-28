"""Operational timed fire perimeters (GeoMAC 2015-2019, WFIGS Daily 2020+).

These are dated snapshots, including IR-mapped polygons when `map_method`
says so. They are not MTBS final scars and not NIFC "All Years" finals.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.logging_ import tool_logger

NIFC_FS = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
WFIGS_DAILY_URL = f"{NIFC_FS}/WFIGS_Daily_Perimeters_Public/FeatureServer/0/query"

GEOMAC_LAYER = {
    2015: "Historic_Geomac_Perimeters_2015",
    2016: "Historic_Geomac_Perimeters_2016",
    2017: "Historic_Geomac_Perimeters_2017",
    2018: "Historic_Geomac_Perimeters_2018",
    2019: "Historic_GeoMAC_Perimeters_2019",
}

SANE_START = datetime(2014, 1, 1, tzinfo=timezone.utc)
SANE_END = datetime(2028, 1, 1, tzinfo=timezone.utc)
PAGE_SIZE = 2000
SERIES_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "timed_series"
# Prefer IR when several polygons share a timestamp.
_MAP_METHOD_RANK = {
    "infrared image": 0,
    "infrared": 1,
    "mixed methods": 2,
    "gps": 3,
    "hand sketch": 4,
}


@dataclass
class TimedSnapshot:
    t: datetime
    fire_id: str
    name: str
    acres: float | None
    map_method: str | None
    geometry_rings: list[list[tuple[float, float]]]
    source: str


@dataclass
class TimedFireSeries:
    fire_id: str
    name: str
    source: str
    snapshots: list[TimedSnapshot] = field(default_factory=list)

    @property
    def n_times(self) -> int:
        return len({s.t for s in self.snapshots})


def _bbox_str(lat: float, lng: float, radius_km: float) -> str:
    dlat = radius_km / 111.0
    dlng = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    return f"{lng - dlng},{lat - dlat},{lng + dlng},{lat + dlat}"


def parse_arcgis_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        t = value
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        ms = float(value)
        # ArcGIS date fields are ms since epoch. Reject garbage years.
        if abs(ms) > 1e15:
            return None
        t = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        if t < SANE_START or t >= SANE_END:
            return None
        return t
    text = str(value).strip()
    if not text:
        return None
    try:
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    if t < SANE_START or t >= SANE_END:
        return None
    return t


def _rings_from_geometry(geom: dict[str, Any] | None) -> list[list[tuple[float, float]]]:
    if not geom:
        return []
    return [[(float(pt[0]), float(pt[1])) for pt in ring] for ring in geom.get("rings", [])]


def _method_rank(method: str | None) -> int:
    if not method:
        return 50
    return _MAP_METHOD_RANK.get(method.strip().lower(), 20)


class TimedPerimeterClient:
    def __init__(self, timeout: float = 60.0):
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def fetch_series(
        self,
        lat: float,
        lng: float,
        year: int,
        site_id: str | None = None,
        radius_km: float = 25.0,
    ) -> TimedFireSeries | None:
        if year <= 2019:
            snaps = self._fetch_geomac(lat, lng, year, site_id, radius_km)
            source = f"geomac_{year}"
        else:
            snaps = self._fetch_wfigs_daily(lat, lng, year, site_id, radius_km)
            source = "wfigs_daily"
        if not snaps:
            return None
        chosen = _select_fire_group(snaps, lat, lng)
        if not chosen:
            return None
        collapsed = _collapse_same_timestamp(chosen)
        fire_id = collapsed[0].fire_id
        name = collapsed[0].name
        return TimedFireSeries(fire_id=fire_id, name=name, source=source, snapshots=collapsed)

    def _paged_query(
        self, url: str, params: dict[str, Any], tool_name: str, site_id: str | None
    ) -> list[dict[str, Any]]:
        features: list[dict[str, Any]] = []
        offset = 0
        while True:
            page_params = {
                **params,
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "json",
            }
            start = time.monotonic()
            try:
                resp = self._client.get(url, params=page_params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                latency_ms = (time.monotonic() - start) * 1000
                tool_logger.log_tool_call(tool_name, page_params, None, site_id, latency_ms, error=str(exc))
                break
            if data.get("error"):
                latency_ms = (time.monotonic() - start) * 1000
                tool_logger.log_tool_call(
                    tool_name, page_params, {"error": data.get("error")}, site_id, latency_ms, error=str(data.get("error"))
                )
                break
            batch = data.get("features") or []
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call(
                tool_name, {**page_params, "offset": offset}, {"count": len(batch)}, site_id, latency_ms
            )
            features.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            if offset > 50_000:
                break
        return features

    def _fetch_geomac(
        self, lat: float, lng: float, year: int, site_id: str | None, radius_km: float
    ) -> list[TimedSnapshot]:
        layer = GEOMAC_LAYER.get(year)
        if layer is None:
            return []
        url = f"{NIFC_FS}/{layer}/FeatureServer/0/query"
        params = {
            "geometry": _bbox_str(lat, lng, radius_km),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "incidentname,perimeterdatetime,gisacres,uniquefireidentifier,irwinid,mapmethod",
            "where": "1=1",
        }
        rows = self._paged_query(url, params, f"geomac:{year}", site_id)
        snaps: list[TimedSnapshot] = []
        for feat in rows:
            attrs = feat.get("attributes") or {}
            t = parse_arcgis_datetime(attrs.get("perimeterdatetime")) or parse_arcgis_datetime(
                attrs.get("datecurrent")
            )
            rings = _rings_from_geometry(feat.get("geometry"))
            if t is None or not rings:
                continue
            fire_id = str(attrs.get("uniquefireidentifier") or attrs.get("irwinid") or "")
            if not fire_id:
                continue
            acres = attrs.get("gisacres")
            snaps.append(
                TimedSnapshot(
                    t=t,
                    fire_id=fire_id,
                    name=str(attrs.get("incidentname") or fire_id),
                    acres=float(acres) if acres is not None else None,
                    map_method=attrs.get("mapmethod"),
                    geometry_rings=rings,
                    source=f"geomac_{year}",
                )
            )
        return snaps

    def _fetch_wfigs_daily(
        self, lat: float, lng: float, year: int, site_id: str | None, radius_km: float
    ) -> list[TimedSnapshot]:
        params = {
            "geometry": _bbox_str(lat, lng, radius_km),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": (
                "poly_IncidentName,poly_PolygonDateTime,poly_GISAcres,poly_MapMethod,"
                "poly_IRWINID,attr_UniqueFireIdentifier,poly_FeatureCategory"
            ),
            "where": f"attr_UniqueFireIdentifier LIKE '{int(year)}-%'",
        }
        rows = self._paged_query(WFIGS_DAILY_URL, params, "wfigs:daily_perimeters", site_id)
        snaps: list[TimedSnapshot] = []
        for feat in rows:
            attrs = feat.get("attributes") or {}
            t = parse_arcgis_datetime(attrs.get("poly_PolygonDateTime"))
            rings = _rings_from_geometry(feat.get("geometry"))
            if t is None or not rings:
                continue
            fire_id = str(attrs.get("attr_UniqueFireIdentifier") or attrs.get("poly_IRWINID") or "")
            if not fire_id:
                continue
            acres = attrs.get("poly_GISAcres")
            snaps.append(
                TimedSnapshot(
                    t=t,
                    fire_id=fire_id,
                    name=str(attrs.get("poly_IncidentName") or fire_id),
                    acres=float(acres) if acres is not None else None,
                    map_method=attrs.get("poly_MapMethod"),
                    geometry_rings=rings,
                    source="wfigs_daily",
                )
            )
        return snaps


def _select_fire_group(snaps: list[TimedSnapshot], lat: float, lng: float) -> list[TimedSnapshot]:
    from src.clients.mtbs import point_in_multipolygon

    groups: dict[str, list[TimedSnapshot]] = {}
    for snap in snaps:
        groups.setdefault(snap.fire_id, []).append(snap)
    containing: list[tuple[int, str]] = []
    fallback: list[tuple[float, str]] = []
    for fire_id, rows in groups.items():
        largest = max(rows, key=lambda s: s.acres or 0.0)
        mp = [largest.geometry_rings]
        if point_in_multipolygon(lat, lng, mp):
            containing.append((len({s.t for s in rows}), fire_id))
        # centroid distance proxy: first vertex of largest ring
        ring = largest.geometry_rings[0] if largest.geometry_rings else []
        if ring:
            lng0, lat0 = ring[0]
            dist = (lat - lat0) ** 2 + (lng - lng0) ** 2
            fallback.append((dist, fire_id))
    if containing:
        containing.sort(reverse=True)
        return sorted(groups[containing[0][1]], key=lambda s: s.t)
    if fallback:
        fallback.sort()
        return sorted(groups[fallback[0][1]], key=lambda s: s.t)
    return []


def _collapse_same_timestamp(snaps: list[TimedSnapshot]) -> list[TimedSnapshot]:
    """One polygon per timestamp: IR first, then largest acres."""
    by_t: dict[datetime, list[TimedSnapshot]] = {}
    for snap in snaps:
        by_t.setdefault(snap.t, []).append(snap)
    out: list[TimedSnapshot] = []
    for t in sorted(by_t):
        candidates = by_t[t]
        candidates.sort(key=lambda s: (_method_rank(s.map_method), -(s.acres or 0.0)))
        out.append(candidates[0])
    return out


def save_series_cache(event_id: str, series: TimedFireSeries | None) -> None:
    SERIES_CACHE.mkdir(parents=True, exist_ok=True)
    path = SERIES_CACHE / f"{event_id}.json"
    if series is None:
        path.write_text(json.dumps({"empty": True}), encoding="utf-8")
        return
    payload = {
        "fire_id": series.fire_id,
        "name": series.name,
        "source": series.source,
        "snapshots": [
            {
                "t": snap.t.isoformat(),
                "fire_id": snap.fire_id,
                "name": snap.name,
                "acres": snap.acres,
                "map_method": snap.map_method,
                "geometry_rings": snap.geometry_rings,
                "source": snap.source,
            }
            for snap in series.snapshots
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_series_cache(event_id: str) -> TimedFireSeries | None:
    path = SERIES_CACHE / f"{event_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("empty"):
        return None
    snaps = [
        TimedSnapshot(
            t=datetime.fromisoformat(row["t"]),
            fire_id=row["fire_id"],
            name=row["name"],
            acres=row.get("acres"),
            map_method=row.get("map_method"),
            geometry_rings=[[(float(a), float(b)) for a, b in ring] for ring in row["geometry_rings"]],
            source=row.get("source") or payload.get("source") or "",
        )
        for row in payload.get("snapshots") or []
    ]
    return TimedFireSeries(
        fire_id=payload["fire_id"], name=payload["name"], source=payload["source"], snapshots=snaps
    )


def series_cache_exists(event_id: str) -> bool:
    return (SERIES_CACHE / f"{event_id}.json").exists()
