"""MTBS final burned-area perimeter client (SRS 6.1, 4.1): the gold training label source.

Public-domain, ~1yr lag, large fires only (>1000 ac West / >500 ac East per SRS 6.1).
Served from USGS/MTBS's own GeoServer WFS - verified live 2026-08-27 (see DECISIONS.md):
`https://edcintl.cr.usgs.gov/geoserver/mtbs/ows`, layer `mtbs:mtbs_fire_polygons`, geometry
column `geom`, 30,730 fires nationally as of this build. This is training-data tooling, not
part of the live watch-loop hot path - MTBS is never live E (SRS 6.1: "CAL FIRE FRAP
historical SHALL NOT be ingested as live E", and MTBS carries the same ~1yr lag).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime

import httpx

from src.logging_ import tool_logger

MTBS_WFS_URL = "https://edcintl.cr.usgs.gov/geoserver/mtbs/ows"
MTBS_TYPE_NAME = "mtbs:mtbs_fire_polygons"


@dataclass
class MTBSFire:
    event_id: str
    incident_name: str
    ignition_date: date | None
    acres: float | None
    centroid_lat: float | None
    centroid_lng: float | None
    geometry_rings: list[list[list[tuple[float, float]]]]  # MultiPolygon: list of polygons of rings


def _parse_multipolygon(geometry: dict) -> list[list[list[tuple[float, float]]]]:
    """Normalizes GeoJSON Polygon or MultiPolygon coordinates to a MultiPolygon shape."""
    coords = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        return [[[(pt[0], pt[1]) for pt in ring] for ring in coords]]
    return [[[(pt[0], pt[1]) for pt in ring] for ring in polygon] for polygon in coords]


def _point_in_ring(lat: float, lng: float, ring: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test on one ring (lng, lat pairs)."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def point_in_multipolygon(
    lat: float, lng: float, geometry_rings: list[list[list[tuple[float, float]]]]
) -> bool:
    """True if (lat, lng) falls inside the fire's final perimeter (the V1 gold label,
    SRS 6.1: "site inside the final perimeter within 72h of t0"). GeoJSON convention: a
    polygon's first ring is its outer boundary, any further rings are holes."""
    for polygon in geometry_rings:
        if not polygon:
            continue
        outer, holes = polygon[0], polygon[1:]
        if _point_in_ring(lat, lng, outer) and not any(_point_in_ring(lat, lng, hole) for hole in holes):
            return True
    return False


class MTBSClient:
    def __init__(self, timeout: float = 60.0):
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def get_fires_in_bbox(
        self,
        min_lng: float,
        min_lat: float,
        max_lng: float,
        max_lat: float,
        min_acres: float | None = None,
        year_start: int | None = None,
        year_end: int | None = None,
        max_features: int = 1000,
    ) -> list[MTBSFire]:
        """Queries MTBS final perimeters intersecting a bbox, optionally filtered by size/year."""
        cql_parts = [f"BBOX(geom,{min_lng},{min_lat},{max_lng},{max_lat},'EPSG:4326')"]
        if min_acres is not None:
            cql_parts.append(f"burnbndac >= {min_acres}")
        if year_start is not None:
            cql_parts.append(f"ig_date >= '{year_start}-01-01'")
        if year_end is not None:
            cql_parts.append(f"ig_date <= '{year_end}-12-31'")

        params = {
            "service": "WFS",
            "version": "1.0.0",
            "request": "GetFeature",
            "typeName": MTBS_TYPE_NAME,
            "outputFormat": "application/json",
            "maxFeatures": str(max_features),
            "CQL_FILTER": " AND ".join(cql_parts),
        }

        start = time.monotonic()
        try:
            resp = self._client.get(MTBS_WFS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call("mtbs:get_fires_in_bbox", params, None, None, latency_ms, error=str(exc))
            return []

        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call(
            "mtbs:get_fires_in_bbox", params, {"count": len(data.get("features", []))}, None, latency_ms
        )

        fires: list[MTBSFire] = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            ig_date_raw = props.get("ig_date")
            ig_date = None
            if ig_date_raw:
                try:
                    ig_date = datetime.fromisoformat(ig_date_raw.rstrip("Z")).date()
                except ValueError:
                    ig_date = None

            lat_str, lng_str = props.get("burnbndlat"), props.get("burnbndlon")
            fires.append(
                MTBSFire(
                    event_id=props.get("event_id", ""),
                    incident_name=props.get("incid_name", "unknown"),
                    ignition_date=ig_date,
                    acres=float(props["burnbndac"]) if props.get("burnbndac") is not None else None,
                    centroid_lat=float(lat_str) if lat_str else None,
                    centroid_lng=float(lng_str) if lng_str else None,
                    geometry_rings=_parse_multipolygon(feature.get("geometry", {})),
                )
            )
        return fires
