"""WFIGS incident locations + operational perimeters client (SRS 4.1, FR-6/FR-8).

Distance from site to a perimeter boundary is always computed geodesically in code via
`pyproj.Geod.inv` (SRS rule 3.2 / build rule 2) - never estimated or guessed by the LLM.
Perimeters here are WFIGS "Interagency Perimeters Current" (operational), never MTBS final,
so every perimeter returned is marked `perimeter_unofficial = True`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
from pyproj import Geod

from src.logging_ import tool_logger

WFIGS_INCIDENTS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Incident_Locations_Current/FeatureServer/0/query"
)
WFIGS_PERIMETERS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query"
)

_GEOD = Geod(ellps="WGS84")


@dataclass
class WFIGSIncident:
    irwin_id: str
    name: str
    acres: float | None
    containment_pct: float | None
    discovery_datetime: str | None
    lat: float
    lng: float


@dataclass
class WFIGSPerimeter:
    irwin_id: str
    name: str
    geometry_rings: list[list[tuple[float, float]]]
    perimeter_unofficial: bool = True


def geodesic_distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Point-to-point geodesic distance in meters, WGS84 (pyproj.Geod.inv)."""
    _, _, dist_m = _GEOD.inv(lng1, lat1, lng2, lat2)
    return dist_m


def distance_to_perimeter_m(site_lat: float, site_lng: float, perimeter: WFIGSPerimeter) -> float:
    """Minimum geodesic distance from the site to any sampled vertex of the perimeter polygon.

    Sampling the boundary vertices (rather than a full point-in-polygon + edge-distance
    routine) is sufficient precision for a 72 h ROS feature and keeps this dependency-free
    beyond pyproj; WFIGS perimeters are already coarse operational polygons.
    """
    best = float("inf")
    for ring in perimeter.geometry_rings:
        for lng, lat in ring:
            d = geodesic_distance_m(site_lat, site_lng, lat, lng)
            if d < best:
                best = d
    return best if best != float("inf") else 999_999.0


def _bbox_str(lat: float, lng: float, radius_km: float) -> str:
    import math

    dlat = radius_km / 111.0
    dlng = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    xmin, ymin, xmax, ymax = lng - dlng, lat - dlat, lng + dlng, lat + dlat
    return f"{xmin},{ymin},{xmax},{ymax}"


class WFIGSClient:
    def __init__(self, timeout: float = 20.0):
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def get_incidents(
        self, lat: float, lng: float, radius_km: float = 100, site_id: str | None = None
    ) -> list[WFIGSIncident]:
        params = {
            "geometry": _bbox_str(lat, lng, radius_km),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "f": "json",
        }
        start = time.monotonic()
        try:
            resp = self._client.get(WFIGS_INCIDENTS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call("wfigs:incidents", params, None, site_id, latency_ms, error=str(exc))
            return []

        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call(
            "wfigs:incidents", params, {"count": len(data.get("features", []))}, site_id, latency_ms
        )

        incidents: list[WFIGSIncident] = []
        for feature in data.get("features", []):
            attrs = feature.get("attributes", {})
            geom = feature.get("geometry", {})
            incidents.append(
                WFIGSIncident(
                    irwin_id=str(attrs.get("IrwinID") or attrs.get("irwin_id") or ""),
                    name=str(attrs.get("IncidentName") or attrs.get("incident_name") or "unknown"),
                    acres=attrs.get("DailyAcres") or attrs.get("acres"),
                    containment_pct=(
                        (attrs.get("PercentContained") or attrs.get("containment_pct") or 0) / 100.0
                        if attrs.get("PercentContained") is not None or attrs.get("containment_pct") is not None
                        else None
                    ),
                    discovery_datetime=attrs.get("FireDiscoveryDateTime") or attrs.get("discovery_datetime"),
                    lat=geom.get("y", lat),
                    lng=geom.get("x", lng),
                )
            )
        return incidents

    def get_perimeters(
        self, lat: float, lng: float, radius_km: float = 100, site_id: str | None = None
    ) -> list[WFIGSPerimeter]:
        params = {
            "geometry": _bbox_str(lat, lng, radius_km),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "f": "json",
        }
        start = time.monotonic()
        try:
            resp = self._client.get(WFIGS_PERIMETERS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call("wfigs:perimeters", params, None, site_id, latency_ms, error=str(exc))
            return []

        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call(
            "wfigs:perimeters", params, {"count": len(data.get("features", []))}, site_id, latency_ms
        )

        perimeters: list[WFIGSPerimeter] = []
        for feature in data.get("features", []):
            attrs = feature.get("attributes", {})
            geom = feature.get("geometry", {})
            rings = [[(pt[0], pt[1]) for pt in ring] for ring in geom.get("rings", [])]
            perimeters.append(
                WFIGSPerimeter(
                    irwin_id=str(attrs.get("IrwinID") or attrs.get("irwin_id") or ""),
                    name=str(attrs.get("IncidentName") or attrs.get("incident_name") or "unknown"),
                    geometry_rings=rings,
                )
            )
        return perimeters
