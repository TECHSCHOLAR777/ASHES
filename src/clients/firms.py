"""NASA FIRMS active-fire thermal hotspot client (SRS 4.1, FR-7).

FIRMS MAP_KEY quota is UNVERIFIED (SRS 10.2); this client raises `FIRMSQuotaWarning` on
signals of quota exhaustion and `FIRMSKeyMissing` if no key is configured, and both are
caught by the caller to set a degraded flag rather than block the pipeline.
"""
from __future__ import annotations

import csv
import io
import os
import time
from dataclasses import dataclass

import httpx

from src.logging_ import tool_logger

FIRMS_AREA_URL_TEMPLATE = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{bbox}/{days}"
)
SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]
# Historical/reprocessed ("Standard Processing") sources, verified live 2026-08-27 against
# the 2020 August Complex fire (see DECISIONS.md). NOAA-21 has no SP archive product yet
# (`VIIRS_NOAA21_SP` returns "Invalid source" - its reprocessing pipeline lags the sensor's
# 2022 launch), so historical queries use only these two VIIRS archives plus MODIS.
ARCHIVE_SOURCES = ["VIIRS_NOAA20_SP", "VIIRS_SNPP_SP", "MODIS_SP"]
RADII_KM = [5, 10, 20]
KM_PER_DEGREE_LAT = 111.0


class FIRMSKeyMissing(RuntimeError):
    pass


class FIRMSQuotaWarning(RuntimeWarning):
    pass


@dataclass
class Hotspot:
    lat: float
    lng: float
    frp: float
    source: str
    acq_date: str
    acq_time: str
    confidence: str


@dataclass
class FIRMSResult:
    hotspots_5km: list[Hotspot]
    hotspots_10km: list[Hotspot]
    hotspots_20km: list[Hotspot]
    count_5km: int
    count_10km: int
    count_20km: int
    frp_sum_5km: float
    unavailable: bool


def _bbox_for_radius(lat: float, lng: float, radius_km: float) -> str:
    dlat = radius_km / KM_PER_DEGREE_LAT
    import math

    dlng = radius_km / (KM_PER_DEGREE_LAT * max(math.cos(math.radians(lat)), 0.1))
    return f"{lng - dlng},{lat - dlat},{lng + dlng},{lat + dlat}"


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    import math

    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, a**0.5))


class FIRMSClient:
    def __init__(self, map_key: str | None = None, timeout: float = 20.0):
        self._map_key = map_key or os.environ.get("FIRMS_MAP_KEY") or ""
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def get_hotspots(
        self, lat: float, lng: float, days: int = 1, site_id: str | None = None
    ) -> FIRMSResult:
        """Live/near-real-time hotspots (used by the watch loop)."""
        return self._fetch(lat, lng, days=days, on_date=None, site_id=site_id)

    def get_historical_hotspots(
        self, lat: float, lng: float, on_date, site_id: str | None = None
    ) -> FIRMSResult:
        """Archive hotspots for one historical UTC date (training-data use, SRS 6.2:
        "E_hist: FIRMS/WFIGS as of t0"). `on_date` is a `datetime.date`."""
        return self._fetch(lat, lng, days=1, on_date=on_date, site_id=site_id)

    def _fetch(
        self, lat: float, lng: float, days: int, on_date, site_id: str | None
    ) -> FIRMSResult:
        if not self._map_key:
            tool_logger.log_tool_call(
                "firms:get_hotspots", {"lat": lat, "lng": lng}, None, site_id, 0.0, error="FIRMSKeyMissing"
            )
            return FIRMSResult([], [], [], 0, 0, 0, 0.0, unavailable=True)

        max_radius = max(RADII_KM)
        bbox = _bbox_for_radius(lat, lng, max_radius)
        all_hotspots: list[Hotspot] = []
        unavailable = False
        sources = ARCHIVE_SOURCES if on_date is not None else SOURCES

        for source in sources:
            url = FIRMS_AREA_URL_TEMPLATE.format(
                map_key=self._map_key, source=source, bbox=bbox, days=days
            )
            if on_date is not None:
                url = f"{url}/{on_date.isoformat()}"
            start = time.monotonic()
            try:
                resp = self._client.get(url)
                resp.raise_for_status()
                text = resp.text
            except httpx.HTTPError as exc:
                latency_ms = (time.monotonic() - start) * 1000
                tool_logger.log_tool_call(
                    f"firms:{source}", {"bbox": bbox, "days": days}, None, site_id, latency_ms, error=str(exc)
                )
                unavailable = True
                continue

            latency_ms = (time.monotonic() - start) * 1000
            if "Invalid" in text[:200] or "quota" in text[:200].lower():
                tool_logger.log_tool_call(
                    f"firms:{source}",
                    {"bbox": bbox, "days": days},
                    {"body_head": text[:200]},
                    site_id,
                    latency_ms,
                    error="quota_or_invalid_key",
                )
                unavailable = True
                continue

            tool_logger.log_tool_call(
                f"firms:{source}", {"bbox": bbox, "days": days}, {"rows": text.count("\n")}, site_id, latency_ms
            )

            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                try:
                    all_hotspots.append(
                        Hotspot(
                            lat=float(row["latitude"]),
                            lng=float(row["longitude"]),
                            frp=float(row.get("frp", 0.0) or 0.0),
                            source=source,
                            acq_date=row.get("acq_date", ""),
                            acq_time=row.get("acq_time", ""),
                            confidence=str(row.get("confidence", "")),
                        )
                    )
                except (KeyError, ValueError):
                    continue

        by_radius: dict[int, list[Hotspot]] = {r: [] for r in RADII_KM}
        for h in all_hotspots:
            dist = _haversine_km(lat, lng, h.lat, h.lng)
            for r in RADII_KM:
                if dist <= r:
                    by_radius[r].append(h)

        frp_sum_5km = sum(h.frp for h in by_radius[5])
        return FIRMSResult(
            hotspots_5km=by_radius[5],
            hotspots_10km=by_radius[10],
            hotspots_20km=by_radius[20],
            count_5km=len(by_radius[5]),
            count_10km=len(by_radius[10]),
            count_20km=len(by_radius[20]),
            frp_sum_5km=frp_sum_5km,
            unavailable=unavailable and not all_hotspots,
        )
