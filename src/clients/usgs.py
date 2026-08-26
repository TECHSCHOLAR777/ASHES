"""USGS NWIS instantaneous gage discharge client (SRS 4.1, FR-45).

The one fresh live query the Response Agent makes: confirms whether a nearby stream has
water right now. Classification uses a static per-gage percentile lookup when available,
falling back to fixed cfs cutoffs; `unknown` when no gage_id is given (never fabricated).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from src.logging_ import tool_logger

USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"

# Static percentile lookup for gages we have historical context on; extend as needed.
# {gage_id: (p10_cfs, p25_cfs, p75_cfs)}
GAGE_PERCENTILES: dict[str, tuple[float, float, float]] = {}


@dataclass
class GageDischarge:
    gage_id: str | None
    discharge_cfs: float | None
    discharge_class: str  # critically_low | low | normal | high | unknown
    fetched_at: str


def _classify(gage_id: str | None, discharge_cfs: float | None) -> str:
    if gage_id is None or discharge_cfs is None:
        return "unknown"
    if gage_id in GAGE_PERCENTILES:
        p10, p25, p75 = GAGE_PERCENTILES[gage_id]
        if discharge_cfs < p10:
            return "critically_low"
        if discharge_cfs < p25:
            return "low"
        if discharge_cfs > p75:
            return "high"
        return "normal"
    if discharge_cfs < 10:
        return "critically_low"
    if discharge_cfs < 25:
        return "low"
    if discharge_cfs > 500:
        return "high"
    return "normal"


class USGSClient:
    def __init__(self, timeout: float = 15.0):
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def get_gage_discharge(self, gage_id: str | None, site_id: str | None = None) -> GageDischarge:
        fetched_at = datetime.now(timezone.utc).isoformat()
        if not gage_id:
            return GageDischarge(gage_id=None, discharge_cfs=None, discharge_class="unknown", fetched_at=fetched_at)

        params = {"sites": gage_id, "parameterCd": "00060", "format": "json"}
        start = time.monotonic()
        try:
            resp = self._client.get(USGS_IV_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call("usgs:iv", params, None, site_id, latency_ms, error=str(exc))
            return GageDischarge(gage_id=gage_id, discharge_cfs=None, discharge_class="unknown", fetched_at=fetched_at)

        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call("usgs:iv", params, {"ok": True}, site_id, latency_ms)

        discharge_cfs = None
        try:
            series = data["value"]["timeSeries"]
            if series:
                values = series[0]["values"][0]["value"]
                if values:
                    discharge_cfs = float(values[-1]["value"])
        except (KeyError, IndexError, ValueError, TypeError):
            discharge_cfs = None

        return GageDischarge(
            gage_id=gage_id,
            discharge_cfs=discharge_cfs,
            discharge_class=_classify(gage_id, discharge_cfs),
            fetched_at=fetched_at,
        )
