"""NWS CAP alerts + SPC fire-weather outlook client (SRS 4.1, FR-10/FR-11).

CAP severity is never suppressed or downgraded (SRS 2.6.3 / FR-42): the raw alert data
flows through untouched to the ActionCard's `weather.red_flag` flag and `reasons[]`.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from src.logging_ import tool_logger

NWS_BASE_URL = "https://api.weather.gov"
SPC_OUTLOOK_URL = "https://www.spc.noaa.gov/products/fire_wx/fwdy1.txt"
USER_AGENT = "fire-copilot/1.0 (contact@example.com)"

RED_FLAG_EVENT_NAMES = {"red flag warning", "fire weather watch", "extreme fire danger"}


@dataclass
class CAPAlert:
    event: str
    severity: str
    urgency: str
    description: str
    effective: str | None
    expires: str | None
    is_red_flag: bool


@dataclass
class SPCOutlook:
    valid_date: str
    elevated: bool
    raw_text: str


class NWSClient:
    def __init__(self, timeout: float = 15.0):
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})

    def close(self) -> None:
        self._client.close()

    def get_cap_alerts(self, lat: float, lng: float, site_id: str | None = None) -> list[CAPAlert]:
        start = time.monotonic()
        url = f"{NWS_BASE_URL}/alerts/active"
        params = {"point": f"{lat},{lng}"}
        try:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call("nws:cap_alerts", params, None, site_id, latency_ms, error=str(exc))
            return []

        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call("nws:cap_alerts", params, data, site_id, latency_ms)

        alerts: list[CAPAlert] = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            event = props.get("event", "")
            alerts.append(
                CAPAlert(
                    event=event,
                    severity=props.get("severity", "Unknown"),
                    urgency=props.get("urgency", "Unknown"),
                    description=props.get("description", ""),
                    effective=props.get("effective"),
                    expires=props.get("expires"),
                    is_red_flag=event.strip().lower() in RED_FLAG_EVENT_NAMES,
                )
            )
        return alerts

    def get_spc_outlook(self, site_id: str | None = None) -> SPCOutlook | None:
        """Fetches the SPC day-1 fire-weather outlook text product and flags 'elevated' risk."""
        start = time.monotonic()
        try:
            resp = self._client.get(SPC_OUTLOOK_URL)
            resp.raise_for_status()
            text = resp.text
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call("nws:spc_outlook", {}, None, site_id, latency_ms, error=str(exc))
            return None

        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call("nws:spc_outlook", {}, {"length": len(text)}, site_id, latency_ms)

        elevated = bool(re.search(r"\b(ELEVATED|CRITICAL|EXTREME)\b", text, re.IGNORECASE))
        valid_date = datetime.now(timezone.utc).date().isoformat()
        match = re.search(r"VALID\s+(\d{6})Z?-", text)
        if match:
            valid_date = match.group(1)
        return SPCOutlook(valid_date=valid_date, elevated=elevated, raw_text=text[:2000])
