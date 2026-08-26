"""HRRR gridded weather client via Herbie (SRS 4.1, FR-12).

Extracts 10 m wind U/V, 2 m temp, 2 m RH from the latest HRRR surface analysis (f00) at a
point, with the previous hour as a fallback when the current hour is not yet published to
NODD. `ndvi_current` in Mireye W is a vintage feature; this client is the only source of
live weather in the system (SRS 10.3: "NDVI/W are vintage - never call live").
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.logging_ import tool_logger


@dataclass
class HRRRWeather:
    wind_u_10m: float
    wind_v_10m: float
    wind_speed_ms: float
    wind_dir_cardinal: str
    temp_c: float
    rh_pct: float
    hrrr_valid_time: str
    stale: bool = False


_CARDINALS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _wind_dir_cardinal(u: float, v: float) -> str:
    # Meteorological "from" direction: wind vector (u,v) points where air is going.
    deg = (math.degrees(math.atan2(-u, -v))) % 360
    idx = int((deg + 11.25) // 22.5) % 16
    return _CARDINALS[idx]


class HRRRClient:
    """Wraps `Herbie` to fetch the latest available HRRR f00 analysis at a point."""

    def __init__(self, max_hours_back: int = 6):
        self._max_hours_back = max_hours_back

    def get_weather(self, lat: float, lng: float, site_id: str | None = None) -> HRRRWeather:
        from herbie import Herbie  # imported lazily: heavy dependency, only needed here

        now = datetime.now(timezone.utc)
        last_error: Exception | None = None

        for hours_back in range(self._max_hours_back):
            run_time = (now - timedelta(hours=hours_back)).replace(minute=0, second=0, microsecond=0)
            start = time.monotonic()
            try:
                h = Herbie(run_time.strftime("%Y-%m-%d %H:%M"), model="hrrr", product="sfc", fxx=0)
                ds_wind = h.xarray(":UGRD:10 m|:VGRD:10 m")
                ds_temp = h.xarray(":TMP:2 m")
                ds_rh = h.xarray(":RH:2 m")

                u = float(ds_wind["u10"].sel(latitude=lat, longitude=lng % 360, method="nearest").values)
                v = float(ds_wind["v10"].sel(latitude=lat, longitude=lng % 360, method="nearest").values)
                temp_k = float(ds_temp["t2m"].sel(latitude=lat, longitude=lng % 360, method="nearest").values)
                rh = float(ds_rh["r2"].sel(latitude=lat, longitude=lng % 360, method="nearest").values)

                latency_ms = (time.monotonic() - start) * 1000
                tool_logger.log_tool_call(
                    "hrrr:get_weather",
                    {"lat": lat, "lng": lng, "run_time": run_time.isoformat()},
                    {"u": u, "v": v, "temp_k": temp_k, "rh": rh},
                    site_id,
                    latency_ms,
                )

                speed = math.sqrt(u**2 + v**2)
                return HRRRWeather(
                    wind_u_10m=u,
                    wind_v_10m=v,
                    wind_speed_ms=speed,
                    wind_dir_cardinal=_wind_dir_cardinal(u, v),
                    temp_c=temp_k - 273.15,
                    rh_pct=rh,
                    hrrr_valid_time=run_time.isoformat(),
                    stale=hours_back > 0,
                )
            except Exception as exc:  # Herbie/xarray raise a wide variety of lookup errors
                latency_ms = (time.monotonic() - start) * 1000
                tool_logger.log_tool_call(
                    "hrrr:get_weather",
                    {"lat": lat, "lng": lng, "run_time": run_time.isoformat()},
                    None,
                    site_id,
                    latency_ms,
                    error=str(exc),
                )
                last_error = exc
                continue

        raise RuntimeError(
            f"HRRR unavailable for the last {self._max_hours_back} hours"
        ) from last_error
