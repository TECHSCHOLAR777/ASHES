"""HRRR gridded weather client via Herbie (SRS 4.1, FR-12).

Extracts 10 m wind U/V, 2 m temp, 2 m RH from the latest HRRR surface analysis (f00) at a
point, with the previous hour as a fallback when the current hour is not yet published to
NODD. `ndvi_current` in Mireye W is a vintage feature; this client is the only source of
live weather in the system (SRS 10.3: "NDVI/W are vintage - never call live").
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.logging_ import tool_logger

# cfgrib/ecCodes is not thread-safe: parsing GRIB2 files concurrently from multiple threads
# corrupts its C-level parser state ("fatal flex scanner internal error", confirmed against
# a live watch-loop run with 5 sites polling in parallel - see DECISIONS.md). All Herbie/
# cfgrib access is serialized process-wide through this lock.
_HRRR_PARSE_LOCK = threading.Lock()


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


def _nearest_grid_index(lat2d, lon2d, lat: float, lng_0_360: float) -> tuple[int, int]:
    """HRRR's native grid is Lambert Conformal: latitude/longitude are 2D curvilinear
    coordinates, not a 1D axis, so `.sel(..., method="nearest")` cannot build a pandas
    index on them (verified against a live fetch - see DECISIONS.md). Nearest neighbor is
    found by brute-force distance over the ~1.9M-cell CONUS grid instead, which takes
    milliseconds with numpy.
    """
    import numpy as np

    dist2 = (lat2d - lat) ** 2 + (lon2d - lng_0_360) ** 2
    flat_idx = int(np.argmin(dist2))
    return np.unravel_index(flat_idx, dist2.shape)


def _wind_dir_cardinal(u: float, v: float) -> str:
    # Meteorological "from" direction: wind vector (u,v) points where air is going.
    deg = (math.degrees(math.atan2(-u, -v))) % 360
    idx = int((deg + 11.25) // 22.5) % 16
    return _CARDINALS[idx]


_DATASET_CACHE_MAX_ENTRIES = 48  # 2 days of hourly runs; bounds memory in a long-lived process


class HRRRClient:
    """Wraps `Herbie` to fetch the latest available HRRR f00 analysis at a point.

    A watch-loop cycle asks this same client for many sites in parallel, and they almost
    always want the same handful of candidate hours. Fetching and parsing the CONUS grid
    is the expensive part (network + cfgrib); extracting one point from an already-parsed
    dataset is cheap. So the parsed dataset is cached per run-time and shared across sites -
    confirmed necessary against a live 5-site poll cycle that took over an hour before this
    cache existed, because every site independently re-downloaded and re-parsed the same
    grid (see DECISIONS.md).
    """

    def __init__(self, max_hours_back: int = 6):
        self._max_hours_back = max_hours_back
        self._dataset_cache: dict[str, tuple] = {}
        self._cache_lock = threading.Lock()

    def _get_datasets(self, run_time: datetime, site_id: str | None) -> tuple:
        key = run_time.isoformat()
        with self._cache_lock:
            cached = self._dataset_cache.get(key)
        if cached is not None:
            return cached

        # Only one thread parses at a time (cfgrib/ecCodes is not thread-safe); a second
        # thread that loses the race re-checks the cache before doing any real work.
        with _HRRR_PARSE_LOCK:
            with self._cache_lock:
                cached = self._dataset_cache.get(key)
            if cached is not None:
                return cached

            from herbie import Herbie  # imported lazily: heavy dependency, only needed here

            start = time.monotonic()
            h = Herbie(run_time.strftime("%Y-%m-%d %H:%M"), model="hrrr", product="sfc", fxx=0)
            ds_wind = h.xarray(":UGRD:10 m|:VGRD:10 m").load()
            ds_temp = h.xarray(":TMP:2 m").load()
            ds_rh = h.xarray(":RH:2 m").load()
            # `.load()` fully materializes the arrays into memory here, inside the parse
            # lock, so every later `.isel()` from other threads only touches plain numpy
            # data - no lazy re-entry into cfgrib's non-thread-safe reader.
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call(
                "hrrr:fetch_grid", {"run_time": key}, {"ok": True}, site_id, latency_ms
            )

            with self._cache_lock:
                self._dataset_cache[key] = (ds_wind, ds_temp, ds_rh)
                if len(self._dataset_cache) > _DATASET_CACHE_MAX_ENTRIES:
                    del self._dataset_cache[min(self._dataset_cache)]
            return ds_wind, ds_temp, ds_rh

    def get_weather(
        self, lat: float, lng: float, site_id: str | None = None, at: datetime | None = None
    ) -> HRRRWeather:
        """Wind/temp/RH nearest to `at` (default: now). Training-data reconstruction (SRS
        6.2: "HRRR archive/reanalysis wind at t0") passes a historical `at`; HRRR's public
        NODD archive on AWS goes back to 2014-07-30, and Herbie fetches any date in range
        with the same code path used for live polling - no separate historical client
        needed. Falls back to earlier hours the same way live polling does, since an
        archived hour can also be missing/corrupt.
        """
        reference = at or datetime.now(timezone.utc)
        last_error: Exception | None = None

        for hours_back in range(self._max_hours_back):
            run_time = (reference - timedelta(hours=hours_back)).replace(minute=0, second=0, microsecond=0)
            start = time.monotonic()
            try:
                ds_wind, ds_temp, ds_rh = self._get_datasets(run_time, site_id)

                lng_0_360 = lng % 360
                y_idx, x_idx = _nearest_grid_index(
                    ds_wind["latitude"].values, ds_wind["longitude"].values, lat, lng_0_360
                )

                u = float(ds_wind["u10"].isel(y=y_idx, x=x_idx).values)
                v = float(ds_wind["v10"].isel(y=y_idx, x=x_idx).values)
                temp_k = float(ds_temp["t2m"].isel(y=y_idx, x=x_idx).values)
                rh = float(ds_rh["r2"].isel(y=y_idx, x=x_idx).values)

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
