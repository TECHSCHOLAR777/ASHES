"""HTTP client for the out-of-process spread_run service (SRS FR-20 / NFR-16).

This module is the only thing `src/` uses to talk to the engine. It never imports
`spread_service` as a library. If the service is not already listening it is
started as a sibling process.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from src.clients.landfire import LandfireStack
from src.geometry.aoi import Geometry
from src.logging_ import tool_logger

DEFAULT_HOST = os.environ.get("SPREAD_SERVICE_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SPREAD_SERVICE_PORT", "8765"))
SERVICE_SCRIPT = Path(__file__).resolve().parent.parent.parent / "spread_service" / "server.py"

_START_LOCK = threading.Lock()
_STARTED: subprocess.Popen | None = None


@dataclass
class SpreadSiteSample:
    eta_hours: float | None
    eta_sigma_hours: float | None
    p_burn_24: float | None
    p_burn_48: float | None
    p_burn_72: float | None
    spread_field_version: str
    inside_aoi: bool
    raw_eta_hours: float | None = None


@dataclass
class SpreadField:
    incident_id: str
    spread_field_version: str
    engine: str
    n_members: int
    arrival_hours: np.ndarray
    eta_sigma_hours: np.ndarray
    p_burn_24: np.ndarray
    p_burn_48: np.ndarray
    p_burn_72: np.ndarray
    transform: list[float]
    west: float
    south: float
    east: float
    north: float
    stale: bool = False

    def sample(self, lat: float, lng: float) -> SpreadSiteSample:
        a, b, c, d, e, f = (list(self.transform) + [0, 0, 0, 0, 0, 0])[:6]
        if a == 0 or e == 0:
            return SpreadSiteSample(None, None, None, None, None, self.spread_field_version, False)
        col = int(round((lng - c) / a - 0.5))
        row = int(round((lat - f) / e - 0.5))
        h, w = self.arrival_hours.shape
        if row < 0 or col < 0 or row >= h or col >= w:
            return SpreadSiteSample(None, None, None, None, None, self.spread_field_version, False)

        def _f(arr: np.ndarray) -> float | None:
            val = arr[row, col]
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        return SpreadSiteSample(
            eta_hours=_f(self.arrival_hours),
            eta_sigma_hours=_f(self.eta_sigma_hours),
            p_burn_24=_f(self.p_burn_24),
            p_burn_48=_f(self.p_burn_48),
            p_burn_72=_f(self.p_burn_72),
            spread_field_version=self.spread_field_version,
            inside_aoi=True,
            raw_eta_hours=_f(self.arrival_hours),
        )


class IncidentRegistry:
    """One spread field per active incident, reused across nearby sites (FR-9, NFR-4)."""

    def __init__(self) -> None:
        self._fields: dict[str, SpreadField] = {}
        self._lock = threading.Lock()

    def get(self, incident_id: str) -> SpreadField | None:
        with self._lock:
            return self._fields.get(incident_id)

    def put(self, field: SpreadField) -> None:
        with self._lock:
            self._fields[field.incident_id] = field


_REGISTRY = IncidentRegistry()


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def ensure_service(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Start spread_service/server.py as a sibling process if nothing is listening."""
    global _STARTED
    if _port_open(host, port):
        return
    with _START_LOCK:
        if _port_open(host, port):
            return
        env = os.environ.copy()
        env["SPREAD_SERVICE_HOST"] = host
        env["SPREAD_SERVICE_PORT"] = str(port)
        _STARTED = subprocess.Popen(
            [sys.executable, str(SERVICE_SCRIPT)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if _port_open(host, port):
                return
            time.sleep(0.1)
        raise RuntimeError(f"spread_run service did not start on {host}:{port}")


def _affine_list(transform) -> list[float]:
    return [float(x) for x in list(transform)[:6]]


def spread_run(
    incident_id: str,
    landfire: LandfireStack,
    aoi: Geometry,
    weather: dict[str, Any],
    perimeter_rings: list[list[list[float]]],
    ignition_points: list[dict[str, float]] | None = None,
    ensemble: dict[str, Any] | None = None,
    site_id: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    reuse: bool = True,
) -> SpreadField:
    """Call the out-of-process engine. Cached per incident_id (NFR-4)."""
    if reuse:
        existing = _REGISTRY.get(incident_id)
        if existing is not None:
            return existing

    ensure_service(host, port)
    payload = {
        "incident_id": incident_id,
        "fbfm40": landfire.fbfm40.tolist(),
        "slope_deg": np.nan_to_num(landfire.slope_deg, nan=0.0).tolist(),
        "transform": _affine_list(landfire.transform),
        "west": landfire.west,
        "south": landfire.south,
        "east": landfire.east,
        "north": landfire.north,
        "weather": weather,
        "perimeter_rings": perimeter_rings,
        "ignition_points": ignition_points or [],
        "ensemble": ensemble or {"n_members": 7, "wind_speed_frac": 0.2, "wind_dir_deg": 20.0, "moisture_frac": 0.15},
        "horizon_hours": 72.0,
        "aoi_cell_count": len(aoi.cells),
    }
    url = f"http://{host}:{port}/spread_run"
    start = time.monotonic()
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call("spread_run", {"incident_id": incident_id}, None, site_id, latency_ms, error=str(exc))
        raise
    latency_ms = (time.monotonic() - start) * 1000
    tool_logger.log_tool_call(
        "spread_run",
        {"incident_id": incident_id, "n_members": payload["ensemble"]["n_members"], "grid": list(landfire.fbfm40.shape)},
        {"spread_field_version": data.get("spread_field_version"), "engine": data.get("engine")},
        site_id,
        latency_ms,
    )
    if not data.get("ok"):
        raise RuntimeError(f"spread_run failed: {data.get('error')}")

    def _arr(key: str) -> np.ndarray:
        raw = data.get(key) or []
        return np.array([[np.nan if v is None else v for v in row] for row in raw], dtype=np.float64)

    field = SpreadField(
        incident_id=incident_id,
        spread_field_version=str(data.get("spread_field_version") or "unknown"),
        engine=str(data.get("engine") or "unknown"),
        n_members=int(data.get("n_members") or 0),
        arrival_hours=_arr("arrival_hours"),
        eta_sigma_hours=_arr("eta_sigma_hours"),
        p_burn_24=_arr("p_burn_24"),
        p_burn_48=_arr("p_burn_48"),
        p_burn_72=_arr("p_burn_72"),
        transform=list(data.get("transform") or _affine_list(landfire.transform)),
        west=float(data.get("west") or landfire.west),
        south=float(data.get("south") or landfire.south),
        east=float(data.get("east") or landfire.east),
        north=float(data.get("north") or landfire.north),
    )
    _REGISTRY.put(field)
    return field
