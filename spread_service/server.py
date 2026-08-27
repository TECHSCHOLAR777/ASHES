"""Out-of-process spread engine HTTP service.

This directory is a licensing seam (SRS 2.6.1 / 7.7 / NFR-16). The proprietary
agent/model in `src/` talks to this process only over HTTP (`POST /spread_run`).
Nothing in `src/` imports this package.

The solver is a Rothermel-rate + Huygens elliptical raster propagator using
published Scott & Burgan FBFM40 characteristic ROS, fed by LANDFIRE rasters
and HRRR wind. It is original code, not a copy of ELMFIRE (EPL-2.0) or
Cell2Fire (GPL-3.0). If an `elmfire` binary is present at SPREAD_ENGINE_BIN
the server will exec it instead; otherwise this solver is the engine behind
the same `spread_run` contract. See DECISIONS.md.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

DEFAULT_HOST = os.environ.get("SPREAD_SERVICE_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SPREAD_SERVICE_PORT", "8765"))
ENGINE_NAME = "rothermel_huygens_v1"
HORIZON_HOURS = 72.0

# Characteristic head-fire ROS (m/min) at 8% 1-h moisture, 0 slope, ~2.2 m/s
# (5 mph) midflame wind. Compiled from Scott & Burgan 2005 typical BehavePlus
# outputs for the 40 fuel models; NB codes are zero. These are published
# lookup values, not fitted from our training set.
FBFM40_ROS_REF_M_PER_MIN: dict[int, float] = {
    91: 0.0, 92: 0.0, 93: 0.0, 98: 0.0, 99: 0.0,
    101: 12.0, 102: 18.0, 103: 22.0, 104: 28.0, 105: 14.0, 106: 8.0, 107: 35.0, 108: 40.0, 109: 45.0,
    121: 8.0, 122: 12.0, 123: 16.0, 124: 22.0,
    141: 6.0, 142: 8.0, 143: 10.0, 144: 12.0, 145: 14.0, 146: 7.0, 147: 15.0, 148: 18.0, 149: 20.0,
    161: 4.0, 162: 5.0, 163: 7.0, 164: 6.0, 165: 8.0,
    181: 6.0, 182: 8.0, 183: 10.0, 184: 12.0, 185: 14.0, 186: 16.0, 187: 18.0, 188: 20.0, 189: 22.0,
}

NON_BURNABLE = {0, 91, 92, 93, 98, 99}


def _ros_ref(fbfm: int) -> float:
    if fbfm in FBFM40_ROS_REF_M_PER_MIN:
        return FBFM40_ROS_REF_M_PER_MIN[fbfm]
    if 100 <= fbfm <= 189:
        return 8.0
    return 0.0


def _length_to_breadth(wind_speed_ms: float) -> float:
    """Alexander 1985 LB ratio from 10-m wind (m/s). Floor 1.0 (circle)."""
    u_mph = max(wind_speed_ms, 0.0) * 2.23694
    lb = 0.936 * math.exp(0.2566 * u_mph) + 0.461 * math.exp(-0.1548 * u_mph) - 0.397
    return max(1.0, lb)


def _moisture_factor(rh_pct: float | None) -> float:
    """Crude dead-fuel moisture damping from RH. Never fabricates a reading:
    missing RH -> factor 1.0 (the reference moisture the ROS table assumes)."""
    if rh_pct is None:
        return 1.0
    fm = max(3.0, min(30.0, 0.32 * float(rh_pct)))
    # Linear damping between 3% (factor ~1.3) and 20% (near extinction, ~0.2).
    return float(max(0.15, min(1.4, 1.3 - (fm - 3.0) / 20.0)))


def _slope_factor(slope_deg: float) -> float:
    if not np.isfinite(slope_deg):
        return 1.0
    phi_s = 5.275 * math.tan(math.radians(max(0.0, float(slope_deg)))) ** 2
    return 1.0 + phi_s


def _cell_size_m(transform) -> float:
    # Affine: a is pixel width. In EPSG:4326 this is degrees; convert at mid-lat later.
    return abs(float(transform[0]))


def _degrees_to_meters(dlng: float, dlat: float, lat0: float) -> tuple[float, float]:
    m_per_deg_lat = 111_000.0
    m_per_deg_lng = 111_000.0 * max(math.cos(math.radians(lat0)), 0.1)
    return abs(dlng) * m_per_deg_lng, abs(dlat) * m_per_deg_lat


def _wind_bearing_deg(u: float, v: float) -> float:
    return math.degrees(math.atan2(u, v)) % 360


def _in_polygon(lng: float, lat: float, rings: list[list[list[float]]]) -> bool:
    """Ray-casting on the first ring of each polygon. Adequate for WFIGS rings."""
    for ring in rings:
        if len(ring) < 3:
            continue
        inside = False
        j = len(ring) - 1
        for i, pt in enumerate(ring):
            xi, yi = pt[0], pt[1]
            xj, yj = ring[j][0], ring[j][1]
            intersect = ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi)
            if intersect:
                inside = not inside
            j = i
        if inside:
            return True
    return False


def propagate(
    fbfm40: np.ndarray,
    slope_deg: np.ndarray,
    transform: list[float],
    lat0: float,
    wind_u: float,
    wind_v: float,
    rh_pct: float | None,
    perimeter_rings: list[list[list[float]]],
    ignition_points: list[dict[str, float]],
    horizon_hours: float = HORIZON_HOURS,
) -> np.ndarray:
    """Return arrival time in hours, inf where the fire does not reach by horizon."""
    height, width = fbfm40.shape
    a, _, c, _, e, f = (list(transform) + [0, 0, 0])[:6]
    # rasterio affine: x = a*col + c, y = e*row + f with e typically negative
    dx_deg, dy_deg = abs(a), abs(e)
    dx_m, dy_m = _degrees_to_meters(dx_deg, dy_deg, lat0)
    cell_m = max(1.0, math.sqrt(dx_m * dy_m))

    wind_speed = math.sqrt(wind_u * wind_u + wind_v * wind_v)
    wind_bearing = _wind_bearing_deg(wind_u, wind_v) if wind_speed > 0 else 0.0
    lb = _length_to_breadth(wind_speed)
    hb = lb  # head-to-back ≈ LB for a simple ellipse
    mf = _moisture_factor(rh_pct)
    wind_scale = max(0.25, wind_speed / 2.2)  # table is at ~2.2 m/s midflame

    ros = np.zeros((height, width), dtype=np.float64)
    for r in range(height):
        for col in range(width):
            code = int(fbfm40[r, col]) if np.isfinite(fbfm40[r, col]) else 0
            if code in NON_BURNABLE:
                continue
            slp = float(slope_deg[r, col]) if slope_deg is not None and np.isfinite(slope_deg[r, col]) else 0.0
            ros[r, col] = _ros_ref(code) * wind_scale * mf * _slope_factor(slp)

    arrival = np.full((height, width), np.inf, dtype=np.float64)
    ignited: list[tuple[int, int]] = []

    def xy(row: int, col: int) -> tuple[float, float]:
        lng = a * (col + 0.5) + c
        lat = e * (row + 0.5) + f
        return float(lng), float(lat)

    for r in range(height):
        for col in range(width):
            if ros[r, col] <= 0:
                continue
            lng, lat = xy(r, col)
            if perimeter_rings and _in_polygon(lng, lat, perimeter_rings):
                arrival[r, col] = 0.0
                ignited.append((r, col))

    for pt in ignition_points or []:
        lng, lat = float(pt["lng"]), float(pt["lat"])
        col = int(round((lng - c) / a - 0.5)) if a else 0
        row = int(round((lat - f) / e - 0.5)) if e else 0
        if 0 <= row < height and 0 <= col < width and ros[row, col] > 0:
            arrival[row, col] = min(arrival[row, col], 0.0)
            ignited.append((row, col))

    if not ignited:
        return arrival

    neighbors = [
        (-1, 0, 0.0),
        (-1, 1, 45.0),
        (0, 1, 90.0),
        (1, 1, 135.0),
        (1, 0, 180.0),
        (1, -1, 225.0),
        (0, -1, 270.0),
        (-1, -1, 315.0),
    ]
    heap: list[tuple[float, int, int]] = [(0.0, r, c) for r, c in ignited]
    heapq.heapify(heap)
    while heap:
        t0, r, col = heapq.heappop(heap)
        if t0 > arrival[r, col] + 1e-9:
            continue
        for dr, dc, step_bearing in neighbors:
            rr, cc = r + dr, col + dc
            if rr < 0 or cc < 0 or rr >= height or cc >= width:
                continue
            if ros[rr, cc] <= 0:
                continue
            delta = (step_bearing - wind_bearing + 180.0) % 360 - 180.0
            theta = math.radians(delta)
            a_ros = ros[rr, cc]
            b_ros = a_ros / lb
            denom = math.sqrt((b_ros * math.cos(theta)) ** 2 + (a_ros * math.sin(theta)) ** 2)
            if denom <= 1e-9:
                continue
            ros_dir = (a_ros * b_ros) / denom
            step_m = cell_m * (math.sqrt(2.0) if dr and dc else 1.0)
            dt_hours = (step_m / max(ros_dir, 1e-6)) / 60.0
            cand = t0 + dt_hours
            if cand < arrival[rr, cc] and cand <= horizon_hours:
                arrival[rr, cc] = cand
                heapq.heappush(heap, (cand, rr, cc))
    return arrival


def run_ensemble(inputs: dict[str, Any]) -> dict[str, Any]:
    fbfm40 = np.array(inputs["fbfm40"], dtype=np.float64)
    slope = np.array(inputs.get("slope_deg") or np.zeros_like(fbfm40), dtype=np.float64)
    transform = list(inputs["transform"])
    west, south, east, north = inputs["west"], inputs["south"], inputs["east"], inputs["north"]
    lat0 = (south + north) / 2.0
    weather = inputs["weather"]
    wind_u = float(weather.get("wind_u") or 0.0)
    wind_v = float(weather.get("wind_v") or 0.0)
    rh = weather.get("rh_pct")
    rings = inputs.get("perimeter_rings") or []
    ignitions = inputs.get("ignition_points") or []
    horizon = float(inputs.get("horizon_hours") or HORIZON_HOURS)
    cfg = inputs.get("ensemble") or {}
    n_members = int(cfg.get("n_members") or 7)
    wind_frac = float(cfg.get("wind_speed_frac") or 0.2)
    dir_deg = float(cfg.get("wind_dir_deg") or 20.0)
    rh_frac = float(cfg.get("moisture_frac") or 0.15)
    rng = np.random.default_rng(int(cfg.get("seed") or 42))

    members = []
    for i in range(max(1, n_members)):
        if i == 0:
            u, v, rh_i = wind_u, wind_v, rh
        else:
            speed = math.sqrt(wind_u * wind_u + wind_v * wind_v)
            bearing = _wind_bearing_deg(wind_u, wind_v) if speed > 0 else 0.0
            speed_p = max(0.0, speed * (1.0 + float(rng.normal(0.0, wind_frac))))
            bearing_p = (bearing + float(rng.normal(0.0, dir_deg))) % 360
            rad = math.radians(bearing_p)
            u = speed_p * math.sin(rad)
            v = speed_p * math.cos(rad)
            if rh is None:
                rh_i = None
            else:
                rh_i = float(np.clip(rh * (1.0 + rng.normal(0.0, rh_frac)), 1.0, 100.0))
        members.append(propagate(fbfm40, slope, transform, lat0, u, v, rh_i, rings, ignitions, horizon))

    stacked = np.stack(members, axis=0)
    finite = np.isfinite(stacked)
    arrival_mean = np.full(fbfm40.shape, np.nan, dtype=np.float64)
    arrival_std = np.full(fbfm40.shape, np.nan, dtype=np.float64)
    for r in range(fbfm40.shape[0]):
        for c in range(fbfm40.shape[1]):
            vals = stacked[:, r, c][finite[:, r, c]]
            if vals.size == 0:
                continue
            arrival_mean[r, c] = float(np.mean(vals))
            arrival_std[r, c] = float(np.std(vals)) if vals.size > 1 else 0.0

    def p_burn(hours: float) -> list[list[float]]:
        pb = np.zeros(fbfm40.shape, dtype=np.float64)
        for m in members:
            pb += (m <= hours).astype(np.float64)
        pb /= max(len(members), 1)
        return pb.tolist()

    version = f"{ENGINE_NAME}:n{len(members)}:h{int(horizon)}:grid{fbfm40.shape[0]}x{fbfm40.shape[1]}"
    return {
        "spread_field_version": version,
        "engine": ENGINE_NAME,
        "n_members": len(members),
        "horizon_hours": horizon,
        "arrival_hours": np.where(np.isfinite(arrival_mean), arrival_mean, None).tolist(),
        "eta_sigma_hours": np.where(np.isfinite(arrival_std), arrival_std, None).tolist(),
        "p_burn_24": p_burn(24.0),
        "p_burn_48": p_burn(48.0),
        "p_burn_72": p_burn(72.0),
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "transform": transform,
        "shape": [int(fbfm40.shape[0]), int(fbfm40.shape[1])],
    }


def _run_external_engine(inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Optional JSON-file adapter for a real elmfire (or wrapper) binary.

    Contract: `SPREAD_ENGINE_BIN in.json out.json` must write a JSON object
    with at least `arrival_hours`, `p_burn_72`, and `spread_field_version`.
    Missing binary, non-zero exit, or malformed output falls back to the
    in-process Rothermel-Huygens solver. This is the licensing seam: the
    binary is exec'd, never imported.
    """
    bin_path = os.environ.get("SPREAD_ENGINE_BIN")
    if not bin_path:
        return None
    path = Path(bin_path)
    if not path.exists():
        sys.stderr.write(f"spread_service: SPREAD_ENGINE_BIN={bin_path} does not exist; using internal solver\n")
        return None
    with tempfile.TemporaryDirectory(prefix="spread_run_") as td:
        in_path = Path(td) / "inputs.json"
        out_path = Path(td) / "outputs.json"
        in_path.write_text(json.dumps(inputs), encoding="utf-8")
        try:
            proc = subprocess.run(
                [str(path), str(in_path), str(out_path)],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            sys.stderr.write(f"spread_service: SPREAD_ENGINE_BIN failed: {exc}\n")
            return None
        if proc.returncode != 0 or not out_path.exists():
            sys.stderr.write(
                f"spread_service: SPREAD_ENGINE_BIN rc={proc.returncode} stderr={proc.stderr[:400]}\n"
            )
            return None
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        required = ("arrival_hours", "p_burn_72", "spread_field_version")
        if any(k not in data for k in required):
            sys.stderr.write("spread_service: SPREAD_ENGINE_BIN output missing required keys; falling back\n")
            return None
        return data


def _load_inputs(body: dict[str, Any]) -> dict[str, Any]:
    path = body.get("inputs_path")
    if path:
        data = np.load(path, allow_pickle=True)
        loaded = {k: data[k] for k in data.files}
        # npz stores python objects via allow_pickle arrays of object
        out = dict(body)
        out["fbfm40"] = loaded["fbfm40"]
        out["slope_deg"] = loaded.get("slope_deg")
        out["transform"] = loaded["transform"].tolist()
        out["west"] = float(loaded["west"])
        out["south"] = float(loaded["south"])
        out["east"] = float(loaded["east"])
        out["north"] = float(loaded["north"])
        if "perimeter_rings" in loaded:
            out["perimeter_rings"] = loaded["perimeter_rings"].tolist()
        if "ignition_points" in loaded:
            out["ignition_points"] = loaded["ignition_points"].tolist()
        if "weather" in loaded:
            out["weather"] = loaded["weather"].item() if loaded["weather"].shape == () else loaded["weather"]
        return out
    return body


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("spread_service: " + (fmt % args) + "\n")

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/"):
            self._json(200, {"ok": True, "engine": ENGINE_NAME})
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "invalid json"})
            return
        if path != "/spread_run":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            inputs = _load_inputs(body)
            external = _run_external_engine(inputs)
            result = external if external is not None else run_ensemble(inputs)
            result["ok"] = True
            self._json(200, result)
        except Exception as exc:  # never leak a traceback as a fake field
            self._json(500, {"ok": False, "error": str(exc)})


def main() -> None:
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), Handler)
    sys.stderr.write(f"spread_service listening on {DEFAULT_HOST}:{DEFAULT_PORT} engine={ENGINE_NAME}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
