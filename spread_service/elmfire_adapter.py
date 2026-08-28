"""JSON/npz → GeoTIFF + elmfire.data → ELMFIRE → spread_run outputs.json.

ELMFIRE (EPL-2.0) is exec'd, never imported. Fortran lives outside this repo
(``ELMFIRE_BIN`` / ``ELMFIRE_INSTALL_DIR``). Native I/O is namelist + GeoTIFF,
not ``bin in.json out.json``; this wrapper is the contract adapter.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ENGINE_VERSION = "elmfire_2025.0212"
MS_TO_MPH = 2.23694
HORIZON_HOURS = 72.0


class ElmfireAdapterError(RuntimeError):
    """Raised when the wrapper cannot produce a field. Never falls back to Huygens."""


def utm_epsg(lat: float, lng: float) -> int:
    zone = int((lng + 180.0) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def dead_fuel_moisture(rh_pct: float | None) -> tuple[float, float, float]:
    """1/10/100-hr dead FMC (%) from RH. Documented, not a live NFDRS station."""
    if rh_pct is None or not math.isfinite(float(rh_pct)):
        return 6.0, 7.0, 8.0
    rh = float(np.clip(rh_pct, 1.0, 100.0))
    m1 = float(np.clip(0.32 * rh, 3.0, 25.0))
    return m1, float(np.clip(m1 + 1.0, 4.0, 28.0)), float(np.clip(m1 + 2.0, 5.0, 30.0))


def wind_speed_dir_mph(wind_u: float, wind_v: float) -> tuple[float, float]:
    """10 m earth-relative u/v → 10 m speed (mph) and meteorological FROM direction."""
    speed_ms = math.hypot(float(wind_u), float(wind_v))
    speed_mph = speed_ms * MS_TO_MPH
    coming_from = (math.degrees(math.atan2(-float(wind_u), -float(wind_v)))) % 360.0
    return speed_mph, coming_from


def _as_array(value: Any, dtype=np.float64) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def load_inputs(in_path: Path) -> dict[str, Any]:
    body = json.loads(in_path.read_text(encoding="utf-8"))
    rasters = body.get("rasters_path")
    if rasters and Path(rasters).exists():
        loaded = np.load(rasters, allow_pickle=True)
        try:
            for key in loaded.files:
                body[key] = loaded[key]
        finally:
            loaded.close()
        if "crs" in body and hasattr(body["crs"], "item"):
            try:
                body["crs"] = str(body["crs"].item())
            except Exception:
                body["crs"] = str(body["crs"])
    return body


def _affine_grid(transform: list[float], height: int, width: int) -> dict[str, float]:
    a, _, c, _, e, f = (list(transform) + [0, 0, 0, 0, 0, 0])[:6]
    west = float(c)
    north = float(f)
    east = west + width * float(a)
    south = north + height * float(e)
    if south > north:
        south, north = north, south
    return {"west": west, "south": south, "east": east, "north": north, "a": float(a), "e": float(e)}


def rasterize_phi(
    height: int,
    width: int,
    transform,
    rings: list[list[list[float]]],
    crs,
) -> np.ndarray:
    """+1 unburned, -1 inside the seed perimeter (ELMFIRE level-set convention)."""
    phi = np.ones((height, width), dtype=np.float32)
    if not rings:
        return phi
    try:
        from rasterio.features import rasterize
        from rasterio.transform import Affine
        from shapely.geometry import mapping, Polygon
        from shapely.ops import transform as shp_transform
        from shapely.validation import make_valid
        from pyproj import Transformer
    except Exception as exc:
        raise ElmfireAdapterError(f"rasterize_phi requires rasterio/shapely/pyproj: {exc}") from exc

    if not isinstance(transform, Affine):
        transform = Affine(*list(transform)[:6])

    geoms = []
    for ring in rings:
        coords = [(float(pt[0]), float(pt[1])) for pt in ring if len(pt) >= 2]
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        try:
            geoms.append(make_valid(Polygon(coords)))
        except Exception:
            continue
    if not geoms:
        return phi
    src_crs = str(crs) if crs else "EPSG:4326"
    # rings are lon/lat; if the destination grid is projected, reproject vertices.
    dst_crs = str(crs)
    if "4326" not in src_crs and "WGS" not in src_crs.upper():
        # Grid is projected; rings still arrive in lon/lat from spread_run.
        to_dst = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)

        def _xy(x, y, z=None):
            return to_dst.transform(x, y)

        geoms = [shp_transform(_xy, g) for g in geoms]
    shapes = [(mapping(g), -1.0) for g in geoms if not g.is_empty]
    if not shapes:
        return phi
    burned = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=1.0,
        dtype=np.float32,
        all_touched=True,
    )
    return burned.astype(np.float32)


def write_namelist(
    path: Path,
    *,
    epsg: int,
    cellsize: float,
    xll: float,
    yll: float,
    tstop_s: float,
    lh: float = 30.0,
    lw: float = 60.0,
) -> None:
    text = f"""&INPUTS
FUELS_AND_TOPOGRAPHY_DIRECTORY = './inputs'
ASP_FILENAME                   = 'asp'
CBD_FILENAME                   = 'cbd'
CBH_FILENAME                   = 'cbh'
CC_FILENAME                    = 'cc'
CH_FILENAME                    = 'ch'
DEM_FILENAME                   = 'dem'
FBFM_FILENAME                  = 'fbfm40'
SLP_FILENAME                   = 'slp'
ADJ_FILENAME                   = 'adj'
PHI_FILENAME                   = 'phi'
DT_METEOROLOGY                 = 3600.0
WEATHER_DIRECTORY              = './inputs'
WS_FILENAME                    = 'ws'
WD_FILENAME                   = 'wd'
M1_FILENAME                   = 'm1'
M10_FILENAME                  = 'm10'
M100_FILENAME                 = 'm100'
LH_MOISTURE_CONTENT            = {lh:.3f}
LW_MOISTURE_CONTENT            = {lw:.3f}
WS_AT_10M                     = .TRUE.
/

&OUTPUTS
OUTPUTS_DIRECTORY    = './outputs'
DTDUMP               = {tstop_s:.1f}
DUMP_FLIN            = .FALSE.
DUMP_SPREAD_RATE     = .FALSE.
DUMP_TIME_OF_ARRIVAL = .TRUE.
CONVERT_TO_GEOTIFF   = .TRUE.
/

&COMPUTATIONAL_DOMAIN
A_SRS = 'EPSG: {epsg}'
COMPUTATIONAL_DOMAIN_CELLSIZE = {cellsize:.3f}
COMPUTATIONAL_DOMAIN_XLLCORNER = {xll:.3f}
COMPUTATIONAL_DOMAIN_YLLCORNER = {yll:.3f}
/

&TIME_CONTROL
SIMULATION_DT    = 5.0
TARGET_CFL       = 0.4
SIMULATION_TSTOP = {tstop_s:.1f}
/

&SIMULATOR
NUM_IGNITIONS = 0
WX_BILINEAR_INTERPOLATION=.TRUE.
/

&MISCELLANEOUS
PATH_TO_GDAL                   = '/usr/bin'
SCRATCH                        = './scratch'
/
"""
    path.write_text(text, encoding="utf-8")


def _write_tif(path: Path, array: np.ndarray, transform, crs, dtype, nodata) -> None:
    import rasterio
    from rasterio.crs import CRS

    arr = np.asarray(array)
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": dtype,
        "crs": CRS.from_user_input(crs) if not isinstance(crs, CRS) else crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype(dtype), 1)


def _reproject_to_utm(
    arrays: dict[str, np.ndarray],
    src_transform,
    src_crs: str,
    dst_epsg: int,
    max_dim: int = 800,
):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    height, width = next(iter(arrays.values())).shape
    src_crs_obj = CRS.from_user_input(src_crs)
    dst_crs = CRS.from_epsg(dst_epsg)
    a, _, c, _, e, f = (list(src_transform) + [0, 0, 0])[:6]
    west, north = float(c), float(f)
    east = west + width * float(a)
    south = north + height * float(e)
    left, bottom, right, top = min(west, east), min(south, north), max(west, east), max(south, north)
    transform, dst_w, dst_h = calculate_default_transform(
        src_crs_obj, dst_crs, width, height, left=left, bottom=bottom, right=right, top=top
    )
    if dst_w > max_dim or dst_h > max_dim:
        scale = max(dst_w / max_dim, dst_h / max_dim)
        dst_w = max(32, int(dst_w / scale))
        dst_h = max(32, int(dst_h / scale))
        # Recalculate with a coarser resolution by using the same bounds.
        from rasterio.transform import from_bounds

        transform = from_bounds(left, bottom, right, top, dst_w, dst_h)
    out = {}
    for name, src in arrays.items():
        dst = np.zeros((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=np.asarray(src, dtype=np.float32),
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs_obj,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        out[name] = dst
    cellsize = abs(float(transform.a))
    xll = float(transform.c)
    yll = float(transform.f) + dst_h * float(transform.e)
    if float(transform.e) < 0:
        yll = float(transform.f) + dst_h * float(transform.e)
    return out, transform, dst_crs, cellsize, xll, yll, dst_h, dst_w


def toa_seconds_to_hours(toa_s: np.ndarray, tstop_s: float) -> np.ndarray:
    arr = np.asarray(toa_s, dtype=np.float64)
    hours = np.full(arr.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(arr) & (arr > 0) & (arr <= tstop_s * 1.05)
    hours[ok] = arr[ok] / 3600.0
    hours[ok & (arr <= 1.0)] = 0.0
    return hours


def p_from_eta(eta_hours: np.ndarray, horizon: float) -> np.ndarray:
    reached = np.isfinite(eta_hours) & (eta_hours <= horizon)
    return reached.astype(np.float64)


def find_elmfire_bin() -> Path:
    env = os.environ.get("ELMFIRE_BIN")
    if env:
        path = Path(env)
        if path.exists():
            return path
        raise ElmfireAdapterError(f"ELMFIRE_BIN={env} does not exist")
    install = os.environ.get("ELMFIRE_INSTALL_DIR")
    candidates = []
    if install:
        candidates.append(Path(install) / "elmfire")
    candidates.extend(
        [
            Path("/home/ubuntu/elmfire/build/linux/bin/elmfire"),
            Path("/usr/local/bin/elmfire"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise ElmfireAdapterError(
        "No ELMFIRE binary. Set ELMFIRE_BIN to the compiled Fortran executable "
        "(clone https://github.com/lautenberger/elmfire outside this tree)."
    )


def _read_toa(outputs: Path, tstop_s: float) -> np.ndarray:
    import rasterio

    tifs = sorted(outputs.glob("time_of_arrival*.tif")) + sorted(outputs.glob("toa_*.tif"))
    bils = sorted(outputs.glob("time_of_arrival*.bil")) + sorted(outputs.glob("toa_*.bil"))
    path = tifs[0] if tifs else (bils[0] if bils else None)
    if path is None:
        names = [p.name for p in outputs.iterdir()] if outputs.exists() else []
        raise ElmfireAdapterError(f"ELMFIRE wrote no TOA raster in {outputs} (files={names})")
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float64)
        nodata = ds.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return toa_seconds_to_hours(arr, tstop_s)


def run_elmfire_workdir(work: Path, bin_path: Path, timeout_s: float) -> None:
    env = os.environ.copy()
    env.setdefault("OMPI_MCA_btl", "^openib")
    cmd_direct = [str(bin_path), str(work / "inputs" / "elmfire.data")]
    try:
        proc = subprocess.run(
            cmd_direct,
            cwd=str(work),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise ElmfireAdapterError(f"ELMFIRE timed out after {timeout_s}s") from exc
    if proc.returncode == 0:
        return
    mpirun = shutil.which("mpirun")
    if not mpirun:
        raise ElmfireAdapterError(
            f"ELMFIRE rc={proc.returncode} stderr={proc.stderr[-800:]} stdout={proc.stdout[-400:]}"
        )
    proc2 = subprocess.run(
        [mpirun, "-n", os.environ.get("ELMFIRE_MPI_NP", "1"), str(bin_path), "./inputs/elmfire.data"],
        cwd=str(work),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env=env,
    )
    if proc2.returncode != 0:
        raise ElmfireAdapterError(
            f"ELMFIRE mpirun rc={proc2.returncode} stderr={proc2.stderr[-800:]} stdout={proc2.stdout[-400:]}"
        )


def build_result(eta: np.ndarray, inputs: dict[str, Any], horizon: float) -> dict[str, Any]:
    sigma = np.where(np.isfinite(eta), 0.0, np.nan)
    transform = list(inputs["transform"])
    return {
        "arrival_hours": eta,
        "eta_sigma_hours": sigma,
        "p_burn_24": p_from_eta(eta, 24.0),
        "p_burn_48": p_from_eta(eta, 48.0),
        "p_burn_72": p_from_eta(eta, min(72.0, horizon)),
        "spread_field_version": f"{ENGINE_VERSION}:n1:h{int(horizon)}:grid{eta.shape[0]}x{eta.shape[1]}",
        "engine": ENGINE_VERSION,
        "n_members": 1,
        "horizon_hours": horizon,
        "west": float(inputs["west"]),
        "south": float(inputs["south"]),
        "east": float(inputs["east"]),
        "north": float(inputs["north"]),
        "transform": transform,
        "shape": [int(eta.shape[0]), int(eta.shape[1])],
    }


ARRAY_KEYS = ("arrival_hours", "eta_sigma_hours", "p_burn_24", "p_burn_48", "p_burn_72")


def write_adapter_outputs(result: dict[str, Any], out_path: Path) -> Path:
    """Write grids as npz. Nested-list JSON of a LANDFIRE tile OOMs the service."""
    arrays_path = out_path.with_suffix(".npz")
    np.savez_compressed(
        arrays_path,
        **{key: np.asarray(result[key], dtype=np.float64) for key in ARRAY_KEYS},
    )
    meta = {key: result[key] for key in result if key not in ARRAY_KEYS}
    meta["arrays_path"] = str(arrays_path)
    for key in ARRAY_KEYS:
        # Presence satisfies the JSON contract; values live in the sidecar.
        meta[key] = None
    out_path.write_text(json.dumps(meta, default=str), encoding="utf-8")
    return arrays_path


def _jsonable(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    for key in ARRAY_KEYS:
        grid = np.asarray(out[key], dtype=np.float64)
        out[key] = np.where(np.isfinite(grid), grid, None).tolist()
    return out


def run_from_inputs(inputs: dict[str, Any], work: Path | None = None) -> dict[str, Any]:
    fbfm = _as_array(inputs.get("fbfm40"))
    if fbfm is None:
        raise ElmfireAdapterError("inputs missing fbfm40")
    height, width = fbfm.shape
    transform = inputs.get("transform")
    if hasattr(transform, "tolist"):
        transform = list(transform)
    src_crs = str(inputs.get("crs") or "EPSG:4326")
    weather = inputs.get("weather") or {}
    wind_u = float(weather.get("wind_u") or 0.0)
    wind_v = float(weather.get("wind_v") or 0.0)
    rh = weather.get("rh_pct")
    ws, wd = wind_speed_dir_mph(wind_u, wind_v)
    m1, m10, m100 = dead_fuel_moisture(rh if rh is None else float(rh))
    horizon = float(inputs.get("horizon_hours") or HORIZON_HOURS)
    tstop_s = horizon * 3600.0

    slope = _as_array(inputs.get("slope_deg"))
    aspect = _as_array(inputs.get("aspect_deg"))
    elev = _as_array(inputs.get("elev_m"))
    cc = _as_array(inputs.get("cc_pct"))
    ch = _as_array(inputs.get("ch_m"))
    cbh = _as_array(inputs.get("cbh_m"))
    cbd = _as_array(inputs.get("cbd_kg_m3"))
    if slope is None:
        slope = np.zeros_like(fbfm)
    if aspect is None:
        aspect = np.zeros_like(fbfm)
    if elev is None:
        elev = np.zeros_like(fbfm)
    if cc is None:
        cc = np.zeros_like(fbfm)
    if ch is None:
        ch = np.zeros_like(fbfm)
    if cbh is None:
        cbh = np.zeros_like(fbfm)
    if cbd is None:
        cbd = np.zeros_like(fbfm)

    lat0 = (float(inputs["south"]) + float(inputs["north"])) / 2.0
    lng0 = (float(inputs["west"]) + float(inputs["east"])) / 2.0
    epsg = utm_epsg(lat0, lng0)

    arrays = {
        "fbfm40": np.nan_to_num(fbfm, nan=91),
        "slp": np.nan_to_num(slope, nan=0.0),
        "asp": np.nan_to_num(aspect, nan=0.0),
        "dem": np.nan_to_num(elev, nan=0.0),
        "cc": np.nan_to_num(cc, nan=0.0),
        "ch": np.nan_to_num(ch, nan=0.0) * 10.0,  # ELMFIRE: 10 * meters
        "cbh": np.nan_to_num(cbh, nan=0.0) * 10.0,
        "cbd": np.nan_to_num(cbd, nan=0.0) * 100.0,
    }
    import rasterio
    from rasterio.transform import Affine

    src_transform = Affine(*list(transform)[:6]) if not isinstance(transform, Affine) else transform
    projected, dst_transform, dst_crs, cellsize, xll, yll, dst_h, dst_w = _reproject_to_utm(
        arrays, src_transform, src_crs, epsg
    )
    rings = inputs.get("perimeter_rings") or []
    ring_list = rings
    if rings and isinstance(rings[0], (list, tuple)):
        first = rings[0]
        # Already a list of rings (each ring is a list of [lng, lat]).
        if first and isinstance(first[0], (list, tuple)) and len(first[0]) >= 2:
            ring_list = rings
        # Single ring passed as [[lng, lat], ...].
        elif first and isinstance(first[0], (int, float)):
            ring_list = [rings]
    phi = rasterize_phi(dst_h, dst_w, dst_transform, ring_list, dst_crs)
    if not np.any(phi < 0):
        ignitions = inputs.get("ignition_points") or []
        if ignitions:
            from pyproj import Transformer

            to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            inv = ~dst_transform
            for pt in ignitions:
                x, y = to_utm.transform(float(pt["lng"]), float(pt["lat"]))
                col, row = inv * (x, y)
                r, c = int(row), int(col)
                if 0 <= r < dst_h and 0 <= c < dst_w:
                    phi[r, c] = -1.0
    if not np.any(phi < 0):
        raise ElmfireAdapterError("no seed cells in phi (empty perimeter and ignition)")

    ws_grid = np.full((dst_h, dst_w), ws, dtype=np.float32)
    wd_grid = np.full((dst_h, dst_w), wd, dtype=np.float32)
    m1_grid = np.full((dst_h, dst_w), m1, dtype=np.float32)
    m10_grid = np.full((dst_h, dst_w), m10, dtype=np.float32)
    m100_grid = np.full((dst_h, dst_w), m100, dtype=np.float32)
    adj = np.ones((dst_h, dst_w), dtype=np.float32)

    own = work is None
    work = Path(work) if work is not None else Path(tempfile.mkdtemp(prefix="elmfire_job_"))
    try:
        (work / "inputs").mkdir(parents=True, exist_ok=True)
        (work / "outputs").mkdir(parents=True, exist_ok=True)
        (work / "scratch").mkdir(parents=True, exist_ok=True)
        int16_maps = {
            "asp": projected["asp"],
            "cbd": projected["cbd"],
            "cbh": projected["cbh"],
            "cc": projected["cc"],
            "ch": projected["ch"],
            "dem": projected["dem"],
            "fbfm40": projected["fbfm40"],
            "slp": projected["slp"],
        }
        for name, arr in int16_maps.items():
            _write_tif(work / "inputs" / f"{name}.tif", np.clip(np.rint(arr), -32768, 32767), dst_transform, dst_crs, "int16", -9999)
        float_maps = {
            "adj": adj,
            "phi": phi,
            "ws": ws_grid,
            "wd": wd_grid,
            "m1": m1_grid,
            "m10": m10_grid,
            "m100": m100_grid,
        }
        for name, arr in float_maps.items():
            _write_tif(work / "inputs" / f"{name}.tif", arr, dst_transform, dst_crs, "float32", -9999.0)
        write_namelist(
            work / "inputs" / "elmfire.data",
            epsg=epsg,
            cellsize=cellsize,
            xll=xll,
            yll=yll,
            tstop_s=tstop_s,
        )
        bin_path = find_elmfire_bin()
        timeout_s = float(os.environ.get("SPREAD_ENGINE_TIMEOUT_S", "1800"))
        run_elmfire_workdir(work, bin_path, timeout_s)
        eta_utm = _read_toa(work / "outputs", tstop_s)
        # Warp TOA back to the caller's WGS84 grid so SpreadField.sample still uses lat/lng.
        from rasterio.warp import Resampling, reproject

        eta_src = np.where(np.isfinite(eta_utm), eta_utm.astype(np.float32), -1.0)
        eta_dst = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=eta_src,
            destination=eta_dst,
            src_transform=dst_transform,
            src_crs=dst_crs,
            dst_transform=src_transform,
            dst_crs=src_crs,
            resampling=Resampling.nearest,
        )
        eta = np.where(eta_dst < 0, np.nan, eta_dst.astype(np.float64))
        return build_result(eta, inputs, horizon)
    finally:
        keep = os.environ.get("ELMFIRE_KEEP_WORKDIR")
        if own and not keep:
            shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        sys.stderr.write("usage: elmfire_adapter in.json out.json\n")
        return 2
    in_path, out_path = Path(argv[0]), Path(argv[1])
    try:
        inputs = load_inputs(in_path)
        result = run_from_inputs(inputs)
        write_adapter_outputs(result, out_path)
    except Exception as exc:
        sys.stderr.write(f"elmfire_adapter: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
