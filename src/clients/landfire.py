"""LANDFIRE Product Service client (SRS FR-21).

Fetches FBFM40 fuel, canopy (CBD/CBH/CC/CH), and DEM/slope/aspect rasters for an
incident AOI. The live API shape was verified 2026-08-27 against
https://lfps.usgs.gov before this parser was written - see DECISIONS.md.

Live-verified facts this module depends on (do not "fix" them back to the old
ArcGIS GPServer docs):

- Machine API is `/api/job/submit`, `/api/job/status`, `/api/products`,
  `/api/healthCheck`. The historical
  `/arcgis/rest/services/LandfireProductService/GPServer/.../submitJob` now
  serves the Next.js HTML form, not JSON.
- Submit accepts JSON with `Email`, `Layer_List`, `Area_of_Interest` (W S E N,
  EPSG:4326), optional `Resample_Resolution` and `Output_Projection`.
- Submit returns `{jobId, status, job:{layerList, areaOfInterest, aoiSize,
  geoArea, ...}}`. `jobId` is a UUID, not an ESRI job id.
- Status returns `{jobId, queuePosition, status, messages[], outputFile}` where
  `outputFile` is a zip of a multi-band GeoTIFF. Band descriptions look like
  `LF2024_FBFM40_CONUS` (region suffix appended to the Layer_List name).
- Default CRS is a local Albers centered on the AOI. Pass `Output_Projection`
  `"4326"` to get WGS84.
- dtype int16, nodata -9999.0. FBFM40 codes are Scott & Burgan (91 urban, 98
  water, 101-189 burnable, ...). CC is percent 0-100. Elev is meters.
- `/api/products` is the live layer catalog. Fuel products are versioned
  `LF2024_FBFM40` etc.; topographic layers are still `LF2020_Elev` /
  `LF2020_SlpD` / `LF2020_Asp`. LF2025 fuel exists but is only SW/NW, not
  full CONUS, so the picker prefers the newest CONUS-complete layer.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np
import rasterio
from rasterio.io import MemoryFile

from src.logging_ import tool_logger

LFPS_BASE_URL = "https://lfps.usgs.gov"
USER_AGENT = "fire-copilot/2.0 (contact@example.com)"
DEFAULT_EMAIL = "ashes-v2@example.com"
DEFAULT_RESAMPLE_M = 90
DEFAULT_OUTPUT_CRS = "4326"

# Scott & Burgan 40 non-burnable / no-fuel codes (NB1-NB9). Never treat these
# as a Rothermel carrier. 0 and the LFPS nodata sentinel are also non-burnable.
NON_BURNABLE_FBFM40 = {0, 91, 92, 93, 98, 99}

# Canopy height / base height are stored as meters * 10; CBD as kg/m^3 * 100.
# Verified against LANDFIRE product documentation and the live LFPS user guide
# value-limits table (CH/CBH max 1000, CBD max 50).
CH_SCALE = 10.0
CBH_SCALE = 10.0
CBD_SCALE = 100.0

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "landfire"

ENGINE_ACRONYMS = ("FBFM40", "CC", "CH", "CBD", "CBH")
TOPO_ACRONYMS = ("Elev", "SlpD", "Asp")


class LandfireRequestFailed(RuntimeError):
    """Raised after retries are exhausted or LFPS reports Failed."""


@dataclass
class LandfireProduct:
    product_name: str
    theme: str
    layer_name: str
    acronym: str
    version: str
    conus: bool
    geo_areas: str


@dataclass
class LandfireStack:
    """One multi-band LANDFIRE clip for an AOI, already opened into numpy.

    Arrays are row-major (north-up). Coordinates of cell centers can be recovered
    with `xy(row, col)`. `vintage` is the fuel-layer version string (e.g. LF2024).
    """

    fbfm40: np.ndarray
    cc_pct: np.ndarray
    ch_m: np.ndarray
    cbd_kg_m3: np.ndarray
    cbh_m: np.ndarray
    elev_m: np.ndarray
    slope_deg: np.ndarray
    aspect_deg: np.ndarray
    transform: Any
    crs: str
    nodata: float
    layer_list: list[str]
    vintage: str
    west: float
    south: float
    east: float
    north: float
    resample_m: int
    source_url: str
    band_names: dict[str, str] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.fbfm40.shape[0])

    @property
    def width(self) -> int:
        return int(self.fbfm40.shape[1])

    def xy(self, row: int, col: int) -> tuple[float, float]:
        """Return (lng, lat) of a cell center when CRS is EPSG:4326."""
        lng, lat = rasterio.transform.xy(self.transform, row, col, offset="center")
        return float(lng), float(lat)

    def index(self, lng: float, lat: float) -> tuple[int, int]:
        row, col = rasterio.transform.rowcol(self.transform, lng, lat)
        return int(row), int(col)

    def sample_fbfm40(self, lat: float, lng: float) -> int | None:
        try:
            row, col = self.index(lng, lat)
        except Exception:
            return None
        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            return None
        value = int(self.fbfm40[row, col])
        if value == int(self.nodata) or value < 0:
            return None
        return value

    def burnable_mask(self) -> np.ndarray:
        """True where FBFM40 is a Scott & Burgan burnable fuel model."""
        vals = self.fbfm40.astype(np.int32)
        nodata = int(self.nodata) if self.nodata is not None else -9999
        burnable = np.ones(vals.shape, dtype=bool)
        burnable &= vals != nodata
        for code in NON_BURNABLE_FBFM40:
            burnable &= vals != code
        return burnable


def _cache_key(west: float, south: float, east: float, north: float, layers: Iterable[str], resample_m: int, crs: str) -> str:
    payload = json.dumps(
        {
            "bbox": [round(west, 5), round(south, 5), round(east, 5), round(north, 5)],
            "layers": list(layers),
            "resample_m": resample_m,
            "crs": crs,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def pick_conus_layer(products: list[LandfireProduct], acronym: str) -> LandfireProduct | None:
    """Newest CONUS-complete layer for an acronym; fall back to any CONUS layer.

    LF2025 fuel is live but only SW/NW. A book-of-sites copilot covering CONUS
    therefore prefers `geoAreas == "All"` (currently LF2024 for fuels, LF2020
    for topo), verified against `/api/products` on 2026-08-27.
    """
    candidates = [p for p in products if p.acronym == acronym and p.conus]
    if not candidates:
        return None

    def sort_key(p: LandfireProduct) -> tuple[int, int, int]:
        geo = (p.geo_areas or "").strip().lower()
        full = 1 if geo == "all" else 0
        year = 0
        digits = "".join(ch for ch in p.version if ch.isdigit())
        if digits:
            year = int(digits)
        # Seasonal suffixes (SP26/SU26/ES26) are not a full-year CONUS product.
        remainder = p.layer_name.replace(p.version + "_", "", 1)
        non_seasonal = 0 if "_" in remainder else 1
        return (full, non_seasonal, year)

    return sorted(candidates, key=sort_key)[-1]


def resolve_engine_layers(products: list[LandfireProduct]) -> list[str]:
    layers: list[str] = []
    for acronym in ENGINE_ACRONYMS + TOPO_ACRONYMS:
        picked = pick_conus_layer(products, acronym)
        if picked is None:
            raise LandfireRequestFailed(f"No CONUS LANDFIRE layer for acronym {acronym}")
        layers.append(picked.layer_name)
    return layers


def _band_matches(description: str | None, layer_name: str) -> bool:
    """LFPS appends a region suffix (`_CONUS`) to the requested layer name."""
    if not description:
        return False
    desc = description.strip()
    return desc == layer_name or desc.startswith(layer_name + "_")


def _decode_band(acronym: str, raw: np.ndarray, nodata: float) -> np.ndarray:
    arr = raw.astype(np.float64)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    if acronym == "CH":
        return arr / CH_SCALE
    if acronym == "CBH":
        return arr / CBH_SCALE
    if acronym == "CBD":
        return arr / CBD_SCALE
    return arr


class LANDFIREClient:
    def __init__(
        self,
        timeout: float = 60.0,
        email: str = DEFAULT_EMAIL,
        cache_dir: Path | str | None = None,
        poll_seconds: float = 3.0,
        poll_timeout_seconds: float = 420.0,
    ):
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        self._email = email
        self._cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self._poll_seconds = poll_seconds
        self._poll_timeout_seconds = poll_timeout_seconds
        self._products: list[LandfireProduct] | None = None

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        try:
            resp = self._client.get(f"{LFPS_BASE_URL}/api/healthCheck")
            data = resp.json()
            return bool(data.get("success"))
        except (httpx.HTTPError, ValueError):
            return False

    def list_products(self, site_id: str | None = None) -> list[LandfireProduct]:
        if self._products is not None:
            return self._products
        start = time.monotonic()
        try:
            resp = self._client.get(f"{LFPS_BASE_URL}/api/products")
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call("landfire:products", {}, None, site_id, latency_ms, error=str(exc))
            raise LandfireRequestFailed(f"LANDFIRE products catalog failed: {exc}") from exc

        products = [
            LandfireProduct(
                product_name=str(item.get("productName") or ""),
                theme=str(item.get("theme") or ""),
                layer_name=str(item.get("layerName") or ""),
                acronym=str(item.get("acronym") or ""),
                version=str(item.get("version") or ""),
                conus=bool(item.get("conus")),
                geo_areas=str(item.get("geoAreas") or ""),
            )
            for item in payload.get("products", [])
        ]
        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call("landfire:products", {}, {"count": len(products)}, site_id, latency_ms)
        self._products = products
        return products

    def fetch_aoi(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        site_id: str | None = None,
        resample_m: int = DEFAULT_RESAMPLE_M,
        output_crs: str = DEFAULT_OUTPUT_CRS,
        layer_list: list[str] | None = None,
    ) -> LandfireStack:
        if west >= east or south >= north:
            raise ValueError("AOI bbox must have west<east and south<north")

        products = self.list_products(site_id=site_id)
        layers = layer_list or resolve_engine_layers(products)
        key = _cache_key(west, south, east, north, layers, resample_m, output_crs)
        cached = self._read_cache(key)
        if cached is not None:
            tool_logger.log_tool_call(
                "landfire:cache_hit",
                {"west": west, "south": south, "east": east, "north": north},
                {"key": key},
                site_id,
                0.0,
            )
            return cached

        job = self._submit_job(layers, west, south, east, north, resample_m, output_crs, site_id)
        job_id = job["jobId"]
        status = self._poll_job(job_id, site_id)
        if status.get("status") != "Succeeded":
            raise LandfireRequestFailed(f"LFPS job {job_id} ended {status.get('status')}")
        zip_url = status.get("outputFile")
        if not zip_url:
            raise LandfireRequestFailed(f"LFPS job {job_id} succeeded with no outputFile")

        tif_bytes, source_url = self._download_zip_tif(zip_url, site_id)
        stack = self._parse_geotiff(
            tif_bytes,
            layers=layers,
            west=west,
            south=south,
            east=east,
            north=north,
            resample_m=resample_m,
            source_url=source_url,
        )
        self._write_cache(key, stack)
        return stack

    def _submit_job(
        self,
        layers: list[str],
        west: float,
        south: float,
        east: float,
        north: float,
        resample_m: int,
        output_crs: str,
        site_id: str | None,
    ) -> dict[str, Any]:
        aoi = f"{west} {south} {east} {north}"
        payload = {
            "Email": self._email,
            "Layer_List": ";".join(layers),
            "Area_of_Interest": aoi,
            "Resample_Resolution": int(resample_m),
            "Output_Projection": str(output_crs),
        }
        start = time.monotonic()
        try:
            resp = self._client.post(f"{LFPS_BASE_URL}/api/job/submit", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call("landfire:submit", payload, None, site_id, latency_ms, error=str(exc))
            raise LandfireRequestFailed(f"LFPS submit failed: {exc}") from exc
        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call(
            "landfire:submit",
            {"Layer_List": payload["Layer_List"], "Area_of_Interest": aoi, "Resample_Resolution": resample_m},
            {"jobId": data.get("jobId"), "status": data.get("status"), "aoiSize": (data.get("job") or {}).get("aoiSize")},
            site_id,
            latency_ms,
        )
        if not data.get("jobId"):
            raise LandfireRequestFailed(f"LFPS submit returned no jobId: {data}")
        return data

    def _poll_job(self, job_id: str, site_id: str | None) -> dict[str, Any]:
        deadline = time.monotonic() + self._poll_timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            start = time.monotonic()
            try:
                resp = self._client.get(f"{LFPS_BASE_URL}/api/job/status", params={"JobId": job_id})
                resp.raise_for_status()
                last = resp.json()
            except httpx.HTTPError as exc:
                latency_ms = (time.monotonic() - start) * 1000
                tool_logger.log_tool_call("landfire:status", {"JobId": job_id}, None, site_id, latency_ms, error=str(exc))
                time.sleep(self._poll_seconds)
                continue
            status = last.get("status")
            if status in {"Succeeded", "Failed", "Canceled"}:
                latency_ms = (time.monotonic() - start) * 1000
                tool_logger.log_tool_call(
                    "landfire:status",
                    {"JobId": job_id},
                    {"status": status, "queuePosition": last.get("queuePosition"), "has_output": bool(last.get("outputFile"))},
                    site_id,
                    latency_ms,
                )
                return last
            time.sleep(self._poll_seconds)
        raise LandfireRequestFailed(f"LFPS job {job_id} timed out after {self._poll_timeout_seconds}s (last={last.get('status')})")

    def _download_zip_tif(self, zip_url: str, site_id: str | None) -> tuple[bytes, str]:
        start = time.monotonic()
        try:
            resp = self._client.get(zip_url)
            resp.raise_for_status()
            blob = resp.content
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call("landfire:download", {"url": zip_url}, None, site_id, latency_ms, error=str(exc))
            raise LandfireRequestFailed(f"LFPS zip download failed: {exc}") from exc
        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call("landfire:download", {"url": zip_url}, {"bytes": len(blob)}, site_id, latency_ms)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_names:
                raise LandfireRequestFailed(f"LFPS zip contained no .tif (files={zf.namelist()})")
            return zf.read(tif_names[0]), zip_url

    def _parse_geotiff(
        self,
        tif_bytes: bytes,
        layers: list[str],
        west: float,
        south: float,
        east: float,
        north: float,
        resample_m: int,
        source_url: str,
    ) -> LandfireStack:
        with MemoryFile(tif_bytes) as mem:
            with mem.open() as ds:
                nodata = ds.nodata if ds.nodata is not None else -9999.0
                descriptions = list(ds.descriptions)
                band_by_acronym: dict[str, np.ndarray] = {}
                band_names: dict[str, str] = {}
                for idx, desc in enumerate(descriptions, start=1):
                    matched = None
                    for layer_name in layers:
                        if _band_matches(desc, layer_name):
                            matched = layer_name
                            break
                    if matched is None:
                        continue
                    # LF2024_FBFM40 -> FBFM40; LF2020_Elev -> Elev; LF2020_SlpD -> SlpD
                    parts = matched.split("_")
                    acronym = "_".join(parts[1:]) if len(parts) > 1 else matched
                    raw = ds.read(idx)
                    band_by_acronym[acronym] = _decode_band(acronym, raw, nodata)
                    band_names[acronym] = str(desc)
                missing = [a for a in ENGINE_ACRONYMS + TOPO_ACRONYMS if a not in band_by_acronym]
                if missing:
                    # Some jobs request a subset (tests, degraded retries). Fill missing
                    # with NaN of the right shape rather than inventing fuel codes.
                    shape = ds.read(1).shape
                    for acronym in missing:
                        band_by_acronym[acronym] = np.full(shape, np.nan, dtype=np.float64)
                fuel_vintage = next((ly.split("_")[0] for ly in layers if "FBFM40" in ly), "unknown")
                return LandfireStack(
                    fbfm40=np.nan_to_num(band_by_acronym["FBFM40"], nan=float(nodata)).astype(np.int16),
                    cc_pct=band_by_acronym["CC"],
                    ch_m=band_by_acronym["CH"],
                    cbd_kg_m3=band_by_acronym["CBD"],
                    cbh_m=band_by_acronym["CBH"],
                    elev_m=band_by_acronym["Elev"],
                    slope_deg=band_by_acronym["SlpD"],
                    aspect_deg=band_by_acronym["Asp"],
                    transform=ds.transform,
                    crs=str(ds.crs) if ds.crs else f"EPSG:{DEFAULT_OUTPUT_CRS}",
                    nodata=float(nodata),
                    layer_list=list(layers),
                    vintage=fuel_vintage,
                    west=west,
                    south=south,
                    east=east,
                    north=north,
                    resample_m=resample_m,
                    source_url=source_url,
                    band_names=band_names,
                )

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.npz"

    def _read_cache(self, key: str) -> LandfireStack | None:
        path = self._cache_path(key)
        meta_path = path.with_suffix(".json")
        if not path.exists() or not meta_path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                transform = rasterio.Affine(*meta["transform"])
                return LandfireStack(
                    fbfm40=data["fbfm40"],
                    cc_pct=data["cc_pct"],
                    ch_m=data["ch_m"],
                    cbd_kg_m3=data["cbd_kg_m3"],
                    cbh_m=data["cbh_m"],
                    elev_m=data["elev_m"],
                    slope_deg=data["slope_deg"],
                    aspect_deg=data["aspect_deg"],
                    transform=transform,
                    crs=meta["crs"],
                    nodata=meta["nodata"],
                    layer_list=meta["layer_list"],
                    vintage=meta["vintage"],
                    west=meta["west"],
                    south=meta["south"],
                    east=meta["east"],
                    north=meta["north"],
                    resample_m=meta["resample_m"],
                    source_url=meta["source_url"],
                    band_names=meta.get("band_names") or {},
                )
        except Exception:
            return None

    def _write_cache(self, key: str, stack: LandfireStack) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        np.savez_compressed(
            path,
            fbfm40=stack.fbfm40,
            cc_pct=stack.cc_pct,
            ch_m=stack.ch_m,
            cbd_kg_m3=stack.cbd_kg_m3,
            cbh_m=stack.cbh_m,
            elev_m=stack.elev_m,
            slope_deg=stack.slope_deg,
            aspect_deg=stack.aspect_deg,
        )
        meta = {
            "transform": list(stack.transform)[:6],
            "crs": stack.crs,
            "nodata": stack.nodata,
            "layer_list": stack.layer_list,
            "vintage": stack.vintage,
            "west": stack.west,
            "south": stack.south,
            "east": stack.east,
            "north": stack.north,
            "resample_m": stack.resample_m,
            "source_url": stack.source_url,
            "band_names": stack.band_names,
        }
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
