"""Incident-first AOI geometry (SRS FR-4 / FR-5).

The uniform geometry contract is `{mode, geom, cells[], source_layer, vintage}`.
V1 used `point`/`parcel` with `cells` length 1. V2 adds `aoi`: a wind-projected
buffer around an active WFIGS perimeter, clipped to burnable LANDFIRE FBFM40.

`model_infer` runs per cell. The book of sites is intersected with the AOI
separately (a site inside the AOI is in play; the AOI itself is the spread-engine
domain). Coarse grids set `aoi_coarse_advisory`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pyproj import Geod
from shapely.geometry import mapping, Point, Polygon, MultiPolygon, shape
from shapely.ops import unary_union

from src.clients.landfire import LandfireStack, NON_BURNABLE_FBFM40
from src.clients.wfigs import WFIGSPerimeter

_GEOD = Geod(ellps="WGS84")

GeometryMode = Literal["point", "parcel", "aoi"]

DEFAULT_BASE_BUFFER_KM = 5.0
DEFAULT_DOWNWIND_BUFFER_KM = 25.0
DEFAULT_UPWIND_BUFFER_KM = 2.0
DEFAULT_MAX_CELLS = 12_000


@dataclass
class Cell:
    lat: float
    lng: float
    row: int
    col: int
    fbfm40: int | None = None


@dataclass
class Geometry:
    """SRS FR-5 contract. `geom` is a GeoJSON geometry dict."""

    mode: GeometryMode
    geom: dict[str, Any]
    cells: list[Cell]
    source_layer: str
    vintage: str
    flags: list[str] = field(default_factory=list)
    west: float | None = None
    south: float | None = None
    east: float | None = None
    north: float | None = None

    def contains_point(self, lat: float, lng: float) -> bool:
        if self.mode == "point" and self.cells:
            return abs(self.cells[0].lat - lat) < 1e-8 and abs(self.cells[0].lng - lng) < 1e-8
        try:
            poly = shape(self.geom)
        except Exception:
            return False
        return bool(poly.contains(Point(lng, lat)) or poly.touches(Point(lng, lat)))


def point_geometry(lat: float, lng: float, source_layer: str = "site", vintage: str = "n/a") -> Geometry:
    return Geometry(
        mode="point",
        geom={"type": "Point", "coordinates": [lng, lat]},
        cells=[Cell(lat=lat, lng=lng, row=0, col=0)],
        source_layer=source_layer,
        vintage=vintage,
        west=lng,
        south=lat,
        east=lng,
        north=lat,
    )


def _rings_to_shapely(rings: list[list[tuple[float, float]]]) -> Polygon | MultiPolygon | None:
    polygons = []
    for ring in rings:
        if len(ring) < 3:
            continue
        coords = list(ring)
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        try:
            poly = Polygon(coords)
        except Exception:
            continue
        if poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        polygons.append(poly)
    if not polygons:
        return None
    if len(polygons) == 1:
        return polygons[0]
    return unary_union(polygons)


def _wind_bearing_deg(wind_u: float | None, wind_v: float | None) -> float | None:
    """Direction the fire is pushed toward (where the wind is going), degrees from north."""
    if wind_u is None or wind_v is None:
        return None
    if wind_u == 0.0 and wind_v == 0.0:
        return None
    return math.degrees(math.atan2(wind_u, wind_v)) % 360


def wind_projected_polygon(
    perimeter: WFIGSPerimeter,
    wind_u: float | None,
    wind_v: float | None,
    base_buffer_km: float = DEFAULT_BASE_BUFFER_KM,
    downwind_buffer_km: float = DEFAULT_DOWNWIND_BUFFER_KM,
    upwind_buffer_km: float = DEFAULT_UPWIND_BUFFER_KM,
) -> tuple[Any, list[str]]:
    """Buffer a WFIGS perimeter, stretched downwind.

    Buffer distances are geodesic meters converted to a geographic buffer in
    degrees at the perimeter centroid. That approximation is documented: at
    CONUS latitudes a 1 km buffer is ~0.009 degrees, and the AOI is later
    clipped to burnable LANDFIRE cells so the degree conversion does not
    invent a fuel boundary.
    """
    flags: list[str] = []
    geom = _rings_to_shapely(perimeter.geometry_rings)
    if geom is None or geom.is_empty:
        return None, flags

    centroid = geom.centroid
    lat0, lng0 = centroid.y, centroid.x
    meters_per_deg_lat = 111_000.0
    meters_per_deg_lng = 111_000.0 * max(math.cos(math.radians(lat0)), 0.1)

    def buf_deg(km: float) -> float:
        # Conservative: use the smaller of lat/lng degree sizes so we do not
        # under-buffer in longitude.
        return (km * 1000.0) / min(meters_per_deg_lat, meters_per_deg_lng)

    base = geom.buffer(buf_deg(base_buffer_km))
    bearing = _wind_bearing_deg(wind_u, wind_v)
    if bearing is None:
        flags.append("aoi_isotropic_buffer")
        return base, flags

    down_lng, down_lat, _ = _GEOD.fwd(lng0, lat0, bearing, downwind_buffer_km * 1000.0)
    up_lng, up_lat, _ = _GEOD.fwd(lng0, lat0, (bearing + 180.0) % 360, upwind_buffer_km * 1000.0)
    down_shift_x = down_lng - lng0
    down_shift_y = down_lat - lat0
    up_shift_x = up_lng - lng0
    up_shift_y = up_lat - lat0

    from shapely.affinity import translate

    downwind = translate(geom, xoff=down_shift_x, yoff=down_shift_y).buffer(buf_deg(base_buffer_km))
    upwind = translate(geom, xoff=up_shift_x, yoff=up_shift_y).buffer(buf_deg(min(upwind_buffer_km, base_buffer_km)))
    return unary_union([base, downwind, upwind]), flags


def _cell_limit_stride(n_burnable: int, max_cells: int) -> int:
    if n_burnable <= max_cells:
        return 1
    # Smallest stride whose sampled count is <= max_cells.
    return int(math.ceil(math.sqrt(n_burnable / max_cells)))


def build_incident_aoi(
    perimeter: WFIGSPerimeter,
    landfire: LandfireStack,
    wind_u: float | None = None,
    wind_v: float | None = None,
    base_buffer_km: float = DEFAULT_BASE_BUFFER_KM,
    downwind_buffer_km: float = DEFAULT_DOWNWIND_BUFFER_KM,
    upwind_buffer_km: float = DEFAULT_UPWIND_BUFFER_KM,
    max_cells: int = DEFAULT_MAX_CELLS,
) -> Geometry:
    poly, flags = wind_projected_polygon(
        perimeter, wind_u, wind_v, base_buffer_km, downwind_buffer_km, upwind_buffer_km
    )
    if poly is None or poly.is_empty:
        return Geometry(
            mode="aoi",
            geom={"type": "Polygon", "coordinates": []},
            cells=[],
            source_layer="wfigs_perimeter+landfire_fbfm40",
            vintage=landfire.vintage,
            flags=flags + ["aoi_empty"],
        )

    burnable = landfire.burnable_mask()
    cells: list[Cell] = []
    min_row, max_row = landfire.height, -1
    min_col, max_col = landfire.width, -1
    for row in range(landfire.height):
        for col in range(landfire.width):
            if not burnable[row, col]:
                continue
            lng, lat = landfire.xy(row, col)
            if not poly.contains(Point(lng, lat)) and not poly.touches(Point(lng, lat)):
                continue
            fbfm = int(landfire.fbfm40[row, col])
            if fbfm in NON_BURNABLE_FBFM40:
                continue
            cells.append(Cell(lat=lat, lng=lng, row=row, col=col, fbfm40=fbfm))
            min_row, max_row = min(min_row, row), max(max_row, row)
            min_col, max_col = min(min_col, col), max(max_col, col)

    stride = _cell_limit_stride(len(cells), max_cells)
    if stride > 1:
        cells = [c for i, c in enumerate(cells) if (c.row % stride == 0 and c.col % stride == 0)]
        flags.append("aoi_coarse_advisory")

    west, south, east, north = poly.bounds
    return Geometry(
        mode="aoi",
        geom=mapping(poly),
        cells=cells,
        source_layer="wfigs_perimeter+landfire_fbfm40",
        vintage=landfire.vintage,
        flags=flags,
        west=float(west),
        south=float(south),
        east=float(east),
        north=float(north),
    )


def dense_grid_in_geom(geom: Geometry, max_cells: int) -> list[tuple[float, float]]:
    """Regular lat/lng grid clipped to the AOI polygon, capped at max_cells (FR-56).

    Used for the Response Agent water map. Samples the geographic envelope, not
    only burnable fuel cells, because water often sits on non-burnable FBFM40
    (98 water / 91 urban). Never invents a point outside the AOI polygon.
    """
    if max_cells <= 0:
        return []
    if geom.west is None or geom.south is None or geom.east is None or geom.north is None:
        return []
    west, south, east, north = geom.west, geom.south, geom.east, geom.north
    if west == east or south == north:
        if geom.cells:
            return [(geom.cells[0].lat, geom.cells[0].lng)][:max_cells]
        return []
    n = max(1, int(math.ceil(math.sqrt(max_cells))))
    points: list[tuple[float, float]] = []
    # If the polygon is a small fraction of its envelope, grow the lattice until
    # we hit max_cells or a documented cap so we do not under-sample.
    for stride_n in (n, n * 2, n * 3):
        points = []
        for i in range(stride_n):
            for j in range(stride_n):
                lat = south + (north - south) * (i + 0.5) / stride_n
                lng = west + (east - west) * (j + 0.5) / stride_n
                if geom.contains_point(lat, lng):
                    points.append((lat, lng))
                    if len(points) >= max_cells:
                        return points
        if points:
            return points
    return points


def aoi_bbox_for_fetch(
    perimeter: WFIGSPerimeter,
    wind_u: float | None,
    wind_v: float | None,
    pad_km: float = 2.0,
    base_buffer_km: float = DEFAULT_BASE_BUFFER_KM,
    downwind_buffer_km: float = DEFAULT_DOWNWIND_BUFFER_KM,
    upwind_buffer_km: float = DEFAULT_UPWIND_BUFFER_KM,
) -> tuple[float, float, float, float] | None:
    """Bounding box to request from LANDFIRE, slightly padded past the wind buffer."""
    poly, _ = wind_projected_polygon(
        perimeter, wind_u, wind_v, base_buffer_km, downwind_buffer_km, upwind_buffer_km
    )
    if poly is None or poly.is_empty:
        return None
    west, south, east, north = poly.bounds
    lat0 = (south + north) / 2.0
    dlat = pad_km / 111.0
    dlng = pad_km / (111.0 * max(math.cos(math.radians(lat0)), 0.1))
    return west - dlng, south - dlat, east + dlng, north + dlat
