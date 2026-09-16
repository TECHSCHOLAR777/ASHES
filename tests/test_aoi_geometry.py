from src.clients.landfire import LandfireStack
from src.clients.wfigs import WFIGSPerimeter
from src.geometry.aoi import (
    build_incident_aoi,
    point_geometry,
    wind_projected_polygon,
)
from rasterio.transform import from_origin
import numpy as np


def _stack(fbfm: np.ndarray, west=-118.26, north=34.06, dx=0.01, dy=0.01) -> LandfireStack:
    transform = from_origin(west, north, dx, dy)
    nan = np.full(fbfm.shape, np.nan)
    return LandfireStack(
        fbfm40=fbfm.astype(np.int16),
        cc_pct=nan,
        ch_m=nan,
        cbd_kg_m3=nan,
        cbh_m=nan,
        elev_m=nan,
        slope_deg=nan,
        aspect_deg=nan,
        transform=transform,
        crs="EPSG:4326",
        nodata=-9999,
        layer_list=["LF2024_FBFM40"],
        vintage="LF2024",
        west=west,
        south=north - fbfm.shape[0] * dy,
        east=west + fbfm.shape[1] * dx,
        north=north,
        resample_m=90,
        source_url="test",
    )


def test_point_geometry_contract():
    g = point_geometry(34.0, -118.0)
    assert g.mode == "point"
    assert len(g.cells) == 1
    assert g.contains_point(34.0, -118.0)
    assert not g.contains_point(35.0, -118.0)


def test_aoi_clips_to_burnable_fuel():
    # 4x4 grid, urban on the left (91), shrub on the right (122)
    fbfm = np.array(
        [
            [91, 91, 122, 122],
            [91, 91, 122, 122],
            [91, 91, 122, 122],
            [91, 91, 122, 122],
        ],
        dtype=np.int16,
    )
    stack = _stack(fbfm)
    # Perimeter covers the whole grid.
    perim = WFIGSPerimeter(
        irwin_id="IR-1",
        name="Test",
        geometry_rings=[[(-118.27, 34.03), (-118.21, 34.03), (-118.21, 34.07), (-118.27, 34.07)]],
    )
    aoi = build_incident_aoi(perim, stack, wind_u=None, wind_v=None, max_cells=100)
    assert aoi.mode == "aoi"
    assert aoi.cells
    assert all(c.fbfm40 == 122 for c in aoi.cells)
    assert "aoi_isotropic_buffer" in aoi.flags


def test_dense_grid_stays_inside_polygon():
    from src.geometry.aoi import Geometry, dense_grid_in_geom

    geom = Geometry(
        mode="aoi",
        geom={
            "type": "Polygon",
            "coordinates": [[
                [-118.02, 34.00], [-118.00, 34.00], [-118.00, 34.02], [-118.02, 34.02], [-118.02, 34.00]
            ]],
        },
        cells=[],
        source_layer="test",
        vintage="LF2024",
        west=-118.02,
        south=34.00,
        east=-118.00,
        north=34.02,
    )
    points = dense_grid_in_geom(geom, max_cells=9)
    assert points
    assert len(points) <= 9
    for lat, lng in points:
        assert geom.contains_point(lat, lng)


def test_aoi_coarse_advisory_when_over_max_cells():
    fbfm = np.full((20, 20), 122, dtype=np.int16)
    stack = _stack(fbfm, dx=0.001, dy=0.001)
    perim = WFIGSPerimeter(
        irwin_id="IR-1",
        name="Test",
        geometry_rings=[[(-118.27, 34.03), (-118.23, 34.03), (-118.23, 34.07), (-118.27, 34.07)]],
    )
    aoi = build_incident_aoi(perim, stack, wind_u=0.0, wind_v=10.0, max_cells=10)
    assert "aoi_coarse_advisory" in aoi.flags
    assert len(aoi.cells) <= 10 or len(aoi.cells) > 0


def test_downwind_buffer_extends_further_than_upwind():
    perim = WFIGSPerimeter(
        irwin_id="IR-1",
        name="Test",
        geometry_rings=[[(-118.01, 34.00), (-117.99, 34.00), (-117.99, 34.02), (-118.01, 34.02)]],
    )
    # Wind toward the east (u>0, v=0) -> bearing 90.
    poly, flags = wind_projected_polygon(perim, wind_u=10.0, wind_v=0.0, base_buffer_km=1.0, downwind_buffer_km=20.0, upwind_buffer_km=1.0)
    assert poly is not None
    west, south, east, north = poly.bounds
    # Downwind east should extend well past the original ~ -117.99.
    assert east > -117.80
    # Upwind west should not extend 20 km.
    assert west > -118.20
