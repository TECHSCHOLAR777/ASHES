"""LANDFIRE client tests. Live LFPS is mocked; raster parsing uses a real in-memory GeoTIFF
with the band-description shape observed live on 2026-08-27 (`LF2024_FBFM40_CONUS`)."""
from __future__ import annotations

import io

import httpx
import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.clients.landfire import (
    LANDFIREClient,
    LandfireProduct,
    LandfireRequestFailed,
    NON_BURNABLE_FBFM40,
    _band_matches,
    pick_conus_layer,
    resolve_engine_layers,
)


def _products_payload():
    return {
        "products": [
            {
                "productName": "40 Scott and Burgan Fire Behavior Fuel Models",
                "theme": "Fuels",
                "layerName": "LF2025_FBFM40",
                "acronym": "FBFM40",
                "version": "LF2025",
                "conus": True,
                "ak": True,
                "hi": False,
                "prvi": False,
                "geoAreas": "SW, NW",
            },
            {
                "productName": "40 Scott and Burgan Fire Behavior Fuel Models",
                "theme": "Seasonal Fuels",
                "layerName": "LF2025_FBFM40_SP26",
                "acronym": "FBFM40",
                "version": "LF2025",
                "conus": True,
                "ak": False,
                "hi": False,
                "prvi": False,
                "geoAreas": "",
            },
            {
                "productName": "40 Scott and Burgan Fire Behavior Fuel Models",
                "theme": "Fuels",
                "layerName": "LF2024_FBFM40",
                "acronym": "FBFM40",
                "version": "LF2024",
                "conus": True,
                "ak": True,
                "hi": True,
                "prvi": True,
                "geoAreas": "All",
            },
            {
                "productName": "Forest Canopy Cover",
                "theme": "Fuels",
                "layerName": "LF2024_CC",
                "acronym": "CC",
                "version": "LF2024",
                "conus": True,
                "geoAreas": "All",
            },
            {
                "productName": "Forest Canopy Height",
                "theme": "Fuels",
                "layerName": "LF2024_CH",
                "acronym": "CH",
                "version": "LF2024",
                "conus": True,
                "geoAreas": "All",
            },
            {
                "productName": "Forest Canopy Bulk Density",
                "theme": "Fuels",
                "layerName": "LF2024_CBD",
                "acronym": "CBD",
                "version": "LF2024",
                "conus": True,
                "geoAreas": "All",
            },
            {
                "productName": "Forest Canopy Base Height",
                "theme": "Fuels",
                "layerName": "LF2024_CBH",
                "acronym": "CBH",
                "version": "LF2024",
                "conus": True,
                "geoAreas": "All",
            },
            {
                "productName": "Elevation",
                "theme": "Topographic",
                "layerName": "LF2020_Elev",
                "acronym": "Elev",
                "version": "LF2020",
                "conus": True,
                "geoAreas": "All",
            },
            {
                "productName": "Slope Degrees",
                "theme": "Topographic",
                "layerName": "LF2020_SlpD",
                "acronym": "SlpD",
                "version": "LF2020",
                "conus": True,
                "geoAreas": "All",
            },
            {
                "productName": "Aspect",
                "theme": "Topographic",
                "layerName": "LF2020_Asp",
                "acronym": "Asp",
                "version": "LF2020",
                "conus": True,
                "geoAreas": "All",
            },
        ]
    }


def _as_products():
    return [
        LandfireProduct(
            product_name=p["productName"],
            theme=p["theme"],
            layer_name=p["layerName"],
            acronym=p["acronym"],
            version=p["version"],
            conus=p["conus"],
            geo_areas=p.get("geoAreas", ""),
        )
        for p in _products_payload()["products"]
    ]


def make_response(json_body=None, status_code=200, content=None, url="https://lfps.usgs.gov/api/job/submit"):
    request = httpx.Request("POST", url)
    if content is not None:
        return httpx.Response(status_code, content=content, request=request)
    return httpx.Response(status_code, json=json_body, request=request)


def test_band_name_matches_live_conus_suffix():
    assert _band_matches("LF2024_FBFM40_CONUS", "LF2024_FBFM40")
    assert _band_matches("LF2024_FBFM40", "LF2024_FBFM40")
    assert not _band_matches("LF2023_FBFM40_CONUS", "LF2024_FBFM40")


def test_picker_prefers_full_conus_over_newer_regional():
    picked = pick_conus_layer(_as_products(), "FBFM40")
    assert picked is not None
    assert picked.layer_name == "LF2024_FBFM40"


def test_picker_skips_seasonal_suffix_layers():
    products = _as_products()
    # Drop the All-coverage LF2024 so the seasonal LF2025 is the only leftover 2025
    # candidate besides regional LF2025_FBFM40. Regional still beats seasonal.
    products = [p for p in products if p.layer_name != "LF2024_FBFM40"]
    picked = pick_conus_layer(products, "FBFM40")
    assert picked is not None
    assert picked.layer_name == "LF2025_FBFM40"


def test_resolve_engine_layers_uses_live_catalog_names():
    layers = resolve_engine_layers(_as_products())
    assert layers == [
        "LF2024_FBFM40",
        "LF2024_CC",
        "LF2024_CH",
        "LF2024_CBD",
        "LF2024_CBH",
        "LF2020_Elev",
        "LF2020_SlpD",
        "LF2020_Asp",
    ]


def test_urban_and_water_are_non_burnable():
    assert 91 in NON_BURNABLE_FBFM40
    assert 98 in NON_BURNABLE_FBFM40
    assert 122 not in NON_BURNABLE_FBFM40


def _synthetic_geotiff_bytes() -> bytes:
    """8 bands named the way a live LFPS 4326 job actually names them."""
    height, width = 4, 5
    transform = from_origin(-118.26, 34.06, 0.004, 0.005)
    descriptions = [
        "LF2024_FBFM40_CONUS",
        "LF2024_CC_CONUS",
        "LF2024_CH_CONUS",
        "LF2024_CBD_CONUS",
        "LF2024_CBH_CONUS",
        "LF2020_Elev_CONUS",
        "LF2020_SlpD_CONUS",
        "LF2020_Asp_CONUS",
    ]
    arrays = [
        np.full((height, width), 122, dtype=np.int16),  # burnable shrub
        np.full((height, width), 40, dtype=np.int16),  # 40% CC
        np.full((height, width), 150, dtype=np.int16),  # 15.0 m CH
        np.full((height, width), 12, dtype=np.int16),  # 0.12 kg/m3 CBD
        np.full((height, width), 20, dtype=np.int16),  # 2.0 m CBH
        np.full((height, width), 400, dtype=np.int16),
        np.full((height, width), 8, dtype=np.int16),
        np.full((height, width), 180, dtype=np.int16),
    ]
    arrays[0][0, 0] = 91  # urban corner
    arrays[0][0, 1] = 98  # water
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=8,
        dtype="int16",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as dst:
        for i, (arr, desc) in enumerate(zip(arrays, descriptions), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, desc)
    return buf.getvalue()


def test_parse_geotiff_uses_live_band_names_and_scales(tmp_path):
    client = LANDFIREClient(cache_dir=tmp_path)
    layers = resolve_engine_layers(_as_products())
    stack = client._parse_geotiff(
        _synthetic_geotiff_bytes(),
        layers=layers,
        west=-118.26,
        south=34.04,
        east=-118.24,
        north=34.06,
        resample_m=90,
        source_url="https://example.invalid/out.zip",
    )
    assert stack.ch_m[1, 1] == 15.0
    assert stack.cbh_m[1, 1] == 2.0
    assert abs(stack.cbd_kg_m3[1, 1] - 0.12) < 1e-9
    assert stack.cc_pct[1, 1] == 40.0
    assert int(stack.fbfm40[1, 1]) == 122
    mask = stack.burnable_mask()
    assert mask[0, 0] is np.False_ or mask[0, 0] == False
    assert mask[0, 1] == False
    assert mask[1, 1] == True
    client.close()


def test_fetch_aoi_happy_path_mocked(tmp_path, mocker):
    client = LANDFIREClient(cache_dir=tmp_path, poll_seconds=0.0)
    tif = _synthetic_geotiff_bytes()
    import zipfile

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("job.tif", tif)
    zip_bytes = zbuf.getvalue()

    def fake_request(method, url, **kwargs):
        if url.endswith("/api/products"):
            return make_response(_products_payload(), url=url)
        if url.endswith("/api/job/submit"):
            return make_response(
                {
                    "jobId": "abc-123",
                    "status": "Pending",
                    "job": {"aoiSize": 20, "geoArea": "SW", "layerList": kwargs.get("json", {}).get("Layer_List")},
                },
                url=url,
            )
        if "/api/job/status" in url:
            return make_response(
                {
                    "jobId": "abc-123",
                    "status": "Succeeded",
                    "queuePosition": -1,
                    "outputFile": "https://lfps.usgs.gov/out.zip",
                    "messages": [],
                },
                url=url,
            )
        if url.endswith("/out.zip"):
            return make_response(content=zip_bytes, url=url)
        raise AssertionError(f"unexpected url {url}")

    mocker.patch.object(client._client, "request", side_effect=fake_request)
    # httpx.Client.get/post call request(); patch get/post instead for clarity.
    mocker.patch.object(
        client._client,
        "get",
        side_effect=lambda url, **kw: fake_request("GET", url, **kw),
    )
    mocker.patch.object(
        client._client,
        "post",
        side_effect=lambda url, **kw: fake_request("POST", url, **kw),
    )

    stack = client.fetch_aoi(-118.26, 34.04, -118.24, 34.06, site_id="s1")
    assert stack.vintage == "LF2024"
    assert stack.width == 5
    assert int(stack.fbfm40[1, 1]) == 122
    # Second call hits the cache (no extra HTTP).
    stack2 = client.fetch_aoi(-118.26, 34.04, -118.24, 34.06, site_id="s1")
    assert int(stack2.fbfm40[1, 1]) == 122
    client.close()


def test_submit_http_failure_raises(tmp_path, mocker):
    client = LANDFIREClient(cache_dir=tmp_path)
    mocker.patch.object(
        client._client, "get", return_value=make_response(_products_payload(), url="https://lfps.usgs.gov/api/products")
    )
    mocker.patch.object(client._client, "post", side_effect=httpx.ConnectError("down"))
    try:
        client.fetch_aoi(-118.26, 34.04, -118.24, 34.06)
        assert False, "expected LandfireRequestFailed"
    except LandfireRequestFailed:
        pass
    client.close()
