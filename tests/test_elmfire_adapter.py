import numpy as np

from spread_service.elmfire_adapter import (
    dead_fuel_moisture,
    p_from_eta,
    rasterize_phi,
    utm_epsg,
    wind_speed_dir_mph,
    write_namelist,
)


def test_utm_and_wind_and_moisture_are_documented_physics():
    assert utm_epsg(39.0, -120.5) == 32610
    assert utm_epsg(34.0, -118.0) == 32611
    ws, wd = wind_speed_dir_mph(0.0, -10.0)  # wind going south → from the north
    assert abs(ws - 10.0 * 2.23694) < 1e-6
    assert abs(wd - 0.0) < 1.0 or abs(wd - 360.0) < 1.0
    m1, m10, m100 = dead_fuel_moisture(20.0)
    assert 3.0 <= m1 < m10 <= m100 <= 30.0


def test_p_burn_is_indicator_of_eta_not_a_fitted_head():
    eta = np.array([[0.0, 24.0], [48.0, np.nan]])
    p72 = p_from_eta(eta, 72.0)
    assert p72[0, 0] == 1.0
    assert p72[0, 1] == 1.0
    assert p72[1, 0] == 1.0
    assert p72[1, 1] == 0.0
    p24 = p_from_eta(eta, 24.0)
    assert p24[0, 1] == 1.0
    assert p24[1, 0] == 0.0


def test_namelist_is_elmfire_not_json(tmp_path):
    path = tmp_path / "elmfire.data"
    write_namelist(path, epsg=32610, cellsize=90.0, xll=0.0, yll=0.0, tstop_s=259200.0)
    text = path.read_text()
    assert "&INPUTS" in text
    assert "WS_AT_10M" in text
    assert "DUMP_TIME_OF_ARRIVAL" in text
    assert "DUMP_BINARY_OUTPUTS" in text
    assert "NUM_IGNITIONS = 0" in text


def test_phi_burns_seed_interior():
    transform = [0.01, 0.0, -120.1, 0.0, -0.01, 40.1]
    rings = [[
        [-120.07, 40.03],
        [-120.03, 40.03],
        [-120.03, 40.07],
        [-120.07, 40.07],
        [-120.07, 40.03],
    ]]
    phi = rasterize_phi(10, 10, transform, rings, "EPSG:4326")
    assert phi.shape == (10, 10)
    assert (phi < 0).any()
    assert (phi > 0).any()


def test_adapter_writes_npz_sidecar_not_nested_grid_json(tmp_path):
    import json
    from pathlib import Path

    from spread_service.elmfire_adapter import write_adapter_outputs
    from spread_service.server import merge_engine_arrays

    eta = np.array([[0.0, 12.0], [np.nan, 80.0]])
    result = {
        "arrival_hours": eta,
        "eta_sigma_hours": np.zeros_like(eta),
        "p_burn_24": p_from_eta(eta, 24.0),
        "p_burn_48": p_from_eta(eta, 48.0),
        "p_burn_72": p_from_eta(eta, 72.0),
        "spread_field_version": "elmfire_2025.0212:n1:h72:grid2x2",
        "engine": "elmfire_2025.0212",
        "n_members": 1,
        "west": -120.0,
        "south": 40.0,
        "east": -119.0,
        "north": 41.0,
        "transform": [0.01, 0.0, -120.0, 0.0, -0.01, 41.0],
        "shape": [2, 2],
    }
    out = tmp_path / "outputs.json"
    write_adapter_outputs(result, out)
    meta = json.loads(out.read_text())
    assert meta["engine"] == "elmfire_2025.0212"
    assert meta["arrival_hours"] is None
    assert Path(meta["arrays_path"]).exists()
    merged = merge_engine_arrays(meta)
    assert merged["arrival_hours"].shape == (2, 2)
    assert merged["p_burn_72"][0, 0] == 1.0
    assert out.stat().st_size < 2000


def test_coarsen_keeps_utm_metres_not_lonlat():
    from rasterio.transform import Affine

    from spread_service.elmfire_adapter import _reproject_to_utm, rasterize_phi

    height = width = 2000
    transform = Affine(0.0005, 0, -106.0, 0, -0.0005, 34.0)
    fbfm = np.full((height, width), 122.0, dtype=np.float32)
    projected, dst_t, dst_crs, cellsize, xll, yll, dh, dw = _reproject_to_utm(
        {"fbfm40": fbfm}, transform, "EPSG:4326", 32613, max_dim=400
    )
    assert dw <= 400 and dh <= 400
    assert cellsize > 50.0
    assert abs(xll) > 10_000
    ring = [[
        [-105.5, 33.5],
        [-105.4, 33.5],
        [-105.4, 33.6],
        [-105.5, 33.6],
        [-105.5, 33.5],
    ]]
    phi = rasterize_phi(dh, dw, dst_t, ring, dst_crs)
    assert (phi < 0).any()
    assert (phi > 0).any()
    assert abs(abs(dst_t.a) - abs(dst_t.e)) < 1e-6


def test_wide_aoi_keeps_square_cells():
    """A 539×800-style rectangle must not ship XDIM != YDIM to Fortran."""
    from rasterio.transform import Affine

    from spread_service.elmfire_adapter import _reproject_to_utm, square_utm_transform

    transform, dw, dh, cell = square_utm_transform(0.0, 0.0, 53_900.0, 80_000.0, max_dim=800)
    assert dw <= 800 and dh <= 800
    assert abs(abs(transform.a) - abs(transform.e)) < 1e-6
    assert abs(cell - abs(transform.a)) < 1e-6

    src_t = Affine(0.0005, 0, -122.6, 0, -0.0005, 43.4)
    fbfm = np.full((400, 1600), 122.0, dtype=np.float32)
    projected, dst_t, _crs, cellsize, _xll, _yll, dh, dw = _reproject_to_utm(
        {"fbfm40": fbfm}, src_t, "EPSG:4326", 32610, max_dim=800
    )
    assert dw <= 800 and dh <= 800
    assert abs(abs(dst_t.a) - abs(dst_t.e)) < 1e-3
    assert projected["fbfm40"].shape == (dh, dw)
    assert cellsize == abs(dst_t.a)


def test_phi_seed_recovers_toa_when_binary_ix_are_zero(tmp_path):
    """Urban pin: fire dies in the seed, toa.bin is zeros, phi < 0 is T=0 arrival."""
    import rasterio
    from rasterio.transform import from_origin

    from spread_service.elmfire_adapter import ElmfireAdapterError, _eta_from_phi_seed, eta_from_elmfire_bin

    work = tmp_path / "job"
    (work / "inputs").mkdir(parents=True)
    (work / "outputs").mkdir()
    phi = np.ones((6, 5), dtype=np.float32)
    phi[2:4, 1:3] = -1.0
    transform = from_origin(0, 6, 30, 30)
    with rasterio.open(
        work / "inputs" / "phi.tif",
        "w",
        driver="GTiff",
        height=6,
        width=5,
        count=1,
        dtype="float32",
        transform=transform,
        crs="EPSG:32611",
        nodata=-9999.0,
    ) as ds:
        ds.write(phi, 1)
    n = np.int32(4)
    zeros2 = np.zeros(4, dtype="<i2")
    zeros4 = np.zeros(4, dtype="<f4")
    payload = (
        _write_fortran_record(int(n).to_bytes(4, "little", signed=True))
        + _write_fortran_record(zeros2.tobytes())
        + _write_fortran_record(zeros2.tobytes())
        + _write_fortran_record(zeros4.tobytes())
        + _write_fortran_record(zeros4.tobytes())
        + _write_fortran_record(zeros4.tobytes())
        + _write_fortran_record(np.zeros(4, dtype="<i1").tobytes())
    )
    (work / "outputs" / "toa_0001_0000001.bin").write_bytes(payload)
    (work / "outputs" / "fire_size_stats.csv").write_text("x\n", encoding="utf-8")

    try:
        eta_from_elmfire_bin(work / "outputs" / "toa_0001_0000001.bin", 6, 5, 259200.0)
        raise AssertionError("zero IX bin should not parse as a spread field")
    except ElmfireAdapterError as exc:
        assert "all 0" in str(exc)

    hours = _eta_from_phi_seed(work, 259200.0)
    assert hours[2, 1] == 0.0
    assert hours[3, 2] == 0.0
    assert not np.isfinite(hours[0, 0])


def test_small_aoi_does_not_oversample_below_landfire_30m():
    from spread_service.elmfire_adapter import square_utm_transform

    transform, dw, dh, cell = square_utm_transform(0.0, 0.0, 4_000.0, 4_000.0, max_dim=800)
    assert cell >= 30.0 - 1e-6
    assert dw <= 140 and dh <= 140
    assert abs(abs(transform.a) - abs(transform.e)) < 1e-6


def _write_fortran_record(payload: bytes) -> bytes:
    n = len(payload)
    return n.to_bytes(4, "little", signed=True) + payload + n.to_bytes(4, "little", signed=True)


def test_binary_toa_recovers_when_geotiff_dump_is_missing(tmp_path):
    """Pin ignitions that die before TSTOP skip time_of_arrival.tif; toa_*.bin still has cells."""
    import struct

    from spread_service.elmfire_adapter import _read_toa, eta_from_elmfire_bin, toa_seconds_to_hours

    height, width = 5, 4
    # Fortran 1-based: IX=2 (col 1), IY=1 (south → numpy row 4)
    ix = np.array([2, 3], dtype="<i2")
    iy = np.array([1, 5], dtype="<i2")
    toa_s = np.array([0.0, 7200.0], dtype="<f4")
    n = np.int32(2)
    payload = (
        _write_fortran_record(struct.pack("<i", int(n)))
        + _write_fortran_record(ix.tobytes())
        + _write_fortran_record(iy.tobytes())
        + _write_fortran_record(toa_s.tobytes())
        + _write_fortran_record(np.zeros(2, dtype="<f4").tobytes())
        + _write_fortran_record(np.zeros(2, dtype="<f4").tobytes())
        + _write_fortran_record(np.zeros(2, dtype="<i1").tobytes())
    )
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "fire_size_stats.csv").write_text("icase,tstop\n", encoding="utf-8")
    bin_path = outputs / "toa_0001_0000001.bin"
    bin_path.write_bytes(payload)

    hours = eta_from_elmfire_bin(bin_path, height, width, 259200.0)
    assert hours[height - 1, 1] == 0.0  # IY=1 south, ignition T=0
    assert abs(hours[0, 2] - 2.0) < 1e-6  # IY=5 north, 7200 s

    recovered = _read_toa(outputs, 259200.0, height, width)
    assert recovered[height - 1, 1] == 0.0
    assert abs(recovered[0, 2] - 2.0) < 1e-6

    zero = toa_seconds_to_hours(np.array([[0.0, -9999.0]]), 72 * 3600)
    assert zero[0, 0] == 0.0
    assert not np.isfinite(zero[0, 1])
