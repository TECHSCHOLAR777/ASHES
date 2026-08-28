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
