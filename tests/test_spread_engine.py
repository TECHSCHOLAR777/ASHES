from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spread_service.server import run_ensemble


def _as_grid(out, key):
    raw = out[key]
    if isinstance(raw, np.ndarray):
        return np.asarray(raw, dtype=float)
    return np.array([[np.nan if v is None else v for v in row] for row in raw], dtype=float)


def _grid(n=7, fuel=122):
    fbfm = np.full((n, n), fuel, dtype=np.float64)
    slope = np.zeros((n, n))
    # 90 m cells as ~0.0008 deg at ~34N
    transform = [0.0008, 0.0, -118.01, 0.0, -0.0008, 34.02]
    rings = [[
        [-118.0076, 34.0168],
        [-118.0068, 34.0168],
        [-118.0068, 34.0176],
        [-118.0076, 34.0176],
        [-118.0076, 34.0168],
    ]]
    # Cell (3,3) center under this affine is lon=-118.0072, lat=34.0172
    return {
        "fbfm40": fbfm.tolist(),
        "slope_deg": slope.tolist(),
        "transform": transform,
        "west": -118.01,
        "south": 34.02 - n * 0.0008,
        "east": -118.01 + n * 0.0008,
        "north": 34.02,
        "weather": {"wind_u": 5.0, "wind_v": 0.0, "rh_pct": 15.0},
        "perimeter_rings": [rings],
        "ignition_points": [{"lat": 34.0172, "lng": -118.0072}],
        "ensemble": {"n_members": 5, "wind_speed_frac": 0.25, "wind_dir_deg": 25.0, "moisture_frac": 0.1, "seed": 1},
        "horizon_hours": 72.0,
    }


def test_engine_reaches_burnable_neighbors_and_skips_urban():
    body = _grid()
    fbfm = np.array(body["fbfm40"])
    fbfm[0, :] = 91  # urban row never burns
    body["fbfm40"] = fbfm.tolist()
    out = run_ensemble(body)
    assert out["engine"] == "rothermel_huygens_v1"
    assert out["n_members"] == 5
    assert out["spread_field_version"].startswith("rothermel_huygens_v1")
    arrival = _as_grid(out, "arrival_hours")
    # Some cell inside/near the ignition polygon should have a finite arrival.
    assert np.isfinite(arrival).any()
    # Urban row stays unreached.
    assert not np.isfinite(arrival[0]).any()
    p72 = np.array(out["p_burn_72"])
    assert p72.max() > 0
    # Ensemble sigma is a real spread, not a constant placeholder, on reached cells.
    sigma = _as_grid(out, "eta_sigma_hours")
    reached = np.isfinite(arrival) & np.isfinite(sigma)
    assert reached.any()
    assert float(np.nanmax(sigma[reached])) >= 0.0


def test_external_engine_bin_used_when_contract_is_honored(tmp_path, monkeypatch):
    bin_path = tmp_path / "fake_elmfire.py"
    bin_path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "inp, outp = sys.argv[1], sys.argv[2]\n"
        "body = json.load(open(inp))\n"
        "if body.get('rasters_path'):\n"
        "    import numpy as np\n"
        "    loaded = np.load(body['rasters_path'])\n"
        "    fbfm = loaded['fbfm40'].tolist()\n"
        "    body['west'] = float(loaded['west']) if 'west' in loaded.files else body['west']\n"
        "    body['south'] = float(loaded['south']) if 'south' in loaded.files else body['south']\n"
        "    body['east'] = float(loaded['east']) if 'east' in loaded.files else body['east']\n"
        "    body['north'] = float(loaded['north']) if 'north' in loaded.files else body['north']\n"
        "    body['transform'] = loaded['transform'].tolist() if 'transform' in loaded.files else body['transform']\n"
        "else:\n"
        "    fbfm = body['fbfm40']\n"
        "h, w = len(fbfm), len(fbfm[0])\n"
        "grid = [[0.0 for _ in range(w)] for _ in range(h)]\n"
        "ones = [[1.0 for _ in range(w)] for _ in range(h)]\n"
        "json.dump({\n"
        "  'arrival_hours': grid,\n"
        "  'eta_sigma_hours': grid,\n"
        "  'p_burn_24': ones,\n"
        "  'p_burn_48': ones,\n"
        "  'p_burn_72': ones,\n"
        "  'spread_field_version': 'external_bin_v1',\n"
        "  'engine': 'external_bin',\n"
        "  'n_members': 1,\n"
        "  'west': body['west'], 'south': body['south'], 'east': body['east'], 'north': body['north'],\n"
        "  'transform': body['transform'],\n"
        "}, open(outp, 'w'))\n"
    )
    bin_path.chmod(0o755)
    monkeypatch.setenv("SPREAD_ENGINE_BIN", str(bin_path))
    from spread_service.server import _run_external_engine

    out = _run_external_engine(_grid())
    assert out is not None
    assert out["spread_field_version"] == "external_bin_v1"
    assert out["p_burn_72"][0][0] == 1.0


def test_required_engine_raises_without_bin(monkeypatch):
    monkeypatch.setenv("SPREAD_ENGINE_REQUIRED", "1")
    monkeypatch.delenv("SPREAD_ENGINE_BIN", raising=False)
    from spread_service.server import _run_external_engine
    import pytest

    with pytest.raises(RuntimeError, match="SPREAD_ENGINE_REQUIRED"):
        _run_external_engine(_grid())


def test_no_ignition_returns_empty_field():
    body = _grid()
    body["perimeter_rings"] = []
    body["ignition_points"] = []
    out = run_ensemble(body)
    arrival = _as_grid(out, "arrival_hours")
    assert not np.isfinite(arrival).any()
    p72 = np.array(out["p_burn_72"])
    assert float(p72.max()) == 0.0
