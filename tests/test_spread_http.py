"""HTTP round-trip against the out-of-process spread_run contract."""
from __future__ import annotations

import socket
import threading
from http.server import ThreadingHTTPServer

import numpy as np
from rasterio.transform import from_origin

from src.clients.landfire import LandfireStack
from src.geometry.aoi import point_geometry
from src.spread.client import spread_run
from spread_service.server import Handler


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_spread_run_http_roundtrip(monkeypatch):
    monkeypatch.delenv("SPREAD_ENGINE_BIN", raising=False)
    monkeypatch.delenv("SPREAD_ENGINE_REQUIRED", raising=False)
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        n = 5
        fbfm = np.full((n, n), 122, dtype=np.int16)
        transform = from_origin(-118.01, 34.02, 0.0008, 0.0008)
        nan = np.zeros((n, n), dtype=np.float64)
        stack = LandfireStack(
            fbfm40=fbfm,
            cc_pct=nan, ch_m=nan, cbd_kg_m3=nan, cbh_m=nan, elev_m=nan,
            slope_deg=nan, aspect_deg=nan,
            transform=transform, crs="EPSG:4326", nodata=-9999,
            layer_list=["LF2024_FBFM40"], vintage="LF2024",
            west=-118.01, south=34.02 - n * 0.0008, east=-118.01 + n * 0.0008, north=34.02,
            resample_m=90, source_url="test",
        )
        aoi = point_geometry(34.0172, -118.0072)
        field = spread_run(
            incident_id="http-test",
            landfire=stack,
            aoi=aoi,
            weather={"wind_u": 4.0, "wind_v": 0.0, "rh_pct": 20.0},
            perimeter_rings=[[
                [-118.0076, 34.0168], [-118.0068, 34.0168],
                [-118.0068, 34.0176], [-118.0076, 34.0176], [-118.0076, 34.0168],
            ]],
            ignition_points=[{"lat": 34.0172, "lng": -118.0072}],
            ensemble={"n_members": 3, "seed": 1},
            host="127.0.0.1",
            port=port,
            reuse=False,
        )
        assert field.engine == "rothermel_huygens_v1"
        assert field.n_members == 3
        sample = field.sample(34.0172, -118.0072)
        assert sample.spread_field_version.startswith("rothermel_huygens_v1")
        assert sample.inside_aoi is True
        # Ignition cell should have a finite arrival, not a placeholder constant sigma-only.
        assert sample.eta_hours is not None
        assert sample.p_burn_72 is not None
        assert sample.p_burn_72 >= 0.0
    finally:
        server.shutdown()
        server.server_close()
