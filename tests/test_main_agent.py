from src.agents.main_agent import MainAgentDeps, Site, build_action_card
from src.cache.w_cache import WCache
from src.clients.firms import FIRMSResult
from src.clients.hrrr import HRRRWeather
from src.clients.wfigs import WFIGSIncident, WFIGSPerimeter
from src.state.site_state import SiteStateStore


def make_deps(tmp_path, monkeypatch, mocker):
    monkeypatch.setenv("MIREYE_KEY_1", "dummy1")
    monkeypatch.setenv("MIREYE_KEY_2", "dummy2")
    monkeypatch.setenv("MIREYE_KEY_3", "dummy3")
    monkeypatch.delenv("OPENAI_KEY", raising=False)

    deps = MainAgentDeps()
    deps.w_cache = WCache(tmp_path / "w_cache.db")
    deps.site_state = SiteStateStore(tmp_path / "site_state.db")

    mocker.patch.object(deps.nws, "get_cap_alerts", return_value=[])
    mocker.patch.object(deps.nws, "get_spc_outlook", return_value=None)
    mocker.patch.object(
        deps.firms, "get_hotspots", return_value=FIRMSResult([], [], [], 0, 0, 0, 0.0, unavailable=False)
    )
    mocker.patch.object(deps.wfigs, "get_incidents", return_value=[])
    mocker.patch.object(deps.wfigs, "get_perimeters", return_value=[])
    mocker.patch.object(
        deps.hrrr,
        "get_weather",
        return_value=HRRRWeather(1.0, 1.0, 1.4142, "NW", 25.0, 30.0, "2026-08-27T00:00:00Z"),
    )
    mocker.patch.object(deps.mireye, "fetch", return_value={})
    mocker.patch.object(deps.landfire, "fetch_aoi", side_effect=RuntimeError("landfire mocked off"))
    mocker.patch.object(deps.osm, "route_to_nearest_station", return_value=None)
    return deps


def test_no_signal_site_gives_no_action(tmp_path, monkeypatch, mocker):
    deps = make_deps(tmp_path, monkeypatch, mocker)
    site = Site(site_id="s1", name="Test Site", lat=34.0, lng=-118.0)

    card = build_action_card(deps, site)

    assert card.action == "no_action"
    assert card.sigma is not None
    assert card.model_version
    assert card.policy_version


def test_active_incident_close_by_escalates(tmp_path, monkeypatch, mocker):
    deps = make_deps(tmp_path, monkeypatch, mocker)
    site = Site(site_id="s2", name="Close Site", lat=34.0, lng=-118.0)

    incident = WFIGSIncident(
        irwin_id="IR-1", name="Big Fire", acres=5000.0, containment_pct=0.1,
        discovery_datetime="2026-08-26T00:00:00Z", lat=34.001, lng=-118.001,
    )
    mocker.patch.object(deps.wfigs, "get_incidents", return_value=[incident])
    # Perimeter ring sits ~200m from the site so dist_perim_m is small regardless of the
    # (untrained/dummy) model's y_hat - this test exercises the distance-driven escalation
    # branch of the policy table, not the model score branch.
    perimeter = WFIGSPerimeter(
        irwin_id="IR-1",
        name="Big Fire",
        geometry_rings=[[(-118.002, 34.001), (-118.000, 34.001), (-118.000, 34.003), (-118.002, 34.003)]],
    )
    mocker.patch.object(deps.wfigs, "get_perimeters", return_value=[perimeter])
    mocker.patch.object(
        deps.firms, "get_hotspots", return_value=FIRMSResult([], [], [1], 1, 1, 1, 15.0, unavailable=False)
    )
    mocker.patch.object(
        deps.mireye,
        "fetch",
        return_value={"housing_units_density_per_km2": 2000.0, "nearest_road_class": "unclassified"},
    )

    card = build_action_card(deps, site)

    assert card.action in ("protect_asset", "evacuate_site")
    assert card.incident.acres == 5000.0  # copied verbatim from WFIGS, never invented
    assert card.incident.containment_pct == 0.1


def test_spread_sample_populates_eta_on_card(tmp_path, monkeypatch, mocker):
    from rasterio.transform import from_origin

    import numpy as np

    from src.clients.landfire import LandfireStack
    from src.spread.client import SpreadField

    deps = make_deps(tmp_path, monkeypatch, mocker)
    site = Site(site_id="s2", name="Close Site", lat=34.0, lng=-118.0)
    incident = WFIGSIncident(
        irwin_id="IR-1", name="Big Fire", acres=5000.0, containment_pct=0.1,
        discovery_datetime="2026-08-26T00:00:00Z", lat=34.001, lng=-118.001,
    )
    perimeter = WFIGSPerimeter(
        irwin_id="IR-1",
        name="Big Fire",
        geometry_rings=[[(-118.02, 33.99), (-117.98, 33.99), (-117.98, 34.02), (-118.02, 34.02)]],
    )
    mocker.patch.object(deps.wfigs, "get_incidents", return_value=[incident])
    mocker.patch.object(deps.wfigs, "get_perimeters", return_value=[perimeter])

    fbfm = np.full((6, 6), 122, dtype=np.int16)
    transform = from_origin(-118.03, 34.03, 0.01, 0.01)
    nan = np.full(fbfm.shape, np.nan)
    stack = LandfireStack(
        fbfm40=fbfm,
        cc_pct=nan, ch_m=nan, cbd_kg_m3=nan, cbh_m=nan, elev_m=nan, slope_deg=nan, aspect_deg=nan,
        transform=transform, crs="EPSG:4326", nodata=-9999,
        layer_list=["LF2024_FBFM40"], vintage="LF2024",
        west=-118.03, south=34.03 - 6 * 0.01, east=-118.03 + 6 * 0.01, north=34.03,
        resample_m=90, source_url="test",
    )
    mocker.patch.object(deps.landfire, "fetch_aoi", return_value=stack)
    field = SpreadField(
        incident_id="IR-1",
        spread_field_version="rothermel_huygens_v1:n7:h72:grid6x6",
        engine="rothermel_huygens_v1",
        n_members=7,
        arrival_hours=np.full((6, 6), 5.0),
        eta_sigma_hours=np.full((6, 6), 1.5),
        p_burn_24=np.full((6, 6), 0.7),
        p_burn_48=np.full((6, 6), 0.85),
        p_burn_72=np.full((6, 6), 0.9),
        transform=list(transform)[:6],
        west=stack.west, south=stack.south, east=stack.east, north=stack.north,
    )
    mocker.patch("src.agents.main_agent.spread_run", return_value=field)

    card = build_action_card(deps, site)

    assert card.eta_hours == 5.0
    assert card.eta_sigma_hours == 1.5
    assert card.spread_field_version == "rothermel_huygens_v1:n7:h72:grid6x6"
    assert card.p_burn_by_T is not None
    assert card.baseline_y == 0.9
    assert card.policy_version == "v2.0.0"


def test_never_invents_acres_when_no_incident(tmp_path, monkeypatch, mocker):
    deps = make_deps(tmp_path, monkeypatch, mocker)
    site = Site(site_id="s3", name="No Fire Site", lat=34.0, lng=-118.0)

    card = build_action_card(deps, site)

    assert card.incident.acres is None
    assert card.incident.irwin_id is None


def test_mireye_fetch_failure_degrades_not_crashes(tmp_path, monkeypatch, mocker):
    deps = make_deps(tmp_path, monkeypatch, mocker)
    mocker.patch.object(deps.mireye, "fetch", side_effect=RuntimeError("mireye down"))
    site = Site(site_id="s4", name="Degraded Site", lat=34.0, lng=-118.0)

    card = build_action_card(deps, site)

    assert "degraded" in card.flags
    assert card.action  # still produced a card
