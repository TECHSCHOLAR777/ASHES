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
        return_value={"housing_units_density_per_km2": 2000.0, "nearest_road_class": "local"},
    )

    card = build_action_card(deps, site)

    assert card.action in ("protect_asset", "evacuate_site")
    assert card.incident.acres == 5000.0  # copied verbatim from WFIGS, never invented
    assert card.incident.containment_pct == 0.1


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
