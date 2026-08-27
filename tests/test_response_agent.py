from src.agents.main_agent import MainAgentDeps, Site
from src.agents.response_agent import run_response_agent
from src.cache.w_cache import WCache
from src.schemas.action_card import ActionCard, IncidentInfo, SiteRef, WeatherInfo
from src.state.site_state import SiteStateStore


def make_deps(tmp_path, monkeypatch, mocker, seed_roles: dict[str, dict] | None = None):
    monkeypatch.setenv("MIREYE_KEY_1", "dummy1")
    monkeypatch.setenv("MIREYE_KEY_2", "dummy2")
    monkeypatch.setenv("MIREYE_KEY_3", "dummy3")
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

    deps = MainAgentDeps()
    deps.w_cache = WCache(tmp_path / "w_cache.db")
    deps.site_state = SiteStateStore(tmp_path / "site_state.db")

    for role, fields in (seed_roles or {}).items():
        deps.w_cache.put("s1", role, fields, {})

    mocker.patch.object(deps.mireye, "fetch", return_value={})  # role J fetch: nothing extra
    mocker.patch.object(deps.usgs, "get_gage_discharge", return_value=_fake_gage())
    return deps


class _FakeGage:
    def __init__(self, discharge_cfs=None, discharge_class="unknown"):
        self.gage_id = None
        self.discharge_cfs = discharge_cfs
        self.discharge_class = discharge_class
        self.fetched_at = "2026-08-27T00:00:00Z"


def _fake_gage():
    return _FakeGage()


def _action_card() -> ActionCard:
    return ActionCard(
        card_id="c1",
        generated_at="2026-08-27T00:00:00Z",
        site=SiteRef(site_id="s1", name="Test Site", lat=34.0, lng=-118.0),
        action="protect_asset",
        sigma=0.2,
        baseline_y=0.5,
        incident=IncidentInfo(),
        weather=WeatherInfo(),
        model_version="v0",
        policy_version="v1.0.0",
    )


def test_d3_drought_appends_low_flow_note(tmp_path, monkeypatch, mocker):
    deps = make_deps(
        tmp_path, monkeypatch, mocker,
        seed_roles={
            "C": {"drought_category": "D3"},
            "I": {"nearest_usgs_gage_id": "01234567"},
        },
    )
    site = Site(site_id="s1", name="Test Site", lat=34.0, lng=-118.0)

    card = run_response_agent(deps, site, _action_card())

    stream_sources = [w for w in card.water_sources if w.type == "stream"]
    assert len(stream_sources) == 1
    assert stream_sources[0].note is not None
    assert "D3" in stream_sources[0].note


def test_critical_habitat_true_sets_retardant_restricted(tmp_path, monkeypatch, mocker):
    deps = make_deps(
        tmp_path, monkeypatch, mocker,
    )
    mocker.patch.object(
        deps.mireye,
        "fetch",
        return_value={"intersects_critical_habitat": True, "critical_habitat_species": "spotted owl"},
    )
    site = Site(site_id="s1", name="Test Site", lat=34.0, lng=-118.0)

    card = run_response_agent(deps, site, _action_card())

    habitat_constraints = [c for c in card.environmental_constraints if c.type == "critical_habitat"]
    assert len(habitat_constraints) == 1
    assert habitat_constraints[0].constraint == "retardant_restricted"
    assert habitat_constraints[0].species_or_designation == "spotted owl"


def test_hazmat_at_400m_is_critical_priority(tmp_path, monkeypatch, mocker):
    deps = make_deps(
        tmp_path, monkeypatch, mocker,
        seed_roles={"H": {"nearest_ust_facility_distance_m": 400.0}},
    )
    site = Site(site_id="s1", name="Test Site", lat=34.0, lng=-118.0)

    card = run_response_agent(deps, site, _action_card())

    ust_sites = [h for h in card.hazmat_sites if h.type == "UST"]
    assert len(ust_sites) == 1
    assert ust_sites[0].priority == "critical"


def test_hazmat_beyond_5km_excluded():
    from src.agents.response_agent import _build_hazmat_sites, _load_policy_cfg

    raw_w = {"nearest_ust_facility_distance_m": 6000.0}
    sites = _build_hazmat_sites(raw_w, _load_policy_cfg())
    assert sites == []


def test_protected_area_sets_coordinate_first(tmp_path, monkeypatch, mocker):
    deps = make_deps(tmp_path, monkeypatch, mocker)
    mocker.patch.object(
        deps.mireye,
        "fetch",
        return_value={
            "intersects_protected_area": True,
            "protected_area_designation": "Wilderness Area",
            "protected_area_manager": "USFS",
        },
    )
    site = Site(site_id="s1", name="Test Site", lat=34.0, lng=-118.0)

    card = run_response_agent(deps, site, _action_card())

    protected = [c for c in card.environmental_constraints if c.type == "protected_area"]
    assert len(protected) == 1
    assert protected[0].constraint == "coordinate_first"
    assert protected[0].manager == "USFS"


def test_response_card_never_invents_missing_fields(tmp_path, monkeypatch, mocker):
    deps = make_deps(tmp_path, monkeypatch, mocker)
    site = Site(site_id="s1", name="Test Site", lat=34.0, lng=-118.0)

    card = run_response_agent(deps, site, _action_card())

    assert card.water_sources == []
    assert card.hazmat_sites == []
    assert card.responsible_agency == "local"  # explicit fallback, not fabricated agency name
