from __future__ import annotations

import json

import numpy as np
from rasterio.transform import from_origin

from src.agents.agent_loop import run_agentic
from src.agents.agent_tools import execute_tool, handle_apply_policy
from src.agents.agent_session import AgentSession
from src.agents.main_agent import Site
from src.agents.spread_viz import downsample_field
from src.clients.landfire import LandfireStack
from src.spread.client import SpreadField
from tests.test_main_agent import make_deps


class _Fn:
    def __init__(self, name: str, arguments: str = "{}"):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, id: str, name: str, arguments: str = "{}"):
        self.id = id
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, tool_calls=None, content: str = ""):
        self.tool_calls = tool_calls
        self.content = content


class _Choice:
    def __init__(self, message):
        self.message = message


class _Resp:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class FakeOpenAI:
    """Minimal chat.completions.create stand-in."""

    def __init__(self, turns: list[list[_TC] | None]):
        self.turns = list(turns)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if not self.turns:
            return _Resp(_Msg(None, "done"))
        calls = self.turns.pop(0)
        if calls is None:
            return _Resp(_Msg(None, "done"))
        return _Resp(_Msg(calls))


def test_agentic_loop_calls_tools_and_policy_owns_action(tmp_path, monkeypatch, mocker):
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    deps = make_deps(tmp_path, monkeypatch, mocker)
    site = Site(site_id="s_agent", name="Test Site", lat=34.0, lng=-118.0)
    client = FakeOpenAI(
        [
            [
                _TC("1", "nws_alerts"),
                _TC("2", "firms_hotspots"),
                _TC("3", "wfigs_incidents"),
                _TC("4", "wfigs_perimeters"),
                _TC("5", "hrrr_weather"),
            ],
            [_TC("6", "mireye_fetch", json.dumps({"roles": ["A", "B", "G"]}))],
            [_TC("7", "apply_policy")],
            [_TC("8", "draft_watch_note")],
            [_TC("9", "commit_report")],
        ]
    )
    report = run_agentic(deps, site, "Is the site at risk?", client=client)
    card = report["action_card"]
    assert card is not None
    assert card["action"] == "no_action"  # mocked empty live signals
    assert card["y_hat"] is None
    assert card["model_version"] == "none"
    tools = [t["tool"] for t in report["trace"]]
    assert "parse_place" in tools
    assert "nws_alerts" in tools
    assert "mireye_fetch" in tools
    assert "apply_policy" in tools
    assert "commit_report" in tools
    assert "model_infer" not in tools
    assert report["brief"]
    assert report["parse"]
    assert "community pin" in (report["parse"].get("honesty") or "").lower()
    assert all(t["tool"] != "unknown" for t in report["trace"])


def test_commit_backfills_mireye_when_model_skips_it(tmp_path, monkeypatch, mocker):
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    deps = make_deps(tmp_path, monkeypatch, mocker)
    mocker.patch.object(
        deps.mireye,
        "fetch",
        return_value={"ndvi_current": 0.42, "aspect_degrees": 180.0, "aspect_cardinal": "S"},
    )
    site = Site(site_id="s_w", name="Test Site", lat=34.0, lng=-118.0)
    client = FakeOpenAI(
        [
            [_TC("1", "nws_alerts"), _TC("2", "hrrr_weather")],
            [_TC("3", "commit_report")],
        ]
    )
    report = run_agentic(deps, site, "risk?", client=client)
    tools = [t["tool"] for t in report["trace"]]
    assert "mireye_fetch" in tools
    assert report["aspects"]


def test_mireye_fetch_falls_back_to_core_fields_on_402(tmp_path, monkeypatch, mocker):
    from src.agents.agent_tools import handle_mireye_fetch
    from src.clients.mireye import MireyeRequestFailed

    monkeypatch.delenv("OPENAI_KEY", raising=False)
    deps = make_deps(tmp_path, monkeypatch, mocker)
    mocker.patch.object(
        deps.mireye,
        "fetch",
        side_effect=[
            MireyeRequestFailed("HTTP 402 credits_exhausted"),
            {"aspect_degrees": 180.0},
            MireyeRequestFailed("HTTP 402 credits_exhausted"),
        ],
    )
    session = AgentSession(deps=deps, site=Site("s1", "T", 34.0, -118.0), question="q")
    out = handle_mireye_fetch(session, {"roles": ["A", "B"]})
    assert out["ok"] is True
    assert session.raw_w["aspect_degrees"] == 180.0
    assert "degraded" in session.flags


def test_llm_cannot_pass_an_action_into_policy(tmp_path, monkeypatch, mocker):
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    deps = make_deps(tmp_path, monkeypatch, mocker)
    session = AgentSession(deps=deps, site=Site("s1", "T", 34.0, -118.0), question="q")
    execute_tool(session, "nws_alerts", {})
    execute_tool(session, "firms_hotspots", {})
    execute_tool(session, "wfigs_incidents", {})
    execute_tool(session, "wfigs_perimeters", {})
    execute_tool(session, "hrrr_weather", {})
    out = handle_apply_policy(session, {"action": "evacuate_site"})
    assert out["action"] == "no_action"
    assert out["action"] != "evacuate_site"


def test_simulate_ignition_uses_engine_sample(tmp_path, monkeypatch, mocker):
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    deps = make_deps(tmp_path, monkeypatch, mocker)
    fbfm = np.full((8, 8), 122, dtype=np.int16)
    transform = from_origin(-118.03, 34.03, 0.01, 0.01)
    nan = np.full(fbfm.shape, np.nan)
    stack = LandfireStack(
        fbfm40=fbfm,
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
        west=-118.03,
        south=34.03 - 8 * 0.01,
        east=-118.03 + 8 * 0.01,
        north=34.03,
        resample_m=90,
        source_url="test",
    )
    mocker.patch.object(deps.landfire, "fetch_aoi", return_value=stack)
    arrival = np.full((8, 8), 12.0)
    p72 = np.full((8, 8), 0.7)
    field = SpreadField(
        incident_id="SIMULATED",
        spread_field_version="elmfire_2025.0212:test",
        engine="elmfire_2025.0212",
        n_members=1,
        arrival_hours=arrival,
        eta_sigma_hours=np.full((8, 8), 3.0),
        p_burn_24=np.full((8, 8), 0.4),
        p_burn_48=np.full((8, 8), 0.6),
        p_burn_72=p72,
        transform=list(transform)[:6],
        west=-118.03,
        south=34.03 - 8 * 0.01,
        east=-118.03 + 8 * 0.01,
        north=34.03,
    )
    spread = mocker.patch("src.agents.agent_tools.spread_run", return_value=field)
    site = Site(site_id="s_sim", name="Sim", lat=34.0, lng=-118.0)
    client = FakeOpenAI(
        [
            [_TC("a", "hrrr_weather")],
            [_TC("b", "simulate_ignition", json.dumps({"lat": 34.0, "lng": -118.0}))],
            [_TC("c", "mireye_fetch")],
            [_TC("d", "commit_report")],
        ]
    )
    report = run_agentic(deps, site, "Simulate ignition", simulate=True, client=client)
    assert report["engine"]["engine"] == "elmfire_2025.0212"
    sample = report["engine"]["sample"]
    assert sample is not None
    assert sample["p_burn_72"] == 0.7
    assert spread.call_args.kwargs["ensemble"]["n_members"] == 7
    assert spread.call_args.kwargs["reuse"] is False
    assert report["engine"]["front"]["eta_hours"] == 12.0
    assert report["action_card"]["spread_field_version"] == "elmfire_2025.0212:test"
    assert report["action_card"]["incident"]["irwin_id"] == "SIMULATED"
    assert report["action_card"]["y_hat"] is None
    assert report["action_card"]["model_version"] == "none"
    assert report["playbook"]
    assert report["engine"]["field"]["width"] >= 1


def test_downsample_keeps_nulls():
    arrival = np.array([[1.0, np.nan], [72.0, 10.0]])
    p72 = np.array([[0.2, np.nan], [0.9, 0.1]])
    field = SpreadField(
        incident_id="x",
        spread_field_version="v",
        engine="e",
        n_members=1,
        arrival_hours=arrival,
        eta_sigma_hours=np.zeros((2, 2)),
        p_burn_24=np.zeros((2, 2)),
        p_burn_48=np.zeros((2, 2)),
        p_burn_72=p72,
        transform=[0.01, 0, -118, 0, -0.01, 34],
        west=-118.0,
        south=33.9,
        east=-117.9,
        north=34.03,
    )
    viz = downsample_field(field, max_dim=8)
    assert viz["arrival_hours"][0][1] is None
    assert viz["p_burn_72"][1][0] == 0.9
    assert viz["max_eta_hours"] == 72.0
    assert viz["horizon_hours"] == 72.0
    assert viz["n_reached"] == 3
    assert viz["burned_bbox"] is not None
    assert viz["n_members"] == 1
    assert viz["p_burn_field_max"]["72"] == 0.9


def test_zero_p_burn_at_community_is_kept_and_front_is_elsewhere():
    transform = [0.01, 0.0, -118.0, 0.0, -0.01, 34.0]
    arrival = np.array([[0.0, np.nan], [np.nan, np.nan]])
    zeros = np.zeros((2, 2))
    p72 = np.array([[1.0, 0.0], [0.0, 0.0]])
    field = SpreadField(
        incident_id="x",
        spread_field_version="v",
        engine="e",
        n_members=7,
        arrival_hours=arrival,
        eta_sigma_hours=zeros,
        p_burn_24=zeros,
        p_burn_48=zeros,
        p_burn_72=p72,
        transform=transform,
        west=-118.0,
        south=33.98,
        east=-117.98,
        north=34.0,
    )
    unreached = field.sample(33.995, -117.985)
    assert unreached.eta_hours is None
    assert unreached.p_burn_72 == 0.0
    front = field.nearest_reached(33.995, -117.985)
    assert front["dist_m"] is not None and front["dist_m"] > 0
    assert front["p_burn_72"] == 1.0
    assert front["eta_hours"] == 0.0
    assert field.p_burn_field_max()["72"] == 1.0


def test_health_endpoint():
    from fastapi.testclient import TestClient
    from src.serve.app import app

    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "openai" in body
    assert "mireye" in body


def test_ui_index_served():
    from fastapi.testclient import TestClient
    from src.serve.app import app

    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert b"ASHES" in r.content
    assert b"app.js?v=pburn4" in r.content
    css = client.get("/assets/styles.css")
    assert css.status_code == 200
    ico = client.get("/favicon.ico")
    assert ico.status_code == 204


def test_openai_tools_have_no_hit_model():
    from src.agents.agent_tools import OPENAI_TOOLS, HANDLERS

    names = [t["function"]["name"] for t in OPENAI_TOOLS]
    assert "model_infer" not in names
    assert "model_infer" not in HANDLERS
    assert "parse_place" in names
    assert "list_suppression_steps" in names
    assert "notify_ops" in names
    assert "apply_policy" in names


def test_extract_place_and_honesty():
    from src.agents.place_parse import extract_place_heuristic, honesty_line

    assert extract_place_heuristic("Is Idyllwild at risk this week?") == "Idyllwild"
    line = honesty_line(
        {"used_supplied_coords": True, "lat": 33.7435, "lng": -116.735, "place": "Idyllwild"}
    )
    assert "community pin" in line.lower()
    assert "Idyllwild" in line
    geo = honesty_line(
        {
            "used_supplied_coords": False,
            "geocoded": True,
            "place": "Idyllwild",
            "lat": 33.7435,
            "lng": -116.735,
            "confidence": "high",
        }
    )
    assert "geocoded" in geo.lower()
    assert "community pin" in geo.lower()


def test_parse_place_geocodes_town_name(tmp_path, monkeypatch, mocker):
    from src.agents.agent_tools import handle_parse_place
    from src.clients.mireye import GeoPoint

    monkeypatch.delenv("OPENAI_KEY", raising=False)
    deps = make_deps(tmp_path, monkeypatch, mocker)
    mocker.patch.object(
        deps.mireye,
        "geocode",
        return_value=GeoPoint(lat=33.7435, lng=-116.735, confidence="high", range_interpolation=False),
    )
    session = AgentSession(
        deps=deps,
        site=Site("s1", "ask-mode-site", 0.0, 0.0),
        question="Is Idyllwild at risk if the chaparral ignites?",
        coords_supplied=False,
    )
    out = handle_parse_place(session, {})
    assert out["ok"] is True
    assert out["geocoded"] is True
    assert session.site.lat == 33.7435
    assert "Idyllwild" in out["honesty"]
    assert "community pin" in out["honesty"].lower()
    assert session.site.name == "Idyllwild"


def test_bucket_skill_sets_playbook(tmp_path, monkeypatch, mocker):
    from src.agents.bucket_skills import handle_suppression_steps
    from src.policy.engine import PolicyResult

    deps = make_deps(tmp_path, monkeypatch, mocker)
    session = AgentSession(deps=deps, site=Site("s1", "T", 33.74, -116.73), question="q")
    session.policy_result = PolicyResult("protect_asset", ["close"], "v2.0.0", [])
    session.raw_w = {"surface_management_agency": "USFS", "nearest_road_class": "residential"}
    out = handle_suppression_steps(session, {})
    assert out["ok"] is True
    assert session.playbook_steps
    assert any("USFS" in s for s in session.playbook_steps)


def test_ask_body_allows_town_only_and_showcase():
    from fastapi.testclient import TestClient
    from src.serve.app import AskBody, app

    body = AskBody(q="Idyllwild")
    assert body.lat is None
    assert body.lng is None
    client = TestClient(app)
    r = client.get("/api/showcase")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "idyllwild_chaparral"
    assert data["simulate"] is True
    assert data["ignition_lat"] == 33.744
    assert abs(data["lat"] - 33.7461) < 1e-4
    js = client.get("/assets/app.js")
    assert js.status_code == 200
    assert b"Load Idyllwild case" in js.content
    assert b"Engine case unfolding" in js.content
    assert b"SHOWCASE_FALLBACK" in js.content
    assert b"y_hat" not in js.content
    assert b"if (next > maxH) next = 0" not in js.content
    assert b'id="play">Play</button>' in js.content
    assert b"idyllwildCase" in js.content
    assert b"P(burn) at the community pin" in js.content
    assert b"0.00 is engine output, not a missing value" in js.content
    assert b"mountPlayback(ev.field, $(\"#playback\"), { autoplay: true })" in js.content
    html = client.get("/")
    assert b"engine clock" in html.content
    css = client.get("/assets/styles.css")
    assert b".pburn-val" in css.content


def test_idyllwild_p_burn_zeros_are_engine_output():
    from fastapi.testclient import TestClient
    from src.serve.app import app

    client = TestClient(app)
    idy = next(
        (c for c in client.get("/api/recent").json()["cards"] if (c.get("site") or {}).get("name") == "Idyllwild"),
        None,
    )
    assert idy, "Idyllwild live report should hydrate"
    report = client.get(f"/api/cards/{idy['card_id']}").json()
    card = report["action_card"]
    p = card["p_burn_by_T"]
    assert p["24"] == 0.0
    assert p["48"] == 0.0
    assert p["72"] == 0.0
    assert card["eta_hours"] is None
    field = (report.get("engine") or {}).get("field") or {}
    assert (field.get("n_reached") or 0) > 0
    assert not (report.get("aspects") or {}), "hydrate must not invent Mireye aspects"


def test_watch_book_is_seeded_for_the_board():
    from fastapi.testclient import TestClient
    from src.serve.app import app

    client = TestClient(app)
    sites = {s["name"]: s for s in client.get("/api/sites").json()["sites"]}
    assert sites["Riverside Warehouse"]["action"] == "monitor"
    assert sites["Napa Valley Winery"]["action"] == "prepare"
    assert sites["Flagstaff Timber Yard"]["action"] == "protect_asset"
    assert sites["Bozeman Plant"]["eta_hours"] == 28.0
    assert sites["Boulder Distribution Center"]["incident_name"] == "Left Hand"
    assert sites["Flagstaff Timber Yard"]["p_burn_by_T"]["72"] == 0.81
    recent_names = [(c.get("site") or {}).get("name") for c in client.get("/api/recent").json()["cards"]]
    assert "Riverside Warehouse" not in recent_names
