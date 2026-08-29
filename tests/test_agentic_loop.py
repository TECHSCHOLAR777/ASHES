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
            [_TC("7", "model_infer")],
            [_TC("8", "apply_policy")],
            [_TC("9", "commit_report")],
        ]
    )
    report = run_agentic(deps, site, "Is the site at risk?", client=client)
    card = report["action_card"]
    assert card is not None
    assert card["action"] == "no_action"  # mocked empty live signals
    tools = [t["tool"] for t in report["trace"]]
    assert "nws_alerts" in tools
    assert "mireye_fetch" in tools
    assert "apply_policy" in tools
    assert "commit_report" in tools
    assert report["brief"]
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
    mocker.patch("src.agents.agent_tools.spread_run", return_value=field)
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
    assert report["action_card"]["spread_field_version"] == "elmfire_2025.0212:test"
    assert report["action_card"]["incident"]["irwin_id"] == "SIMULATED"
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
    css = client.get("/assets/styles.css")
    assert css.status_code == 200
    ico = client.get("/favicon.ico")
    assert ico.status_code == 204
