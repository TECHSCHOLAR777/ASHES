"""ASHES copilot HTTP UI. Serves the agentic loop, ActionCards, and ELMFIRE map."""
from __future__ import annotations

import json
import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).resolve().parent / "static"
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"

app = FastAPI(title="ASHES", version="v2-agentic")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

_DEPS_LOCK = threading.Lock()
_DEPS = None
_STORE_LOCK = threading.Lock()
_REPORTS: dict[str, dict[str, Any]] = {}
_LAST_BY_SITE: dict[str, str] = {}
_INCIDENTS: dict[str, dict[str, Any]] = {}


def _get_deps():
    global _DEPS
    with _DEPS_LOCK:
        if _DEPS is None:
            from src.agents.main_agent import MainAgentDeps

            _DEPS = MainAgentDeps()
        return _DEPS


class AskBody(BaseModel):
    lat: float | None = None
    lng: float | None = None
    q: str = "Is this site at risk?"
    name: str = "ask-mode-site"
    site_id: str | None = None
    simulate: bool = False
    ignition_lat: float | None = None
    ignition_lng: float | None = None
    buffer_km: float | None = None


def _store_report(report: dict[str, Any]) -> None:
    card = report.get("action_card") or {}
    card_id = card.get("card_id") or str(uuid.uuid4())
    site = report.get("site") or {}
    site_id = site.get("site_id")
    with _STORE_LOCK:
        _REPORTS[card_id] = report
        if site_id:
            _LAST_BY_SITE[site_id] = card_id
        irwin = ((card.get("incident") or {}).get("irwin_id")) if card else None
        if irwin and report.get("engine", {}).get("field"):
            _INCIDENTS[irwin] = {
                "irwin_id": irwin,
                "incident_name": (card.get("incident") or {}).get("incident_name"),
                "engine": report.get("engine"),
                "map": report.get("map"),
                "card_id": card_id,
            }


def _usable_coords(lat: float | None, lng: float | None) -> bool:
    if lat is None or lng is None:
        return False
    if abs(float(lat)) < 1e-6 and abs(float(lng)) < 1e-6:
        return False
    return 24.0 <= float(lat) <= 50.0 and -126.0 <= float(lng) <= -66.0


def _hydrate_saved_reports() -> None:
    """Only hydrate the Idyllwild showcase file, never the old warehouse 0 h seed."""
    path = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "agentic_live_report.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for key in ("ask", "simulate"):
        report = payload.get(key)
        if not isinstance(report, dict) or not report.get("action_card"):
            continue
        name = str((report.get("site") or {}).get("name") or "")
        if "idyllwild" not in name.lower():
            continue
        _store_report(report)


def _demo_book_report(row: dict[str, Any]) -> dict[str, Any]:
    """Fill the watch-board book so empty sites still show a card. Live Ask/simulate overwrite."""
    card_id = f"demo-{row['site_id']}"
    now = "2026-08-30T18:00:00+00:00"
    site = {
        "site_id": row["site_id"],
        "name": row["name"],
        "lat": row["lat"],
        "lng": row["lng"],
        "mode": "point",
        "geocode_confidence": "high",
        "range_interpolation": False,
    }
    card = {
        "card_id": card_id,
        "generated_at": now,
        "site": site,
        "action": row["action"],
        "eta_hours": row["eta_hours"],
        "eta_sigma_hours": row["eta_sigma_hours"],
        "p_burn_by_T": row["p_burn"],
        "sigma": 0.18,
        "incident": {
            "irwin_id": row["irwin_id"],
            "incident_name": row["incident_name"],
            "acres": row["acres"],
            "containment_pct": row["containment_pct"],
            "dist_perimeter_m": row["dist_perimeter_m"],
            "hours_since_discovery": row["hours_since_discovery"],
        },
        "weather": {
            "red_flag": row["red_flag"],
            "wind_speed_ms": row["wind_speed_ms"],
            "wind_dir_cardinal": row["wind_dir_cardinal"],
            "rh_pct": row["rh_pct"],
            "hrrr_valid_time": now,
        },
        "recommended_actions": row["playbook"],
        "reasons": row["reasons"],
        "flags": row["flags"],
        "spread_field_version": row["spread_field_version"],
        "policy_version": "v2.0.0",
        "model_version": "none",
    }
    return {
        "demo_book": True,
        "site": site,
        "action_card": card,
        "engine": {"engine": "elmfire_2025.0212", "field": None},
        "playbook": row["playbook"],
        "brief": row["brief"],
        "aspects": {},
        "trace": [],
        "parse": {"honesty": row["honesty"]},
        "simulate": False,
    }


def _hydrate_demo_watch_book() -> None:
    rows = [
        {
            "site_id": "site_001",
            "name": "Riverside Warehouse",
            "lat": 33.9806,
            "lng": -117.3755,
            "action": "monitor",
            "eta_hours": 58.0,
            "eta_sigma_hours": 7.0,
            "p_burn": {"24": 0.02, "48": 0.08, "72": 0.14},
            "irwin_id": "IR-RIV-1044",
            "incident_name": "Box Springs",
            "acres": 420.0,
            "containment_pct": 0.15,
            "dist_perimeter_m": 18400.0,
            "hours_since_discovery": 11.0,
            "red_flag": False,
            "wind_speed_ms": 4.1,
            "wind_dir_cardinal": "W",
            "rh_pct": 27.0,
            "spread_field_version": "elmfire_2025.0212:n1:h72:grid160x160",
            "flags": ["perimeter_unofficial"],
            "reasons": ["Perimeter 18.4 km out; keep the watch cadence."],
            "playbook": [
                "Keep the watch cadence on Box Springs.",
                "Responsible land agency (Mireye): local.",
                "Re-ask if the perimeter moves inside 15 km or a Red Flag is issued.",
            ],
            "brief": "Riverside Warehouse is on watch for Box Springs. Engine ETA 58 h.",
            "honesty": "Community pin for Riverside Warehouse (33.9806, -117.3755).",
        },
        {
            "site_id": "site_002",
            "name": "Napa Valley Winery",
            "lat": 38.2975,
            "lng": -122.2869,
            "action": "prepare",
            "eta_hours": 36.0,
            "eta_sigma_hours": 5.0,
            "p_burn": {"24": 0.12, "48": 0.31, "72": 0.48},
            "irwin_id": "IR-NAP-2210",
            "incident_name": "Atlas Grade",
            "acres": 2100.0,
            "containment_pct": 0.08,
            "dist_perimeter_m": 7200.0,
            "hours_since_discovery": 19.0,
            "red_flag": True,
            "wind_speed_ms": 7.4,
            "wind_dir_cardinal": "N",
            "rh_pct": 18.0,
            "spread_field_version": "elmfire_2025.0212:n1:h72:grid192x192",
            "flags": [],
            "reasons": ["ETA 36 h is inside the 24–72 h prepare window."],
            "playbook": [
                "Stage documents and a site contact list.",
                "Confirm road class for egress (Mireye nearest_road_class=primary).",
                "Notify local that the site is in the 24–72 h prepare window.",
            ],
            "brief": "Napa Valley Winery is in the prepare window for Atlas Grade. Engine ETA 36 h. Red Flag is active.",
            "honesty": "Community pin for Napa Valley Winery (38.2975, -122.2869).",
        },
        {
            "site_id": "site_003",
            "name": "Boulder Distribution Center",
            "lat": 40.0150,
            "lng": -105.2705,
            "action": "monitor",
            "eta_hours": 70.0,
            "eta_sigma_hours": 9.0,
            "p_burn": {"24": 0.01, "48": 0.04, "72": 0.09},
            "irwin_id": "IR-BOU-0881",
            "incident_name": "Left Hand",
            "acres": 180.0,
            "containment_pct": 0.22,
            "dist_perimeter_m": 22100.0,
            "hours_since_discovery": 8.0,
            "red_flag": False,
            "wind_speed_ms": 3.6,
            "wind_dir_cardinal": "W",
            "rh_pct": 34.0,
            "spread_field_version": "elmfire_2025.0212:n1:h72:grid140x140",
            "flags": [],
            "reasons": ["Perimeter 22.1 km out; watch Left Hand."],
            "playbook": [
                "Keep the watch cadence on Left Hand.",
                "Responsible land agency (Mireye): USFS.",
            ],
            "brief": "Boulder Distribution Center is on watch for Left Hand. Engine ETA 70 h.",
            "honesty": "Community pin for Boulder Distribution Center (40.0150, -105.2705).",
        },
        {
            "site_id": "site_004",
            "name": "Flagstaff Timber Yard",
            "lat": 35.1983,
            "lng": -111.6513,
            "action": "protect_asset",
            "eta_hours": 9.0,
            "eta_sigma_hours": 2.0,
            "p_burn": {"24": 0.41, "48": 0.67, "72": 0.81},
            "irwin_id": "IR-FLG-3302",
            "incident_name": "Dry Lake",
            "acres": 5600.0,
            "containment_pct": 0.05,
            "dist_perimeter_m": 2100.0,
            "hours_since_discovery": 14.0,
            "red_flag": True,
            "wind_speed_ms": 8.2,
            "wind_dir_cardinal": "SW",
            "rh_pct": 12.0,
            "spread_field_version": "elmfire_2025.0212:n1:h72:grid220x220",
            "flags": [],
            "reasons": ["Perimeter 2.1 km away; protecting the asset."],
            "playbook": [
                "ETA 9 h (engine). Protect the asset.",
                "Contact USFS.",
                "Access: nearest_road_class=secondary.",
                "Water (Mireye): Lake Mary.",
            ],
            "brief": "Flagstaff Timber Yard is protect_asset for Dry Lake. Engine ETA 9 h. Red Flag is active.",
            "honesty": "Community pin for Flagstaff Timber Yard (35.1983, -111.6513).",
        },
        {
            "site_id": "site_005",
            "name": "Bozeman Plant",
            "lat": 45.6770,
            "lng": -111.0429,
            "action": "prepare",
            "eta_hours": 28.0,
            "eta_sigma_hours": 4.0,
            "p_burn": {"24": 0.18, "48": 0.39, "72": 0.55},
            "irwin_id": "IR-BZN-0912",
            "incident_name": "Bridger Foothills",
            "acres": 980.0,
            "containment_pct": 0.12,
            "dist_perimeter_m": 9100.0,
            "hours_since_discovery": 16.0,
            "red_flag": False,
            "wind_speed_ms": 5.0,
            "wind_dir_cardinal": "S",
            "rh_pct": 22.0,
            "spread_field_version": "elmfire_2025.0212:n1:h72:grid176x176",
            "flags": [],
            "reasons": ["ETA 28 h is inside the 24–72 h prepare window."],
            "playbook": [
                "Stage documents and a site contact list.",
                "Confirm road class for egress (Mireye nearest_road_class=tertiary).",
                "Notify local that the site is in the 24–72 h prepare window.",
            ],
            "brief": "Bozeman Plant is in the prepare window for Bridger Foothills. Engine ETA 28 h.",
            "honesty": "Community pin for Bozeman Plant (45.6770, -111.0429).",
        },
    ]
    for row in rows:
        _store_report(_demo_book_report(row))


_hydrate_demo_watch_book()
_hydrate_saved_reports()


def _sse(body: AskBody) -> Any:
    from src.agents.agent_loop import run_agentic
    from src.agents.main_agent import Site

    q: queue.Queue = queue.Queue()
    coords_supplied = _usable_coords(body.lat, body.lng)
    site = Site(
        site_id=body.site_id or f"ask_{uuid.uuid4().hex[:8]}",
        name=body.name,
        lat=float(body.lat) if coords_supplied else 0.0,
        lng=float(body.lng) if coords_supplied else 0.0,
        address=None if coords_supplied else body.q,
    )

    def on_event(ev: dict[str, Any]) -> None:
        q.put(ev)

    def worker() -> None:
        try:
            deps = _get_deps()
            report = run_agentic(
                deps,
                site,
                body.q,
                simulate=body.simulate,
                ignition_lat=body.ignition_lat,
                ignition_lng=body.ignition_lng,
                buffer_km=body.buffer_km,
                coords_supplied=coords_supplied,
                on_event=on_event,
            )
            _store_report(report)
        except Exception as exc:
            q.put({"event": "error", "message": str(exc)})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            try:
                item = q.get(timeout=15)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response

    return Response(status_code=204)


@app.get("/api/health")
def health():
    from src.spread.client import DEFAULT_HOST, DEFAULT_PORT, _port_open

    elmfire = os.environ.get("ELMFIRE_BIN") or ""
    bin_path = os.environ.get("SPREAD_ENGINE_BIN") or ""
    return {
        "ok": True,
        "openai": bool(os.environ.get("OPENAI_KEY")),
        "mireye": bool(os.environ.get("MIREYE_KEY_1") or os.environ.get("MIREYE_KEY_2")),
        "firms": bool(os.environ.get("FIRMS_MAP_KEY")),
        "spread_host": DEFAULT_HOST,
        "spread_port": DEFAULT_PORT,
        "spread_up": _port_open(DEFAULT_HOST, DEFAULT_PORT),
        "elmfire_bin": bool(elmfire and Path(elmfire).exists()),
        "spread_engine_bin": bool(bin_path and Path(bin_path).exists()),
        "engine_required": os.environ.get("SPREAD_ENGINE_REQUIRED", "").lower() in {"1", "true", "yes"},
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    }


@app.get("/api/recent")
def recent():
    with _STORE_LOCK:
        items = []
        for cid, report in list(_REPORTS.items())[-20:]:
            if report.get("demo_book"):
                continue
            card = report.get("action_card") or {}
            items.append(
                {
                    "card_id": cid,
                    "site": report.get("site"),
                    "action": card.get("action"),
                    "eta_hours": card.get("eta_hours"),
                    "spread_field_version": card.get("spread_field_version"),
                }
            )
    return {"cards": list(reversed(items))}


@app.get("/api/sites")
def sites():
    from src.agents.main_agent import load_sites

    rows = []
    for s in load_sites():
        with _STORE_LOCK:
            card_id = _LAST_BY_SITE.get(s.site_id)
            report = _REPORTS.get(card_id) if card_id else None
        card = (report or {}).get("action_card")
        rows.append(
            {
                "site_id": s.site_id,
                "name": s.name,
                "lat": s.lat,
                "lng": s.lng,
                "last_card_id": card_id,
                "action": (card or {}).get("action"),
                "eta_hours": (card or {}).get("eta_hours"),
                "eta_sigma_hours": (card or {}).get("eta_sigma_hours"),
                "sigma": (card or {}).get("sigma"),
                "engine": ((report or {}).get("engine") or {}).get("engine"),
                "dist_perimeter_m": ((card or {}).get("incident") or {}).get("dist_perimeter_m"),
                "incident_name": ((card or {}).get("incident") or {}).get("incident_name"),
                "red_flag": ((card or {}).get("weather") or {}).get("red_flag"),
                "spread_field_version": (card or {}).get("spread_field_version"),
                "flags": (card or {}).get("flags") or [],
            }
        )
    return {"sites": rows}


@app.get("/api/cards/{card_id}")
def get_card(card_id: str):
    with _STORE_LOCK:
        report = _REPORTS.get(card_id)
    if not report:
        raise HTTPException(404, "card not found")
    return report


@app.get("/api/incidents/{irwin_id}")
def get_incident(irwin_id: str):
    with _STORE_LOCK:
        hit = _INCIDENTS.get(irwin_id)
        related = [
            {"card_id": cid, "site": r.get("site"), "action": (r.get("action_card") or {}).get("action")}
            for cid, r in _REPORTS.items()
            if ((r.get("action_card") or {}).get("incident") or {}).get("irwin_id") == irwin_id
        ]
    if not hit:
        raise HTTPException(404, "incident not found")
    hit = dict(hit)
    hit["cards"] = related
    return hit


@app.get("/api/showcase")
def showcase():
    import yaml

    path = Path(__file__).resolve().parent.parent.parent / "config" / "showcase.yaml"
    if not path.exists():
        raise HTTPException(404, "showcase.yaml missing")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


@app.get("/api/policy")
def policy():
    from src.policy.engine import load_policy_config

    return load_policy_config()


@app.get("/api/logs")
def logs(limit: int = 200):
    files = sorted(LOG_DIR.glob("*.jsonl")) if LOG_DIR.exists() else []
    if not files:
        return {"lines": []}
    path = files[-1]
    rows: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(line.rstrip("\n"))
    return {"file": path.name, "lines": rows[-max(1, min(limit, 500)) :]}


@app.post("/api/ask")
def ask(body: AskBody):
    return _sse(body)


@app.post("/api/simulate")
def simulate(body: AskBody):
    body.simulate = True
    if body.ignition_lat is None and body.lat is not None and _usable_coords(body.lat, body.lng):
        # Only default ignition to the site when the caller did not send a separate pin.
        pass
    if body.buffer_km is None:
        body.buffer_km = 8
    return _sse(body)
