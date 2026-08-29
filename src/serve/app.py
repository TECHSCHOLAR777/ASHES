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
    lat: float
    lng: float
    q: str = "Is this site at risk?"
    name: str = "ask-mode-site"
    site_id: str | None = None
    simulate: bool = False
    ignition_lat: float | None = None
    ignition_lng: float | None = None


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


def _hydrate_saved_reports() -> None:
    path = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "agentic_live_report.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for key in ("ask", "simulate"):
        report = payload.get(key)
        if isinstance(report, dict) and report.get("action_card"):
            _store_report(report)


_hydrate_saved_reports()


def _sse(body: AskBody) -> Any:
    from src.agents.agent_loop import run_agentic
    from src.agents.main_agent import Site

    q: queue.Queue = queue.Queue()
    site = Site(
        site_id=body.site_id or f"ask_{uuid.uuid4().hex[:8]}",
        name=body.name,
        lat=body.lat,
        lng=body.lng,
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
                "y_hat": (card or {}).get("y_hat"),
                "sigma": (card or {}).get("sigma"),
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
    if body.ignition_lat is None:
        body.ignition_lat = body.lat
    if body.ignition_lng is None:
        body.ignition_lng = body.lng
    return _sse(body)
