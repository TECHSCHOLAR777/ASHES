#!/usr/bin/env python3
"""Live agentic e2e for the Idyllwild chaparral showcase.

Community pin is the town. Ignition is west in burnable shrub so ELMFIRE can
leave the seed. Writes data/models/agentic_live_report.json (no secrets).
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.agents.agent_loop import run_agentic  # noqa: E402
from src.agents.main_agent import MainAgentDeps, Site  # noqa: E402


ACTIONS = {"monitor", "prepare", "protect_asset", "evacuate_site", "inspect_after", "no_action"}
OUT = Path("data/models/agentic_live_report.json")
SUMMARY = Path("data/models/agentic_live_summary.json")

TOWN = {"lat": 33.7461, "lng": -116.7139, "name": "Idyllwild"}
IGNITION = {"lat": 33.7440, "lng": -116.7320}
BUFFER_KM = 8.0
Q = (
    "Idyllwild is the community. Simulate ignition in the chaparral west of town "
    "and report whether the town is in play. Use live feeds, Mireye aspects, and ELMFIRE."
)


def _run() -> dict:
    deps = MainAgentDeps()
    site = Site(
        site_id=f"live_{uuid.uuid4().hex[:8]}",
        name=TOWN["name"],
        lat=TOWN["lat"],
        lng=TOWN["lng"],
    )
    return run_agentic(
        deps,
        site,
        Q,
        simulate=True,
        ignition_lat=IGNITION["lat"],
        ignition_lng=IGNITION["lng"],
        buffer_km=BUFFER_KM,
        coords_supplied=True,
    )


def _aspect_values(report: dict) -> dict:
    out = {}
    for role, block in (report.get("aspects") or {}).items():
        fields = block.get("fields") if isinstance(block, dict) else None
        if not fields:
            continue
        vals = {}
        for name, meta in fields.items():
            if isinstance(meta, dict):
                if meta.get("value") is not None:
                    vals[name] = meta["value"]
            elif meta is not None:
                vals[name] = meta
        if vals:
            out[role] = vals
    return out


def _slim(report: dict | None) -> dict | None:
    if not report:
        return None
    card = report.get("action_card") or {}
    eng = report.get("engine") or {}
    sample = eng.get("sample") or {}
    field = eng.get("field") or {}
    return {
        "action": card.get("action"),
        "incident": (card.get("incident") or {}).get("incident_name"),
        "irwin_id": (card.get("incident") or {}).get("irwin_id"),
        "dist_perimeter_m": (card.get("incident") or {}).get("dist_perimeter_m"),
        "spread_field_version": card.get("spread_field_version"),
        "engine": eng.get("engine"),
        "eta_hours": card.get("eta_hours"),
        "p_burn_by_T": card.get("p_burn_by_T"),
        "site_inside_aoi": sample.get("inside_aoi"),
        "sample": sample,
        "n_reached": field.get("n_reached"),
        "max_eta_hours": field.get("max_eta_hours"),
        "tools": [t.get("tool") for t in report.get("trace") or []],
        "aspects": _aspect_values(report),
        "brief_replaced": report.get("brief_replaced"),
        "flags": card.get("flags"),
        "simulate": report.get("simulate"),
        "parse": report.get("parse"),
        "playbook": report.get("playbook"),
        "brief": (report.get("brief") or "")[:800],
        "site": report.get("site"),
        "ignition": (report.get("map") or {}).get("ignition"),
    }


def _ok(report: dict) -> list[str]:
    problems = []
    card = report.get("action_card")
    if not card:
        problems.append("no ActionCard")
        return problems
    if card.get("action") not in ACTIONS:
        problems.append(f"bad action {card.get('action')}")
    tools = [t["tool"] for t in report.get("trace") or []]
    for required in ("nws_alerts", "mireye_fetch", "simulate_ignition", "apply_policy", "commit_report"):
        if required not in tools:
            problems.append(f"missing tool {required}")
    eng = (report.get("engine") or {}).get("engine")
    if not eng:
        problems.append("simulator produced no engine id (LANDFIRE/spread degraded)")
    field = (report.get("engine") or {}).get("field") or {}
    n_reached = field.get("n_reached")
    if n_reached is not None and n_reached <= 1:
        problems.append(f"engine did not leave the seed (n_reached={n_reached})")
    site = report.get("site") or {}
    ign = (report.get("map") or {}).get("ignition") or {}
    if site.get("lat") and ign.get("lat") and abs(site["lat"] - ign["lat"]) < 1e-4:
        problems.append("ignition is on the town pin")
    return problems


def main() -> int:
    if not os.environ.get("OPENAI_KEY"):
        print("OPENAI_KEY missing", file=sys.stderr)
        return 2
    if not (os.environ.get("MIREYE_KEY_1") or os.environ.get("MIREYE_KEY_2")):
        print("MIREYE_KEY missing", file=sys.stderr)
        return 2

    report = _run()
    problems = _ok(report)
    payload = {"ask": None, "simulate": report}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, default=str)
    OUT.write_text(raw[:800_000], encoding="utf-8")
    summary = {
        "simulate": _slim(report),
        "problems": problems,
        "out": str(OUT),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(
        {
            "action": (report.get("action_card") or {}).get("action"),
            "eta_hours": (report.get("action_card") or {}).get("eta_hours"),
            "engine": (report.get("engine") or {}).get("engine"),
            "n_reached": ((report.get("engine") or {}).get("field") or {}).get("n_reached"),
            "max_eta_hours": ((report.get("engine") or {}).get("field") or {}).get("max_eta_hours"),
            "tools": [t["tool"] for t in report.get("trace") or []],
            "problems": problems,
            "summary": str(SUMMARY),
        },
        indent=2,
    ))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
